"""WhisperX-пайплайн на том же FLEURS-ru: upload → poll → текст.

Отличие от whisper-1: тут полный пайплайн — VAD, чанкование, выравнивание,
а не голый вызов модели. Именно его видит юзер speechcore.neuraldeep.ru.

⚠️ Каждый сэмпл = одна транскрипция из дневной квоты аккаунта.
"""
import json
import os
import statistics
import time

import jiwer
import requests
from datasets import Audio, load_dataset

from stt_bench import duration_sec, normalize

BASE = os.environ.get("SC_URL", "https://speechcore.neuraldeep.ru/api")
TOKEN = os.environ["SC"]
LIMIT = int(os.environ.get("LIMIT", "150"))
H = {"Authorization": f"Bearer {TOKEN}"}


def run_one(wav: bytes, idx: int) -> tuple[str, float]:
    t0 = time.time()
    up = requests.post(
        f"{BASE}/upload", headers=H,
        files={"file": (f"fleurs_{idx}.wav", wav, "audio/wav")},
        params={"model": "large-v3", "language": "ru", "diarize": "false"},
        timeout=180,
    )
    up.raise_for_status()
    task_id = up.json()["task_id"]

    while time.time() - t0 < 600:
        time.sleep(2)
        st = requests.get(f"{BASE}/transcriptions/{task_id}/status", headers=H, timeout=60)
        st.raise_for_status()
        data = st.json()
        status = (data.get("status") or "").lower()
        if status in ("completed", "done", "success"):
            break
        if status in ("failed", "error"):
            raise RuntimeError(f"задача упала: {data.get('error') or status}")
    else:
        raise TimeoutError("не дождались завершения")

    res = requests.get(f"{BASE}/transcriptions/{task_id}", headers=H, timeout=60)
    res.raise_for_status()
    body = res.json()
    text = body.get("text") or body.get("transcription") or ""
    if not text and isinstance(body.get("segments"), list):
        text = " ".join(seg.get("text", "") for seg in body["segments"])
    return text, time.time() - t0


stream = load_dataset("google/fleurs", "ru_ru", split="test", streaming=True)
stream = stream.cast_column("audio", Audio(decode=False))
samples = []
for item in stream:
    samples.append({"wav": item["audio"]["bytes"], "ref": item.get("transcription") or ""})
    if len(samples) >= LIMIT:
        break
print(f"взято {len(samples)} сэмплов", flush=True)

refs, hyps, lats, errors = [], [], [], 0
for i, s in enumerate(samples, 1):
    try:
        text, dt = run_one(s["wav"], i)
    except Exception as exc:
        errors += 1
        print(f"  ошибка #{i}: {type(exc).__name__}: {str(exc)[:160]}", flush=True)
        if errors >= 5 and not hyps:
            break
        continue
    refs.append(normalize(s["ref"]))
    hyps.append(normalize(text))
    lats.append(dt)
    if i % 10 == 0:
        print(f"  {i}/{len(samples)} WER {jiwer.wer(refs, hyps)*100:.2f}%", flush=True)

if hyps:
    audio_sec = sum(duration_sec(s["wav"]) for s in samples[:len(hyps)])
    report = {
        "wer": round(jiwer.wer(refs, hyps) * 100, 2),
        "cer": round(jiwer.cer(refs, hyps) * 100, 2),
        "ok": len(hyps), "failed": errors,
        "latency_median_s": round(statistics.median(lats), 2),
        "latency_p90_s": round(sorted(lats)[int(len(lats) * 0.9)], 2),
        "rtf_end_to_end": round(sum(lats) / audio_sec, 3),
    }
    with open("speechcore.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
else:
    print(f"ни одного успешного результата, ошибок: {errors}")
