"""Deepgram nova-3 и nova-2 на том же FLEURS-ru, та же нормализация."""
import json
import os
import statistics
import time

import jiwer
import requests
from datasets import Audio, load_dataset

from stt_bench import duration_sec, normalize

KEY = os.environ["DG"]
LIMIT = int(os.environ.get("LIMIT", "150"))
MODELS = ["nova-3", "nova-2"]


def transcribe(model: str, wav: bytes) -> tuple[str, float]:
    t0 = time.time()
    r = requests.post(
        "https://api.deepgram.com/v1/listen",
        params={"model": model, "language": "ru", "punctuate": "true", "smart_format": "false"},
        headers={"Authorization": f"Token {KEY}", "Content-Type": "audio/wav"},
        data=wav, timeout=180,
    )
    dt = time.time() - t0
    r.raise_for_status()
    return r.json()["results"]["channels"][0]["alternatives"][0]["transcript"], dt


stream = load_dataset("google/fleurs", "ru_ru", split="test", streaming=True)
stream = stream.cast_column("audio", Audio(decode=False))
samples = []
for item in stream:
    samples.append({"wav": item["audio"]["bytes"], "ref": item.get("transcription") or ""})
    if len(samples) >= LIMIT:
        break
print(f"взято {len(samples)} сэмплов", flush=True)

report = {}
for model in MODELS:
    refs, hyps, lats, errors = [], [], [], 0
    for i, s in enumerate(samples, 1):
        try:
            text, dt = transcribe(model, s["wav"])
        except Exception as exc:
            errors += 1
            if errors <= 3:
                print(f"  ошибка #{i}: {type(exc).__name__}: {str(exc)[:120]}", flush=True)
            continue
        refs.append(normalize(s["ref"]))
        hyps.append(normalize(text))
        lats.append(dt)
        if i % 50 == 0:
            print(f"  {model} {i}/{len(samples)} WER {jiwer.wer(refs, hyps)*100:.2f}%", flush=True)
    audio_sec = sum(duration_sec(s["wav"]) for s in samples[:len(hyps)])
    report[model] = {
        "wer": round(jiwer.wer(refs, hyps) * 100, 2),
        "cer": round(jiwer.cer(refs, hyps) * 100, 2),
        "ok": len(hyps), "failed": errors,
        "latency_median_s": round(statistics.median(lats), 2),
        "latency_p90_s": round(sorted(lats)[int(len(lats) * 0.9)], 2),
        "rtf": round(sum(lats) / audio_sec, 3),
    }
    print(f"ИТОГО {model}: {json.dumps(report[model], ensure_ascii=False)}", flush=True)

with open("deepgram150.json", "w", encoding="utf-8") as fh:
    json.dump(report, fh, ensure_ascii=False, indent=2)
print(json.dumps(report, ensure_ascii=False, indent=2))
