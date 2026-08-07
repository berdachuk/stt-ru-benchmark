"""Прогон одного движка по набору и запись ПОСТРОЧНОГО результата в JSONL.

Разделение намеренное: этот скрипт только собирает сырые транскрипты, метрики
считает analyze.py. Так любой может пересчитать WER своей нормализацией, не
переспрашивая API и не веря нашим агрегатам.

Движки:
  whisper-1, whisper-podlodka-turbo — OpenAI-совместимый API (переменная K)
  gigaam-v3                         — GigaAM, русская ASR (переменная RPA)
  nova-2, nova-3                    — Deepgram (переменная DG)
  speechcore                        — асинхронный пайплайн upload → poll (SC)

Пример:
  K=sk-... python harness/run_bench.py --engine whisper-1 --limit 150
  DG=...   python harness/run_bench.py --engine nova-2 --limit 150
"""
from __future__ import annotations

import argparse
import io
import json
import os
import time
from pathlib import Path

import requests
import soundfile as sf
from datasets import Audio, load_dataset

DATASET = "google/fleurs"
CONFIG = "ru_ru"

OPENAI_BASE = os.environ.get("HUB_URL", "https://api.neuraldeep.ru/v1")
SPEECHCORE_BASE = os.environ.get("SC_URL", "https://speechcore.neuraldeep.ru/api")
DEEPGRAM_URL = "https://api.deepgram.com/v1/listen"

OPENAI_ENGINES = ("whisper-1", "whisper-podlodka-turbo")
# GigaAM — русская ASR SberDevices, обучена на русском целенаправленно, а не как
# один из ста языков. Живёт на отдельном OpenAI-совместимом эндпоинте.
RPA_BASE = os.environ.get("RPA_URL", "https://private.rpa.icu/v1")
RPA_ENGINES = ("gigaam-v3",)
DEEPGRAM_ENGINES = ("nova-2", "nova-3")


def duration_sec(wav: bytes) -> float:
    info = sf.info(io.BytesIO(wav))
    return info.frames / info.samplerate


def call_openai(engine: str, wav: bytes) -> str:
    r = requests.post(
        f"{OPENAI_BASE}/audio/transcriptions",
        headers={"Authorization": f"Bearer {os.environ['K']}"},
        files={"file": ("sample.wav", wav, "audio/wav")},
        data={"model": engine, "language": "ru", "response_format": "json"},
        timeout=300,
    )
    r.raise_for_status()
    return r.json().get("text") or ""


def call_rpa(engine: str, wav: bytes) -> str:
    r = requests.post(
        f"{RPA_BASE}/audio/transcriptions",
        headers={"Authorization": f"Bearer {os.environ['RPA']}"},
        files={"file": ("sample.wav", wav, "audio/wav")},
        data={"model": engine},
        timeout=300,
    )
    r.raise_for_status()
    return r.json().get("text") or ""


def call_deepgram(engine: str, wav: bytes) -> str:
    r = requests.post(
        DEEPGRAM_URL,
        params={"model": engine, "language": "ru", "punctuate": "true", "smart_format": "false"},
        headers={"Authorization": f"Token {os.environ['DG']}", "Content-Type": "audio/wav"},
        data=wav, timeout=300,
    )
    r.raise_for_status()
    return r.json()["results"]["channels"][0]["alternatives"][0]["transcript"]


def call_speechcore(engine: str, wav: bytes) -> str:
    """⚠️ Задержка тут включает загрузку, очередь и поллинг — она НЕ сравнима
    с одиночным POST у остальных движков."""
    head = {"Authorization": f"Bearer {os.environ['SC']}"}
    up = requests.post(
        f"{SPEECHCORE_BASE}/upload", headers=head,
        files={"file": ("sample.wav", wav, "audio/wav")},
        params={"model": "large-v3", "language": "ru", "diarize": "false"},
        timeout=300,
    )
    up.raise_for_status()
    task_id = up.json()["task_id"]

    deadline = time.time() + 600
    while time.time() < deadline:
        time.sleep(2)
        st = requests.get(f"{SPEECHCORE_BASE}/transcriptions/{task_id}/status",
                          headers=head, timeout=60)
        st.raise_for_status()
        status = (st.json().get("status") or "").lower()
        if status in ("completed", "done", "success"):
            break
        if status in ("failed", "error"):
            raise RuntimeError(f"задача упала: {status}")
    else:
        raise TimeoutError("не дождались завершения")

    res = requests.get(f"{SPEECHCORE_BASE}/transcriptions/{task_id}", headers=head, timeout=60)
    res.raise_for_status()
    body = res.json()
    text = body.get("text") or body.get("transcription") or ""
    if not text and isinstance(body.get("segments"), list):
        text = " ".join(seg.get("text", "") for seg in body["segments"])
    return text


def pick_caller(engine: str):
    if engine in OPENAI_ENGINES:
        return call_openai
    if engine in RPA_ENGINES:
        return call_rpa
    if engine in DEEPGRAM_ENGINES:
        return call_deepgram
    if engine == "speechcore":
        return call_speechcore
    raise SystemExit(f"неизвестный движок: {engine}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    call = pick_caller(args.engine)
    out_path = Path(args.out or f"results/raw/{args.engine}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stream = load_dataset(DATASET, CONFIG, split=args.split, streaming=True)
    stream = stream.cast_column("audio", Audio(decode=False))

    written = failed = 0
    started = time.time()
    with out_path.open("w", encoding="utf-8") as fh:
        # Шапка: без неё через месяц не понять, на какой версии всё считалось.
        fh.write(json.dumps({
            "_meta": {
                "engine": args.engine, "dataset": f"{DATASET}/{CONFIG}", "split": args.split,
                "limit": args.limit, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "endpoint": {"whisper-1": OPENAI_BASE, "whisper-podlodka-turbo": OPENAI_BASE,
                             "nova-2": DEEPGRAM_URL, "nova-3": DEEPGRAM_URL,
                             "gigaam-v3": RPA_BASE,
                             "speechcore": SPEECHCORE_BASE}[args.engine],
            }
        }, ensure_ascii=False) + "\n")

        for idx, item in enumerate(stream):
            if idx >= args.limit:
                break
            wav = item["audio"]["bytes"]
            ref = item.get("transcription") or item.get("raw_transcription") or ""
            t0 = time.time()
            try:
                hyp = call(args.engine, wav)
                error = None
            except Exception as exc:
                hyp, error = "", f"{type(exc).__name__}: {str(exc)[:200]}"
                failed += 1
            fh.write(json.dumps({
                "idx": idx, "duration_s": round(duration_sec(wav), 3),
                "ref_raw": ref, "hyp_raw": hyp,
                "latency_s": round(time.time() - t0, 3), "error": error,
            }, ensure_ascii=False) + "\n")
            written += 1
            if written % 25 == 0:
                print(f"  {written}/{args.limit}  ошибок {failed}  "
                      f"{time.time() - started:.0f}с", flush=True)

    print(f"{args.engine}: записано {written}, ошибок {failed} → {out_path}")


if __name__ == "__main__":
    main()
