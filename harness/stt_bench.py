"""WER/CER наших STT-моделей на русском — через боевой API, как их видит юзер.

Гоняем FLEURS-ru (test) через OpenAI-совместимый API:
  whisper-1               — Whisper large-v3 на собственных GPU
  whisper-podlodka-turbo  — русский файн-тюн Whisper-turbo

Через API, а не локально на GPU, специально: не надо занимать GPU, и
измеряется ровно то, что получает клиент, вместе с сетью и очередью.

Нормализация текста — как принято для WER-замеров
(lowercase, ё→е, снятие пунктуации), иначе сравнивались бы знаки препинания.

Запуск:  K=<sk-...> python stt_bench.py [--limit 150]
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import statistics
import time

import jiwer
import requests
import soundfile as sf
from datasets import Audio, load_dataset

BASE = os.environ.get("HUB_URL", "https://api.neuraldeep.ru/v1")  # любой OpenAI-совместимый
MODELS = ["whisper-1", "whisper-podlodka-turbo"]

_PUNCT_RE = re.compile(r"[^\wа-яёА-ЯЁ\s]", re.UNICODE)


def normalize(text: str) -> str:
    """Как в проде: регистр, ё→е, без пунктуации, одинарные пробелы."""
    if not text:
        return ""
    text = text.lower().replace("ё", "е")
    text = _PUNCT_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def duration_sec(wav: bytes) -> float:
    info = sf.info(io.BytesIO(wav))
    return info.frames / info.samplerate


def transcribe(model: str, wav: bytes, key: str) -> tuple[str, float]:
    t0 = time.time()
    resp = requests.post(
        f"{BASE}/audio/transcriptions",
        headers={"Authorization": f"Bearer {key}"},
        files={"file": ("sample.wav", wav, "audio/wav")},
        data={"model": model, "language": "ru", "response_format": "json"},
        timeout=300,
    )
    elapsed = time.time() - t0
    resp.raise_for_status()
    return (resp.json().get("text") or ""), elapsed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=150, help="сколько сэмплов взять")
    ap.add_argument("--out", default="stt_bench_result.json")
    args = ap.parse_args()
    key = os.environ["K"]

    print(f"[1/3] Тяну FLEURS-ru test (streaming), первые {args.limit} сэмплов…", flush=True)
    # decode=False: берём сырые байты файла и шлём их в API как есть. Так не нужен
    # torchcodec и нет лишнего пересжатия — модель слышит ровно исходный файл.
    stream = load_dataset("google/fleurs", "ru_ru", split="test", streaming=True)
    stream = stream.cast_column("audio", Audio(decode=False))
    samples = []
    for item in stream:
        wav = item["audio"]["bytes"]
        if wav is None:
            with open(item["audio"]["path"], "rb") as fh:
                wav = fh.read()
        samples.append({"wav": wav,
                        "ref": item.get("transcription") or item.get("raw_transcription") or ""})
        if len(samples) >= args.limit:
            break
    total_sec = sum(duration_sec(s["wav"]) for s in samples)
    print(f"      взято {len(samples)} сэмплов, {total_sec/60:.1f} мин аудио", flush=True)

    report = {}
    for model in MODELS:
        print(f"[2/3] {model}…", flush=True)
        refs, hyps, lats, errors = [], [], [], 0
        for i, s in enumerate(samples, 1):
            try:
                text, elapsed = transcribe(model, s["wav"], key)
            except Exception as exc:
                errors += 1
                if errors <= 3:
                    print(f"      ошибка на #{i}: {type(exc).__name__}: {str(exc)[:120]}", flush=True)
                continue
            refs.append(normalize(s["ref"]))
            hyps.append(normalize(text))
            lats.append(elapsed)
            if i % 25 == 0:
                print(f"      {i}/{len(samples)}  промежуточный WER "
                      f"{jiwer.wer(refs, hyps)*100:.2f}%", flush=True)
        if not hyps:
            report[model] = {"error": "ни одного успешного ответа", "failed": errors}
            continue
        audio_sec = sum(duration_sec(s["wav"]) for s in samples[:len(hyps)])
        report[model] = {
            "wer": round(jiwer.wer(refs, hyps) * 100, 2),
            "cer": round(jiwer.cer(refs, hyps) * 100, 2),
            "ok": len(hyps),
            "failed": errors,
            "latency_median_s": round(statistics.median(lats), 2),
            "latency_p90_s": round(sorted(lats)[int(len(lats) * 0.9)], 2),
            "rtf": round(sum(lats) / audio_sec, 3),  # <1 = быстрее реального времени
        }
        print(f"      ИТОГО {model}: WER {report[model]['wer']}%  CER {report[model]['cer']}%  "
              f"медиана {report[model]['latency_median_s']}с  RTF {report[model]['rtf']}", flush=True)

    print("[3/3] Готово.")
    meta = {"dataset": "google/fleurs ru_ru test", "samples": len(samples),
            "audio_minutes": round(total_sec / 60, 1), "models": report}
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
