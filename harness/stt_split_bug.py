"""Сколько WER подлодки — это разорванные пробелом слова, а не ошибка слуха.

Чиним ТОЛЬКО очевидный дефект: если склейка двух соседних слов гипотезы даёт
слово, которое есть в эталоне, а по отдельности их там нет — значит слово было
разорвано при декодировании. Считаем WER до и после починки.
"""
import json
import os

import jiwer
from datasets import Audio, load_dataset

from stt_bench import MODELS, normalize, transcribe

LIMIT = int(os.environ.get("LIMIT", "150"))
key = os.environ["K"]


def repair_splits(hyp: str, ref: str) -> tuple[str, int]:
    """Склеить пары слов, которые вместе есть в эталоне, а по отдельности — нет."""
    ref_words = set(ref.split())
    words = hyp.split()
    out, fixed, i = [], 0, 0
    while i < len(words):
        if i + 1 < len(words):
            merged = words[i] + words[i + 1]
            if merged in ref_words and words[i] not in ref_words and words[i + 1] not in ref_words:
                out.append(merged)
                fixed += 1
                i += 2
                continue
        out.append(words[i])
        i += 1
    return " ".join(out), fixed


stream = load_dataset("google/fleurs", "ru_ru", split="test", streaming=True)
stream = stream.cast_column("audio", Audio(decode=False))
samples = []
for item in stream:
    samples.append({"wav": item["audio"]["bytes"], "ref": item.get("transcription") or ""})
    if len(samples) >= LIMIT:
        break

report = {}
for model in MODELS:
    refs, raw, repaired, splits, affected = [], [], [], 0, 0
    for s in samples:
        ref = normalize(s["ref"])
        hyp = normalize(transcribe(model, s["wav"], key)[0])
        fixed_hyp, n = repair_splits(hyp, ref)
        refs.append(ref)
        raw.append(hyp)
        repaired.append(fixed_hyp)
        splits += n
        affected += 1 if n else 0
    report[model] = {
        "wer_raw": round(jiwer.wer(refs, raw) * 100, 2),
        "wer_repaired": round(jiwer.wer(refs, repaired) * 100, 2),
        "разорванных_слов": splits,
        "сэмплов_с_разрывом": f"{affected}/{len(samples)}",
    }
    print(model, json.dumps(report[model], ensure_ascii=False), flush=True)

print(json.dumps(report, ensure_ascii=False, indent=2))
