"""Из чего складывается разрыв WER: показать реальные расхождения обеих моделей."""
import os

import jiwer
from datasets import Audio, load_dataset

from stt_bench import MODELS, normalize, transcribe

LIMIT = 40
key = os.environ["K"]

stream = load_dataset("google/fleurs", "ru_ru", split="test", streaming=True)
stream = stream.cast_column("audio", Audio(decode=False))
samples = []
for item in stream:
    samples.append({"wav": item["audio"]["bytes"],
                    "ref": item.get("transcription") or ""})
    if len(samples) >= LIMIT:
        break

shown = 0
for s in samples:
    ref = normalize(s["ref"])
    hyps = {m: normalize(transcribe(m, s["wav"], key)[0]) for m in MODELS}
    werr = {m: jiwer.wer(ref, h) for m, h in hyps.items()}
    # интересны случаи, где подлодка ошиблась заметно сильнее
    if werr["whisper-podlodka-turbo"] - werr["whisper-1"] < 0.08:
        continue
    shown += 1
    print("=" * 78)
    print(f"эталон    : {ref}")
    for m in MODELS:
        print(f"{m:<24}: {hyps[m]}   [WER {werr[m]*100:.0f}%]")
    if shown >= 6:
        break
print(f"\nпоказано расхождений: {shown} из {LIMIT} прослушанных")
