"""Собрать длинные записи из коротких высказываний FLEURS.

Зачем именно склейка, а не готовый длинный корпус: так меняется РОВНО одна вещь —
длина файла. Слова, дикторы и эталон те же, что в основном прогоне, поэтому любое
ухудшение WER — эффект нарезки, окон и склейки внутри движка, а не разницы
материала. Готовый длинный корпус смешал бы эти два фактора.

Между высказываниями вставляется короткая пауза: без неё фразы стыкуются встык,
и VAD не может найти границу там, где её слышит человек.

Эталон длинного файла = конкатенация эталонов его кусков.

Побочно это даёт то, чего нет ни в одном публичном корпусе: ТОЧНЫЕ границы
каждой фразы. Мы их сами и расставили при склейке, поэтому знаем без всякого
выравнивания и без судей. Пишем в манифест — на них меряется точность
таймкодов (harness/timing_bench.py).

Длина паузы вынесена в --gap намеренно: это управляемая переменная. Если
таймкоды уезжают из-за того, что движок линейно раскладывает слова внутри
окна и не знает про тишину внутри него, то ошибка обязана расти вместе с
паузой. Три набора с разной паузой превращают догадку в проверяемое
предсказание.

Запуск:  python harness/make_long.py --minutes 1 5 15 30 --out data/long
         python harness/make_long.py --minutes 15 --gap 3.0 --out data/long_gap3
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset

DATASET = "google/fleurs"
CONFIG = "ru_ru"
GAP_SEC = 0.4  # пауза между высказываниями (по умолчанию; см. --gap)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, nargs="+", default=[1, 5, 15, 30])
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="data/long")
    ap.add_argument("--gap", type=float, default=GAP_SEC,
                    help="пауза между высказываниями, с — управляемая переменная замера таймкодов")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    gap_sec = args.gap
    stream = load_dataset(DATASET, CONFIG, split=args.split, streaming=True)
    stream = stream.cast_column("audio", Audio(decode=False))

    targets = sorted(args.minutes)
    need_sec = max(targets) * 60
    chunks, refs, total = [], [], 0.0
    rate = None
    for item in stream:
        data, sr = sf.read(io.BytesIO(item["audio"]["bytes"]), dtype="float32")
        rate = rate or sr
        if sr != rate:
            raise SystemExit(f"разная частота дискретизации: {sr} против {rate}")
        chunks.append(data)
        refs.append((item.get("transcription") or "").strip())
        total += len(data) / sr + gap_sec
        if total >= need_sec:
            break

    gap = np.zeros(int(rate * gap_sec), dtype="float32")
    manifest = []
    for minutes in targets:
        limit = minutes * 60
        acc, used, dur = [], 0, 0.0
        # bounds — истинные [start, end] каждой фразы в собранном файле.
        # Курсор идёт ровно так же, как потом склеиваются куски: фраза, затем
        # пауза. Никакого выравнивания тут нет — это арифметика склейки.
        bounds, cursor = [], 0.0
        for data, ref_text in zip(chunks, refs):
            piece = len(data) / rate + gap_sec
            if dur + piece > limit and acc:
                break
            speech = len(data) / rate
            bounds.append({"i": used, "start": round(cursor, 3),
                           "end": round(cursor + speech, 3), "text": ref_text})
            cursor += speech + gap_sec
            acc.append(data)
            dur += piece
            used += 1
        audio = np.concatenate([x for pair in zip(acc, [gap] * len(acc)) for x in pair])
        name = f"long_{int(minutes):02d}min.wav"
        sf.write(out_dir / name, audio, rate, subtype="PCM_16")
        ref = " ".join(refs[:used])
        (out_dir / f"long_{int(minutes):02d}min.txt").write_text(ref, encoding="utf-8")
        manifest.append({"file": name, "target_minutes": minutes,
                         "duration_s": round(len(audio) / rate, 1),
                         "utterances": used, "ref_words": len(ref.split()),
                         "gap_sec": gap_sec, "bounds": bounds})
        print(f"{name}: {len(audio)/rate/60:.1f} мин, {used} высказываний, "
              f"{len(ref.split())} слов эталона")

    (out_dir / "manifest.json").write_text(
        json.dumps({"gap_sec": gap_sec, "sample_rate": rate, "files": manifest},
                   ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
