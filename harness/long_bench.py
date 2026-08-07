"""Прогон длинных записей: WER плюс то, что на коротких проявиться не могло.

На фразе в 11 секунд движку нечем сломаться. На получасовой записи включаются
нарезка на окна, VAD и склейка сегментов — и появляются три отказа, которых
короткий замер не видит:

  обрыв       — вернулся текст только за начало файла;
  зацикливание — модель ушла в повтор одной фразы (болезнь Whisper на длинном);
  потери на швах — слова, съеденные на границах окон.

Поэтому кроме WER считаются: доля возвращённых слов от эталона и доля
повторяющихся 5-грамм в гипотезе.

Запуск:  K=... DG=... RPA=... SC=... python harness/long_bench.py --engines whisper-1 nova-2
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import jiwer
import soundfile as sf

from analyze import normalize
from run_bench import pick_caller


def repetition_ratio(text: str, n: int = 5) -> float:
    """Доля 5-грамм, встретившихся больше одного раза. У здорового текста это
    единицы процентов; при зацикливании уходит к 1."""
    words = text.split()
    if len(words) < n * 2:
        return 0.0
    grams = Counter(tuple(words[i:i + n]) for i in range(len(words) - n + 1))
    repeated = sum(c for c in grams.values() if c > 1)
    return repeated / sum(grams.values())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", nargs="+", required=True)
    ap.add_argument("--data", default="data/long")
    ap.add_argument("--files", nargs="*", default=None, help="имена wav; по умолчанию все")
    ap.add_argument("--out", default="results/long.jsonl")
    args = ap.parse_args()

    data_dir = Path(args.data)
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    files = args.files or [f["file"] for f in manifest["files"]]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as fh:
        for engine in args.engines:
            call = pick_caller(engine)
            for name in files:
                wav_path = data_dir / name
                wav = wav_path.read_bytes()
                info = sf.info(str(wav_path))
                dur = info.frames / info.samplerate
                ref = (data_dir / name.replace(".wav", ".txt")).read_text(encoding="utf-8")

                t0 = time.time()
                try:
                    hyp, error = call(engine, wav), None
                except Exception as exc:
                    hyp, error = "", f"{type(exc).__name__}: {str(exc)[:200]}"
                elapsed = time.time() - t0

                rn, hn = normalize(ref), normalize(hyp)
                row = {
                    "engine": engine, "file": name, "duration_s": round(dur, 1),
                    "latency_s": round(elapsed, 2), "rtf": round(elapsed / dur, 3),
                    "wer": round(jiwer.wer(rn, hn) * 100, 2) if hn else None,
                    "ref_words": len(rn.split()), "hyp_words": len(hn.split()),
                    "coverage": round(len(hn.split()) / max(1, len(rn.split())), 3),
                    "repetition_5gram": round(repetition_ratio(hn), 3),
                    "error": error, "hyp_raw": hyp,
                }
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                status = error or (
                    f"WER {row['wer']:>6.2f}%  покрытие {row['coverage']:.2f}  "
                    f"повторы {row['repetition_5gram']:.2f}  "
                    f"{elapsed:.1f}с  RTF {row['rtf']:.3f}")
                print(f"{engine:<24} {name:<18} {status}", flush=True)


if __name__ == "__main__":
    main()
