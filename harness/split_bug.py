"""Диагностика: какая часть WER — разорванные пробелом слова, а не ошибка слуха.

Работает по готовым JSONL, без обращений к API.

Чинится ТОЛЬКО очевидный дефект: если склейка двух соседних слов гипотезы даёт
слово из эталона, а по отдельности их там нет — значит слово было разорвано при
декодировании.

⚠️ Это ORACLE-оценка: починка подсматривает в эталон, в проде так нельзя.
Число отвечает на вопрос «какая доля ошибок — разрывы», и НЕ является
альтернативным WER модели.

Запуск:  python harness/split_bug.py "results/raw150/*.jsonl"
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import jiwer

from analyze import load, normalize


def repair(hyp: str, ref: str) -> tuple[str, int]:
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    report = {}
    for path in sorted({p for pat in args.files for p in glob.glob(pat)}):
        meta, rows = load(path)
        if not rows:
            continue
        engine = meta.get("engine") or Path(path).stem
        refs, raw, fixed_hyps, splits, affected = [], [], [], 0, 0
        for row in rows:
            ref, hyp = normalize(row["ref_raw"]), normalize(row["hyp_raw"])
            fixed, n = repair(hyp, ref)
            refs.append(ref)
            raw.append(hyp)
            fixed_hyps.append(fixed)
            splits += n
            affected += 1 if n else 0
        report[engine] = {
            "wer": round(jiwer.wer(refs, raw) * 100, 2),
            "wer_oracle_repaired": round(jiwer.wer(refs, fixed_hyps) * 100, 2),
            "разорванных_слов": splits,
            "сэмплов_с_разрывом": f"{affected}/{len(rows)}",
        }
        print(f"{engine:<26} WER {report[engine]['wer']:>6.2f}%  "
              f"после oracle-починки {report[engine]['wer_oracle_repaired']:>6.2f}%  "
              f"разрывов {splits:>3} в {affected}/{len(rows)} сэмплах")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                       encoding="utf-8")


if __name__ == "__main__":
    main()
