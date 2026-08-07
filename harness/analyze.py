"""Метрики из построчных JSONL: WER/CER, доверительные интервалы, попарные разницы.

Считает две нормализации, чтобы отделить распознавание от оформления:

  базовая  — нижний регистр, ё→е, снятие пунктуации, схлопывание пробелов;
  + числа  — то же плюс перевод числительных прописью в цифры (number-parser).

Зачем вторая. Движки по-разному пишут числа: одни цифрами, другие словами.
Эталон FLEURS пишет цифрами, поэтому модель, отвечающая словами, получает
штраф за оформление, а не за ошибку слуха — известная ловушка мультиязычных
замеров (arXiv:2409.02449). ⚠️ Для русского перевод неполон: количественные
числительные разбираются, порядковые («тысяча девятьсот девяностом») — нет.
Поэтому колонка с числами ДОПОЛНЯЕТ базовую, а не заменяет её.

Интервалы — бутстрап по высказываниям (Bisani & Ney, ICASSP 2004): выборка
переcэмплируется с возвращением, WER пересчитывается корпусно на каждой реплике.
Для пар движков считается разница на ОДНИХ И ТЕХ ЖЕ высказываниях (paired) —
так шум набора сокращается и видно, значима ли разница.

Запуск:  python harness/analyze.py results/raw/*.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import re
from pathlib import Path

import jiwer

_PUNCT_RE = re.compile(r"[^\wа-яёА-ЯЁ\s]", re.UNICODE)

try:
    from number_parser import parse as _parse_numbers
except ImportError:  # необязательная зависимость
    _parse_numbers = None


def normalize(text: str, numbers: bool = False) -> str:
    if not text:
        return ""
    text = text.lower().replace("ё", "е")
    if numbers and _parse_numbers is not None:
        try:
            text = _parse_numbers(text, language="ru")
        except Exception:
            pass
    text = _PUNCT_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def load(path: str) -> tuple[dict, list[dict]]:
    meta, rows = {}, []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if "_meta" in obj:
            meta = obj["_meta"]
        elif not obj.get("error"):
            rows.append(obj)
    return meta, rows


def corpus_wer(rows: list[dict], numbers: bool) -> float:
    refs = [normalize(r["ref_raw"], numbers) for r in rows]
    hyps = [normalize(r["hyp_raw"], numbers) for r in rows]
    return jiwer.wer(refs, hyps) * 100


def corpus_cer(rows: list[dict], numbers: bool) -> float:
    refs = [normalize(r["ref_raw"], numbers) for r in rows]
    hyps = [normalize(r["hyp_raw"], numbers) for r in rows]
    return jiwer.cer(refs, hyps) * 100


def per_row_counts(rows: list[dict], numbers: bool) -> list[tuple[int, int]]:
    """(ошибок, длина эталона) на высказывание.

    Корпусный WER = сумма ошибок / сумма длин, поэтому бутстрап можно гонять по
    этим парам, а не пересчитывать выравнивание на каждой реплике. Результат
    математически тот же, но 50 000 реплик считаются за секунды, а не за часы —
    это важно, когда интервал проходит близко к нулю и грубость перцентиля
    начинает влиять на вывод.
    """
    out = []
    for r in rows:
        ref, hyp = normalize(r["ref_raw"], numbers), normalize(r["hyp_raw"], numbers)
        m = jiwer.process_words([ref], [hyp])
        out.append((m.substitutions + m.deletions + m.insertions,
                    m.substitutions + m.deletions + m.hits))
    return out


def _percentile(values: list[float], q: float) -> float:
    """Перцентиль с линейной интерполяцией (как numpy), а не грубым индексом."""
    if not values:
        return float("nan")
    pos = q * (len(values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def bootstrap_ci(rows: list[dict], numbers: bool, runs: int = 1000, seed: int = 0):
    """95%-й интервал WER бутстрапом по высказываниям."""
    counts = per_row_counts(rows, numbers)
    rnd = random.Random(seed)
    n = len(counts)
    vals = []
    for _ in range(runs):
        e = l = 0
        for _ in range(n):
            de, dl = counts[rnd.randrange(n)]
            e += de
            l += dl
        vals.append(e / l * 100 if l else 0.0)
    vals.sort()
    return _percentile(vals, 0.025), _percentile(vals, 0.975)


def paired_delta_ci(a: list[dict], b: list[dict], numbers: bool,
                    runs: int = 1000, seed: int = 0):
    """Разница WER(a) − WER(b) на общих высказываниях, с 95%-м интервалом."""
    by_idx_b = {r["idx"]: r for r in b}
    pairs = [(r, by_idx_b[r["idx"]]) for r in a if r["idx"] in by_idx_b]
    if not pairs:
        return None
    ca = per_row_counts([p[0] for p in pairs], numbers)
    cb = per_row_counts([p[1] for p in pairs], numbers)
    point = (sum(x[0] for x in ca) / sum(x[1] for x in ca)
             - sum(x[0] for x in cb) / sum(x[1] for x in cb)) * 100
    rnd = random.Random(seed)
    n = len(pairs)
    vals = []
    for _ in range(runs):
        ea = la = eb = lb = 0
        for _ in range(n):
            j = rnd.randrange(n)
            ea += ca[j][0]; la += ca[j][1]
            eb += cb[j][0]; lb += cb[j][1]
        vals.append((ea / la - eb / lb) * 100 if la and lb else 0.0)
    vals.sort()
    return point, _percentile(vals, 0.025), _percentile(vals, 0.975), n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--runs", type=int, default=1000)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    paths = sorted({p for pattern in args.files for p in glob.glob(pattern)})
    data = {}
    for path in paths:
        meta, rows = load(path)
        if rows:
            data[meta.get("engine") or Path(path).stem] = rows

    print(f"{'движок':<26} {'WER':>7} {'95% ДИ':>16} {'CER':>7} | {'WER+числа':>10} {'CER+числа':>10}   n")
    print("-" * 96)
    summary = {}
    for engine, rows in sorted(data.items(), key=lambda kv: corpus_wer(kv[1], False)):
        wer = corpus_wer(rows, False)
        lo, hi = bootstrap_ci(rows, False, args.runs)
        entry = {
            "wer": round(wer, 2), "ci95": [round(lo, 2), round(hi, 2)],
            "cer": round(corpus_cer(rows, False), 2),
            "wer_numbers": round(corpus_wer(rows, True), 2),
            "cer_numbers": round(corpus_cer(rows, True), 2),
            "n": len(rows),
            "latency_median_s": round(sorted(r["latency_s"] for r in rows)[len(rows) // 2], 2),
        }
        summary[engine] = entry
        print(f"{engine:<26} {entry['wer']:>6.2f}% [{lo:>5.2f}; {hi:>5.2f}] {entry['cer']:>6.2f}% |"
              f" {entry['wer_numbers']:>9.2f}% {entry['cer_numbers']:>9.2f}%  {len(rows):>4}")

    names = list(data)
    if len(names) > 1:
        print("\nПопарная разница WER на общих высказываниях (базовая нормализация):")
        print("отрицательная = первый лучше; интервал, накрывающий 0, = разница не доказана\n")
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                res = paired_delta_ci(data[a], data[b], False, args.runs)
                if not res:
                    continue
                point, lo, hi, n = res
                verdict = "разница не доказана" if lo <= 0 <= hi else "разница значима"
                print(f"  {a:<24} − {b:<24} {point:+6.2f} п.п. "
                      f"[{lo:+5.2f}; {hi:+5.2f}]  n={n}  → {verdict}")
                summary.setdefault("_pairs", {})[f"{a} − {b}"] = {
                    "delta_pp": round(point, 2), "ci95": [round(lo, 2), round(hi, 2)],
                    "significant": not (lo <= 0 <= hi), "n": n,
                }

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nсводка → {args.json_out}")


if __name__ == "__main__":
    main()
