"""Сводная таблица «движок × домен» и место каждого движка внутри домена.

Общего среднего по доменам НЕТ намеренно. Домены разной сложности и разного
размера, и усреднение по ним даёт число, которое ничего не описывает: движок,
выигравший студийную начитку, может проигрывать спонтанную речь. Смысл сводки
ровно в том, чтобы это расхождение было видно, а не спрятано в одну цифру.

Запуск:  python harness/summarize.py --domains fleurs podlodka rulibrispeech
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from analyze import bootstrap_ci, corpus_wer, load

DOMAIN_TITLES = {
    "fleurs": "студийная начитка новостей",
    "fleurs775": "студийная начитка новостей",
    "podlodka": "спонтанная речь: лекции и интервью",
    "rulibrispeech": "аудиокниги",
    "sova_rudevices": "бытовые команды устройствам",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", nargs="+", required=True)
    ap.add_argument("--results", default="results")
    ap.add_argument("--runs", type=int, default=20000)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    table: dict[str, dict[str, dict]] = {}
    for domain in args.domains:
        folder = Path(args.results) / domain
        if not folder.is_dir():
            print(f"⚠️  нет папки {folder}, домен пропущен")
            continue
        for path in sorted(folder.glob("*.jsonl")):
            meta, rows = load(str(path))
            if not rows:
                continue
            engine = meta.get("engine") or path.stem
            wer = corpus_wer(rows, False)
            lo, hi = bootstrap_ci(rows, False, args.runs)
            table.setdefault(engine, {})[domain] = {
                "wer": round(wer, 2), "ci95": [round(lo, 2), round(hi, 2)], "n": len(rows)}

    domains = [d for d in args.domains if any(d in v for v in table.values())]
    width = max(len(e) for e in table) + 2

    header = f"{'движок':<{width}}" + "".join(f"{d:>26}" for d in domains)
    print(header)
    print("-" * len(header))
    for engine in sorted(table, key=lambda e: table[e].get(domains[0], {}).get("wer", 999)):
        line = f"{engine:<{width}}"
        for d in domains:
            cell = table[engine].get(d)
            line += (f"{cell['wer']:>8.2f}% [{cell['ci95'][0]:>5.2f};{cell['ci95'][1]:>6.2f}]"
                     if cell else f"{'—':>26}")
        print(line)

    print("\nМесто внутри домена (по точечной оценке WER):")
    for d in domains:
        order = sorted((v[d]["wer"], e) for e, v in table.items() if d in v)
        n = next(v[d]["n"] for v in table.values() if d in v)
        print(f"  {d} — {DOMAIN_TITLES.get(d, '')}, n={n}")
        print("     " + "  ›  ".join(f"{e} {w:.2f}%" for w, e in order))

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(table, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
        print(f"\nсводка → {args.json_out}")


if __name__ == "__main__":
    main()
