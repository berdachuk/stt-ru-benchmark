"""Перепрогнать ТОЛЬКО те высказывания, где движок вернул ошибку.

Зачем отдельным шагом. Ошибка вызова — это не «модель не смогла», а состояние
лейна в ту минуту: очередь, таймаут, рестарт. Если такие строки просто выбросить,
выборка станет разной у разных движков и сравнение перестанет быть парным.
Поэтому дефекты добираются повторно, а сколько их было — остаётся в отчёте.

Файл переписывается на месте, порядок строк и `_meta` сохраняются; в починенную
строку добавляется `retried: true`.

Запуск:  K=... DG=... SC=... python harness/retry_failed.py results/raw/*.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

from datasets import Audio, load_dataset

from run_bench import CONFIG, DATASET, pick_caller


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--attempts", type=int, default=3, help="сколько заходов на строку")
    ap.add_argument("--pause", type=float, default=5.0, help="пауза между заходами, с")
    args = ap.parse_args()

    paths = sorted({p for pattern in args.files for p in glob.glob(pattern)})
    audio_cache: dict[int, bytes] = {}

    for path in paths:
        lines = [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
        meta = next((x["_meta"] for x in lines if "_meta" in x), {})
        engine = meta.get("engine") or Path(path).stem
        broken = [x for x in lines if not x.get("_meta") and x.get("error")]
        if not broken:
            print(f"{engine:<24} ошибок нет")
            continue

        print(f"{engine:<24} ошибок {len(broken)}: {[x['idx'] for x in broken]}", flush=True)
        need = {x["idx"] for x in broken}
        if not need <= audio_cache.keys():
            stream = load_dataset(DATASET, CONFIG, split=meta.get("split", "test"), streaming=True)
            stream = stream.cast_column("audio", Audio(decode=False))
            for idx, item in enumerate(stream):
                if idx in need:
                    audio_cache[idx] = item["audio"]["bytes"]
                if idx > max(need):
                    break

        call = pick_caller(engine)
        fixed = 0
        for row in broken:
            for attempt in range(1, args.attempts + 1):
                t0 = time.time()
                try:
                    row["hyp_raw"] = call(engine, audio_cache[row["idx"]])
                    row["latency_s"] = round(time.time() - t0, 3)
                    row["error"] = None
                    row["retried"] = True
                    fixed += 1
                    break
                except Exception as exc:
                    row["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
                    print(f"   idx={row['idx']} заход {attempt}/{args.attempts}: {row['error'][:90]}",
                          flush=True)
                    if attempt < args.attempts:
                        time.sleep(args.pause)

        with Path(path).open("w", encoding="utf-8") as fh:
            for obj in lines:
                fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
        print(f"{engine:<24} починено {fixed} из {len(broken)}", flush=True)


if __name__ == "__main__":
    main()
