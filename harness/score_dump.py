"""Оценить готовые сегменты (JSON с прогона движка) тем же мерилом, что и живые API.

Нужен, когда движок гоняется не через сеть, а прямо на боксе — например при
переборе chunk_size внутри WhisperX. Метрика и сопоставление те же, что в
timing_bench, иначе числа были бы несравнимы.
"""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from timing_bench import match, metrics, refine_onsets

man = json.loads(Path(sys.argv[1]).read_text())
wav = Path(sys.argv[2])
entry = next(f for f in man["files"] if f["file"] == wav.name)
bounds = refine_onsets(entry["bounds"], wav)

print(f"{'вариант':16} {'сегм':>5} {'сопост':>7} {'медиана':>9} {'p90':>7} {'>2с':>6} {'назад':>7}")
for f in sorted(sys.argv[3:]):
    d = json.loads(Path(f).read_text())
    pairs, _ = match(bounds, d["segments"])
    m = metrics(pairs)
    if not m.get("n"):
        print(f"{d['name']:16} нет сопоставленных фраз"); continue
    print(f"{d['name']:16} {len(d['segments']):>5} {m['n']:>3}/{len(bounds):<3} "
          f"{m['median']:>8.2f}с {m['p90']:>6.2f}с {m['gt_2s']:>5.1f}% {m['back_share']:>6.1f}%")
