"""Точность таймкодов: насколько время в транскрипте расходится с реальностью.

ЗАЧЕМ ОТДЕЛЬНЫЙ ЗАМЕР. WER меряет, ЧТО сказано, и молчит о том, КОГДА. Между тем
транскрипт длинной записи нужен ровно для того, чтобы ткнуть в цитату и попасть в
это место. Ни один публичный бенчмарк русской речи таймкоды не проверяет — ни
наш основной прогон, ни сводка Альфацефей, ни лидерборды HF. Этот скрипт закрывает
дыру.

ОТКУДА ЭТАЛОН. Длинные файлы мы собираем сами из коротких фраз FLEURS
(harness/make_long.py), поэтому границы каждой фразы известны ТОЧНО — мы их сами
и расставили. Не нужны ни судьи, ни второй движок, ни выравнивание: истина
задана построением.

ЧТО СЧИТАЕМ И ПОЧЕМУ ИМЕННО ТАК.
  • Смещение ЗНАКОВОЕ. Дефект, ради которого всё затевалось, односторонний —
    время уезжает назад. Модуль спрятал бы именно это.
  • Медиана и p90, а не среднее: единичный выброс в сотню секунд перекашивает
    среднее и делает его бесполезным.
  • Доля промахов > 0.5 с и > 2 с. Полсекунды — «попал в свою реплику»,
    две — «попал в чужую».
  • Наклон ошибки внутри окна. Батченый WhisperX склеивает речь в окна по 30 с
    и раскладывает слова внутри линейно. Если так, ошибка обязана РАСТИ от
    начала окна и обнуляться на границе — «пила». Считаем корреляцию ошибки с
    позицией внутри окна: это прямая проверка механизма, а не догадка.

КРИТЕРИЙ «ПОЧИНИЛИ» — не «стало меньше», а другое поведение: ошибка перестала
копиться монотонно. Симметричная дрожь в пределах ±0.3 с это уже точность
выравнивания, и она нормальна.

Запуск:
  python harness/timing_bench.py --engine whisper-1 --file data/long/long_15min.wav
  python harness/timing_bench.py --analyze results/timing/*.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics as st
import time
from pathlib import Path

import requests

OPENAI_BASE = os.environ.get("HUB_URL", "https://api.neuraldeep.ru/v1")
SPEECHCORE_BASE = os.environ.get("SC_URL", "https://speechcore.neuraldeep.ru/api")
WINDOW_SEC = float(os.environ.get("TIMING_WINDOW_SEC", "30"))  # окно склейки WhisperX

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def norm(text: str) -> list[str]:
    return _PUNCT.sub(" ", (text or "").lower().replace("ё", "е")).split()


# ── движки: каждый возвращает список сегментов {start, end, text} ──


def segments_openai(model: str, wav: Path) -> list[dict]:
    """Хабовые whisper-модели. response_format=verbose_json — иначе таймкодов нет
    вовсе, обычный json отдаёт только текст."""
    with wav.open("rb") as fh:
        r = requests.post(
            f"{OPENAI_BASE}/audio/transcriptions",
            headers={"Authorization": f"Bearer {os.environ['K']}"},
            files={"file": (wav.name, fh, "audio/wav")},
            data={"model": model, "language": "ru", "response_format": "verbose_json"},
            timeout=1800,
        )
    r.raise_for_status()
    body = r.json()
    return [{"start": s["start"], "end": s["end"], "text": s.get("text", ""),
             "words": s.get("words") or []}
            for s in (body.get("segments") or [])]


def segments_speechcore(model: str, wav: Path) -> list[dict]:
    """Асинхронный пайплайн: upload → poll. Токен в SC."""
    hdr = {"Authorization": f"Bearer {os.environ['SC']}"}
    with wav.open("rb") as fh:
        r = requests.post(f"{SPEECHCORE_BASE}/transcribe", headers=hdr,
                          files={"file": (wav.name, fh, "audio/wav")},
                          data={"language": "ru", "model": "large-v3"}, timeout=600)
    r.raise_for_status()
    task_id = r.json().get("task_id") or r.json().get("id")
    for _ in range(600):
        time.sleep(5)
        p = requests.get(f"{SPEECHCORE_BASE}/tasks/{task_id}", headers=hdr, timeout=60)
        p.raise_for_status()
        body = p.json()
        if body.get("status") in ("completed", "done", "success"):
            segs = body.get("segments") or (body.get("result") or {}).get("segments") or []
            return [{"start": s["start"], "end": s["end"], "text": s.get("text", "")} for s in segs]
        if body.get("status") in ("failed", "error"):
            raise RuntimeError(f"speechcore: {body}")
    raise TimeoutError("speechcore: не дождались")


def segments_parakeet(model: str, wav: Path) -> list[dict]:
    """Локальная модель — таймкоды родные, TDT отдаёт их вместе с текстом."""
    import librosa
    import torch
    from transformers import AutoModelForTDT, AutoProcessor

    mid = os.environ.get("LOCAL_MODEL", "nvidia/parakeet-tdt-0.6b-v3")
    proc = AutoProcessor.from_pretrained(mid)
    m = AutoModelForTDT.from_pretrained(mid, dtype="auto", device_map="cuda")
    sr = proc.feature_extractor.sampling_rate
    audio, _ = librosa.load(str(wav), sr=sr, mono=True)
    inputs = proc(audio, sampling_rate=sr, return_tensors="pt").to(m.device)
    with torch.no_grad():
        out = m.generate(**inputs, return_dict_in_generate=True)
    texts, stamps = proc.decode(out.sequences, durations=out.durations, skip_special_tokens=True)
    segs = []
    for chunk in (stamps[0] if isinstance(stamps, list) else stamps):
        segs.append({"start": float(chunk["start"]), "end": float(chunk["end"]),
                     "text": chunk.get("text") or chunk.get("segment") or ""})
    if not segs and texts:
        raise RuntimeError("parakeet вернул текст без таймкодов")
    return segs


ENGINES = {
    "whisper-1": segments_openai,
    "whisper-podlodka-turbo": segments_openai,
    "speechcore": segments_speechcore,
    "parakeet-tdt-0.6b-v3": segments_parakeet,
}


# ── сопоставление и метрики ──


def word_stream(segs: list[dict]) -> tuple[list[str], list[float], str]:
    """Плоский поток слов гипотезы с временем начала каждого.

    Если движок отдаёт пословные таймкоды (whisper их отдаёт всегда) — берём
    их как есть. Если только посегментные — раскладываем слова внутри сегмента
    равномерно; тогда ошибка замера ограничена длиной сегмента, и это честно
    помечено в результате как resolution=segment.
    """
    words, times = [], []
    exact = any(s.get("words") for s in segs)
    for s in segs:
        if exact:
            for w in s.get("words") or []:
                for tok in norm(w.get("word", "")):
                    words.append(tok)
                    times.append(float(w.get("start", s["start"])))
        else:
            toks = norm(s["text"])
            span = max(float(s["end"]) - float(s["start"]), 0.0)
            for i, tok in enumerate(toks):
                words.append(tok)
                times.append(float(s["start"]) + (span * i / len(toks) if toks else 0.0))
    return words, times, ("word" if exact else "segment")


def match(bounds: list[dict], segs: list[dict]) -> tuple[list[dict], str]:
    """Каждой эталонной фразе — момент, когда движок начал её произносить.

    Сопоставляем ПО ТЕКСТУ: время — это ровно то, что проверяется, опираться на
    него значило бы рассуждать по кругу.

    Выравниваем два потока слов (эталон и гипотеза) через difflib. Он даёт
    монотонное глобальное выравнивание, устойчивое к пропускам и вставкам
    распознавания. Самодельный «курсор по сегментам» этого не умел: одно ложное
    совпадение уводило его вперёд навсегда — на наборе с паузой 0.2 с
    сопоставилось 9 фраз из 76 и вылезло смещение в 669 с.

    Берём время слова гипотезы, выровненного на ПЕРВОЕ слово фразы (или на одно
    из первых трёх, если первое не распозналось). Фразы, где не совпало и это,
    пропускаем — лучше меньше точек, чем выдуманные.
    """
    from difflib import SequenceMatcher

    hyp, hyp_t, resolution = word_stream(segs)
    ref, first_of = [], {}   # first_of: индекс первого слова фразы → сама фраза
    for b in bounds:
        toks = norm(b["text"])
        if not toks:
            continue
        first_of[len(ref)] = b
        ref.extend(toks)

    aligned: dict[int, int] = {}
    for blk in SequenceMatcher(None, ref, hyp, autojunk=False).get_matching_blocks():
        for k in range(blk.size):
            aligned[blk.a + k] = blk.b + k

    out = []
    for ref_i, b in first_of.items():
        hit = next((aligned[ref_i + d] for d in range(3) if ref_i + d in aligned), None)
        if hit is None:
            continue
        pred = hyp_t[hit]
        out.append({"true_start": b["start"], "pred_start": round(pred, 3),
                    "offset": round(pred - b["start"], 3), "text": b["text"][:60]})
    return out, resolution


def metrics(pairs: list[dict]) -> dict:
    if not pairs:
        return {"n": 0}
    off = [p["offset"] for p in pairs]
    off_sorted = sorted(off)
    # Наклон внутри окна: если движок раскладывает слова линейно по окну в 30 с,
    # ошибка растёт от начала окна к концу и падает на границе.
    xs = [p["true_start"] % WINDOW_SEC for p in pairs]
    slope = None
    if len(pairs) > 5:
        mx, my = st.mean(xs), st.mean(off)
        den = sum((x - mx) ** 2 for x in xs)
        if den > 0:
            slope = round(sum((x - mx) * (y - my) for x, y in zip(xs, off)) / den, 4)
    return {
        "n": len(pairs),
        "median": round(st.median(off), 2),
        "p90": round(off_sorted[int(0.9 * (len(off_sorted) - 1))], 2),
        "min": round(min(off), 2), "max": round(max(off), 2),
        "gt_0.5s": round(100 * sum(abs(o) > 0.5 for o in off) / len(off), 1),
        "gt_2s": round(100 * sum(abs(o) > 2 for o in off) / len(off), 1),
        "back_share": round(100 * sum(o < 0 for o in off) / len(off), 1),
        "slope_in_window": slope,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine")
    ap.add_argument("--file")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--analyze", nargs="*")
    args = ap.parse_args()

    if args.analyze:
        print(f"{'движок':26} {'набор':14} {'n':>4} {'медиана':>8} {'p90':>7} "
              f"{'макс':>7} {'>0.5с':>7} {'>2с':>6} {'назад':>7} {'наклон':>7}")
        for f in sorted(args.analyze):
            d = json.loads(Path(f).read_text(encoding="utf-8"))
            m = d["metrics"]
            if not m.get("n"):
                print(f"{d['engine']:26} {d['set']:14}  нет сопоставленных фраз")
                continue
            print(f"{d['engine']:26} {d['set']:14} {m['n']:>4} {m['median']:>7.2f}с "
                  f"{m['p90']:>6.2f}с {m['max']:>6.2f}с {m['gt_0.5s']:>6.1f}% "
                  f"{m['gt_2s']:>5.1f}% {m['back_share']:>6.1f}% "
                  f"{(m['slope_in_window'] if m['slope_in_window'] is not None else 0):>7.3f}")
        return

    wav = Path(args.file)
    man_path = Path(args.manifest or wav.parent / "manifest.json")
    man = json.loads(man_path.read_text(encoding="utf-8"))
    entry = next(f for f in man["files"] if f["file"] == wav.name)
    if "bounds" not in entry:
        raise SystemExit("в манифесте нет границ — пересобери набор новым make_long.py")

    call = ENGINES.get(args.engine)
    if call is None:
        raise SystemExit(f"неизвестный движок: {args.engine}. Есть: {', '.join(ENGINES)}")

    t0 = time.time()
    segs = call(args.engine, wav)
    pairs, resolution = match(entry["bounds"], segs)
    res = {
        "engine": args.engine, "set": wav.parent.name + "/" + wav.stem,
        "gap_sec": entry.get("gap_sec", man.get("gap_sec")),
        "duration_s": entry["duration_s"], "utterances": entry["utterances"],
        "segments_returned": len(segs), "matched": len(pairs), "resolution": resolution,
        "elapsed_s": round(time.time() - t0, 1),
        "metrics": metrics(pairs), "pairs": pairs,
    }
    out = Path(args.out or f"results/timing/{wav.parent.name}_{wav.stem}_{args.engine}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    m = res["metrics"]
    print(f"{args.engine} · {res['set']} · пауза {res['gap_sec']}с · "
          f"сегментов {len(segs)}, сопоставлено {len(pairs)}/{len(entry['bounds'])} "
          f"(разрешение: {resolution})")
    if m.get("n"):
        print(f"  смещение: медиана {m['median']}с, p90 {m['p90']}с, макс {m['max']}с · "
              f">2с у {m['gt_2s']}% · назад {m['back_share']}% · наклон {m['slope_in_window']}")
    print(f"  → {out}")


if __name__ == "__main__":
    main()
