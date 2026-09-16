"""Прогон одного движка по набору и запись ПОСТРОЧНОГО результата в JSONL.

Разделение намеренное: этот скрипт только собирает сырые транскрипты, метрики
считает analyze.py. Так любой может пересчитать WER своей нормализацией, не
переспрашивая API и не веря нашим агрегатам.

Движки:
  whisper-1, whisper-podlodka-turbo — OpenAI-совместимый API (переменная K)
  speaches                          — Avroflex Speaches (LAN .88:8002 или HUB_URL;
                                      K опционален; model id см. SPEACHES_MODEL)
  gigaam-local                      — Avroflex GigaAM v3 (LAN .88:8003 или GIGAAM_URL)
  gemma4-e2b, gemma4-e4b, gemma4-12b — Avroflex Gemma 4 ASR (LAN .88:8004 или GEMMA_URL;
                                      сервер должен быть переключён на ту же модель)
  qwen2-audio-7b                    — Avroflex Qwen2-Audio (LAN .88:8005 или QWEN_AUDIO_URL)
  gigaam-v3                         — GigaAM через партнёрский API (переменная RPA)
  parakeet-tdt-0.6b-v3              — локальная модель, считается ПРЯМО ЗДЕСЬ на GPU
                                      (нет сети → latency_s это чистый инференс,
                                       он НЕ сравним с сетевыми движками)
  nova-2, nova-3                    — Deepgram (переменная DG)
  speechcore                        — асинхронный пайплайн upload → poll (SC)

Пример:
  K=sk-... python harness/run_bench.py --engine whisper-1 --limit 150
  DG=...   python harness/run_bench.py --engine nova-2 --limit 150
  python harness/run_bench.py --engine speaches --domain podlodka --limit 20
  python harness/run_bench.py --engine gigaam-local --domain fleurs --limit 50
  python harness/run_bench.py --engine gemma4-e2b --domain podlodka --limit 20
"""
from __future__ import annotations

import argparse
import io
import json
import os
import time
from pathlib import Path

import requests
import soundfile as sf
from datasets import Audio, load_dataset

# Домены. Один корпус ничего не доказывает: чистая начитка, аудиокнига,
# спонтанная лекция и бытовая команда — четыре разные задачи, и порядок движков
# на них может отличаться. Поэтому WER считается ПО ДОМЕНАМ, без общего среднего.
DOMAINS = {
    # чистая студийная начитка новостных фраз
    "fleurs": {"id": "google/fleurs", "config": "ru_ru", "split": "test", "text": "transcription"},
    # аудиокниги: чтение, но другой стиль и длинные фразы
    "rulibrispeech": {"id": "bond005/rulibrispeech", "config": None, "split": "test", "text": "transcription"},
    # спонтанная речь: лекции и интервью подкаста
    "podlodka": {"id": "bond005/podlodka_speech", "config": None, "split": "test", "text": "transcription"},
    # бытовые команды устройствам: короткие, разговорные, шумные
    "sova_rudevices": {"id": "bond005/sova_rudevices", "config": None, "split": "test", "text": "transcription"},
}

DATASET = "google/fleurs"
CONFIG = "ru_ru"

OPENAI_BASE = os.environ.get("HUB_URL", "https://api.neuraldeep.ru/v1")
# Avroflex Speaches на llm-server; HUB_URL перекрывает и его.
SPEACHES_BASE = os.environ.get("HUB_URL", "http://192.168.0.88:8002/v1")
SPEACHES_MODEL = os.environ.get(
    "SPEACHES_MODEL", "deepdml/faster-whisper-large-v3-turbo-ct2"
)
# Avroflex GigaAM на том же GPU1; отдельный URL, чтобы не путать с HUB_URL/Speaches.
GIGAAM_LOCAL_BASE = os.environ.get("GIGAAM_URL", "http://192.168.0.88:8003/v1")
GIGAAM_LOCAL_MODEL = os.environ.get("GIGAAM_MODEL", "gigaam-v3")
# Gemma 4 multimodal ASR (один контейнер :8004, модель переключается в compose).
GEMMA_BASE = os.environ.get("GEMMA_URL", "http://192.168.0.88:8004/v1")
# Qwen2-Audio ASR (:8005).
QWEN_AUDIO_BASE = os.environ.get("QWEN_AUDIO_URL", "http://192.168.0.88:8005/v1")
QWEN_AUDIO_MODEL = os.environ.get("QWEN_AUDIO_MODEL", "qwen2-audio-7b")
SPEECHCORE_BASE = os.environ.get("SC_URL", "https://speechcore.neuraldeep.ru/api")
DEEPGRAM_URL = "https://api.deepgram.com/v1/listen"

OPENAI_ENGINES = ("whisper-1", "whisper-podlodka-turbo")
SPEACHES_ENGINES = ("speaches",)
GIGAAM_LOCAL_ENGINES = ("gigaam-local",)
GEMMA_ENGINES = ("gemma4-e2b", "gemma4-e4b", "gemma4-12b")
QWEN_AUDIO_ENGINES = ("qwen2-audio-7b",)
# GigaAM — русская ASR SberDevices, обучена на русском целенаправленно, а не как
# один из ста языков. Живёт на отдельном OpenAI-совместимом эндпоинте.
RPA_BASE = os.environ.get("RPA_URL", "https://private.rpa.icu/v1")
RPA_ENGINES = ("gigaam-v3",)
DEEPGRAM_ENGINES = ("nova-2", "nova-3")
# Локальные движки: модель грузится в этот же процесс, запрос никуда не уходит.
# Держим отдельно, потому что latency у них означает другое — только инференс,
# без сети, очереди и загрузки файла.
LOCAL_ENGINES = ("parakeet-tdt-0.6b-v3",)
_local_model = None


def duration_sec(wav: bytes) -> float:
    info = sf.info(io.BytesIO(wav))
    return info.frames / info.samplerate


def _openai_headers() -> dict[str, str]:
    """Bearer только если K задан: LAN Speaches без auth, llm-88 — с Bearer."""
    key = os.environ.get("K")
    return {"Authorization": f"Bearer {key}"} if key else {}


def call_openai(engine: str, wav: bytes) -> str:
    if "K" not in os.environ:
        raise SystemExit("нужна переменная окружения K (Bearer для OpenAI-совместимого API)")
    r = requests.post(
        f"{OPENAI_BASE}/audio/transcriptions",
        headers=_openai_headers(),
        files={"file": ("sample.wav", wav, "audio/wav")},
        data={"model": engine, "language": "ru", "response_format": "json"},
        timeout=300,
    )
    r.raise_for_status()
    return r.json().get("text") or ""


def call_speaches(_engine: str, wav: bytes) -> str:
    """Avroflex Speaches: alias `speaches` → длинный model id без слэша в имени файла."""
    r = requests.post(
        f"{SPEACHES_BASE}/audio/transcriptions",
        headers=_openai_headers(),
        files={"file": ("sample.wav", wav, "audio/wav")},
        data={"model": SPEACHES_MODEL, "language": "ru", "response_format": "json"},
        timeout=300,
    )
    r.raise_for_status()
    return r.json().get("text") or ""


def call_gigaam_local(_engine: str, wav: bytes) -> str:
    """Avroflex GigaAM v3 на .88:8003 (тот же 5060 Ti, что Speaches)."""
    r = requests.post(
        f"{GIGAAM_LOCAL_BASE}/audio/transcriptions",
        headers=_openai_headers(),
        files={"file": ("sample.wav", wav, "audio/wav")},
        data={"model": GIGAAM_LOCAL_MODEL, "language": "ru", "response_format": "json"},
        timeout=300,
    )
    r.raise_for_status()
    return r.json().get("text") or ""


def call_gemma(engine: str, wav: bytes) -> str:
    """Avroflex Gemma 4 ASR на .88:8004; model id = alias движка."""
    r = requests.post(
        f"{GEMMA_BASE}/audio/transcriptions",
        headers=_openai_headers(),
        files={"file": ("sample.wav", wav, "audio/wav")},
        data={"model": engine, "language": "ru", "response_format": "json"},
        timeout=600,
    )
    r.raise_for_status()
    return r.json().get("text") or ""


def call_qwen_audio(_engine: str, wav: bytes) -> str:
    """Avroflex Qwen2-Audio на .88:8005."""
    r = requests.post(
        f"{QWEN_AUDIO_BASE}/audio/transcriptions",
        headers=_openai_headers(),
        files={"file": ("sample.wav", wav, "audio/wav")},
        data={"model": QWEN_AUDIO_MODEL, "language": "ru", "response_format": "json"},
        timeout=600,
    )
    r.raise_for_status()
    return r.json().get("text") or ""


def call_rpa(engine: str, wav: bytes) -> str:
    r = requests.post(
        f"{RPA_BASE}/audio/transcriptions",
        headers={"Authorization": f"Bearer {os.environ['RPA']}"},
        files={"file": ("sample.wav", wav, "audio/wav")},
        data={"model": engine},
        timeout=300,
    )
    r.raise_for_status()
    return r.json().get("text") or ""


def call_deepgram(engine: str, wav: bytes) -> str:
    r = requests.post(
        DEEPGRAM_URL,
        params={"model": engine, "language": "ru", "punctuate": "true", "smart_format": "false"},
        headers={"Authorization": f"Token {os.environ['DG']}", "Content-Type": "audio/wav"},
        data=wav, timeout=300,
    )
    r.raise_for_status()
    return r.json()["results"]["channels"][0]["alternatives"][0]["transcript"]


def call_speechcore(engine: str, wav: bytes) -> str:
    """⚠️ Задержка тут включает загрузку, очередь и поллинг — она НЕ сравнима
    с одиночным POST у остальных движков."""
    head = {"Authorization": f"Bearer {os.environ['SC']}"}
    up = requests.post(
        f"{SPEECHCORE_BASE}/upload", headers=head,
        files={"file": ("sample.wav", wav, "audio/wav")},
        params={"model": "large-v3", "language": "ru", "diarize": "false"},
        timeout=300,
    )
    up.raise_for_status()
    task_id = up.json()["task_id"]

    deadline = time.time() + 600
    while time.time() < deadline:
        time.sleep(2)
        st = requests.get(f"{SPEECHCORE_BASE}/transcriptions/{task_id}/status",
                          headers=head, timeout=60)
        st.raise_for_status()
        status = (st.json().get("status") or "").lower()
        if status in ("completed", "done", "success"):
            break
        if status in ("failed", "error"):
            raise RuntimeError(f"задача упала: {status}")
    else:
        raise TimeoutError("не дождались завершения")

    res = requests.get(f"{SPEECHCORE_BASE}/transcriptions/{task_id}", headers=head, timeout=60)
    res.raise_for_status()
    body = res.json()
    text = body.get("text") or body.get("transcription") or ""
    if not text and isinstance(body.get("segments"), list):
        text = " ".join(seg.get("text", "") for seg in body["segments"])
    return text


def call_local(engine: str, wav: bytes) -> str:
    """Parakeet TDT — многоязычная ASR NVIDIA (25 языков, русский в их числе).
    Язык определяет сама, подсказку не принимает: у остальных движков мы
    честно передаём language=ru, здесь такой ручки нет — это её свойство,
    а не поблажка."""
    global _local_model
    import numpy as np
    import torch

    if _local_model is None:
        from transformers import AutoModelForTDT, AutoProcessor
        mid = os.environ.get("LOCAL_MODEL", "nvidia/parakeet-tdt-0.6b-v3")
        proc = AutoProcessor.from_pretrained(mid)
        model = AutoModelForTDT.from_pretrained(mid, dtype="auto", device_map="cuda")
        model.eval()
        _local_model = (proc, model)
    proc, model = _local_model

    audio, sr = sf.read(io.BytesIO(wav), dtype="float32")
    if audio.ndim > 1:                      # модель ждёт моно
        audio = audio.mean(axis=1)
    want = proc.feature_extractor.sampling_rate
    if sr != want:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=want)
    inputs = proc(audio, sampling_rate=want, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, return_dict_in_generate=True)
    text = proc.batch_decode(out.sequences, skip_special_tokens=True)[0]
    return (text or "").strip()


def pick_caller(engine: str):
    if engine in LOCAL_ENGINES:
        return call_local
    if engine in SPEACHES_ENGINES:
        return call_speaches
    if engine in GIGAAM_LOCAL_ENGINES:
        return call_gigaam_local
    if engine in GEMMA_ENGINES:
        return call_gemma
    if engine in QWEN_AUDIO_ENGINES:
        return call_qwen_audio
    if engine in OPENAI_ENGINES:
        return call_openai
    if engine in RPA_ENGINES:
        return call_rpa
    if engine in DEEPGRAM_ENGINES:
        return call_deepgram
    if engine == "speechcore":
        return call_speechcore
    raise SystemExit(f"неизвестный движок: {engine}")


def endpoint_for(engine: str) -> str:
    if engine in SPEACHES_ENGINES:
        return SPEACHES_BASE
    if engine in GIGAAM_LOCAL_ENGINES:
        return GIGAAM_LOCAL_BASE
    if engine in GEMMA_ENGINES:
        return GEMMA_BASE
    if engine in QWEN_AUDIO_ENGINES:
        return QWEN_AUDIO_BASE
    return {
        "whisper-1": OPENAI_BASE,
        "whisper-podlodka-turbo": OPENAI_BASE,
        "nova-2": DEEPGRAM_URL,
        "nova-3": DEEPGRAM_URL,
        "gigaam-v3": RPA_BASE,
        "speechcore": SPEECHCORE_BASE,
    }.get(engine, "local-gpu")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--domain", default="fleurs", choices=sorted(DOMAINS))
    ap.add_argument("--split", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    call = pick_caller(args.engine)
    domain = DOMAINS[args.domain]
    split = args.split or domain["split"]
    out_path = Path(args.out or f"results/{args.domain}/{args.engine}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    load_args = [domain["id"]] + ([domain["config"]] if domain["config"] else [])
    stream = load_dataset(*load_args, split=split, streaming=True)
    stream = stream.cast_column("audio", Audio(decode=False))

    written = failed = 0
    started = time.time()
    with out_path.open("w", encoding="utf-8") as fh:
        # Шапка: без неё через месяц не понять, на какой версии всё считалось.
        meta = {
            "engine": args.engine, "domain": args.domain,
            "dataset": domain["id"], "config": domain["config"], "split": split,
            "limit": args.limit, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "endpoint": endpoint_for(args.engine),
        }
        if args.engine in SPEACHES_ENGINES:
            meta["model"] = SPEACHES_MODEL
        if args.engine in GIGAAM_LOCAL_ENGINES:
            meta["model"] = GIGAAM_LOCAL_MODEL
        if args.engine in GEMMA_ENGINES:
            meta["model"] = args.engine
        if args.engine in QWEN_AUDIO_ENGINES:
            meta["model"] = QWEN_AUDIO_MODEL
        fh.write(json.dumps({"_meta": meta}, ensure_ascii=False) + "\n")

        for idx, item in enumerate(stream):
            if idx >= args.limit:
                break
            wav = item["audio"]["bytes"]
            ref = item.get(domain["text"]) or item.get("raw_transcription") or ""
            t0 = time.time()
            try:
                hyp = call(args.engine, wav)
                error = None
            except Exception as exc:
                hyp, error = "", f"{type(exc).__name__}: {str(exc)[:200]}"
                failed += 1
            fh.write(json.dumps({
                "idx": idx, "duration_s": round(duration_sec(wav), 3),
                "ref_raw": ref, "hyp_raw": hyp,
                "latency_s": round(time.time() - t0, 3), "error": error,
            }, ensure_ascii=False) + "\n")
            written += 1
            if written % 25 == 0:
                print(f"  {written}/{args.limit}  ошибок {failed}  "
                      f"{time.time() - started:.0f}с", flush=True)

    print(f"{args.engine}: записано {written}, ошибок {failed} → {out_path}")


if __name__ == "__main__":
    main()
