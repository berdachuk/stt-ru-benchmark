# Resource snapshots — ASR comparison on .88 GPU1 (RTX 5060 Ti 16 GB)

Sequential runs: Speaches/GigaAM stopped while Gemma/Qwen/Ollama occupy GPU1.

## Resources

| engine | VRAM GPU1 | notes |
|---|---:|---|
| speaches (CT2 turbo) | ~2208 MiB | baseline |
| gigaam-local (v3_e2e_rnnt) | ~674 MiB | baseline |
| Speaches+GigaAM together | ~2910 MiB | fits easily |
| gemma4-e2b (bf16, HF) | ~10129 MiB | port 8004 |
| gemma4-e4b (bf16 offload) | CPU offload | RTF≈4.7 — ignore speed |
| gemma4-e4b-8bit (bnb) | ~11269 MiB | ~4.5 GiB free; **WER broken** |
| gemma4-e4b-4bit (bnb PTQ) | ~9261 MiB | ~6.6 GiB free |
| gemma4-e4b-qat (HF QAT + bnb4) | ~9365 MiB | ~6.5 GiB free |
| gemma4-12b (bnb4, HF) | ~8907 MiB | poor ASR quality |
| **gemma4-12b-qat** (Ollama Q4_0 GGUF) | **~9193 MiB** | ~6.7 GiB free; `gemma4:12b-it-qat` |
| qwen2-audio-7b Instruct 4-bit | ~6615–7777 MiB | port 8005 |

## WER + speed

### Podlodka n=20

| engine | WER | 95% CI | ×RT | med lat |
|---|---:|---|---:|---:|
| gigaam-local | **7.30%** | [4.80; 9.91] | 40.5× | 0.51s |
| speaches | 7.38% | [5.28; 9.95] | 9.1× | 1.66s |
| gemma4-e4b (offload) | 9.27% | [6.98; 11.69] | 0.21×† | 89s |
| gemma4-e2b | 9.70% | [7.34; 12.58] | 10.9× | 2.03s |
| **gemma4-12b-qat** | **10.73%** | [7.55; 15.21] | **8.0×** | 2.85s |
| gemma4-e4b-4bit | 11.07% | [8.55; 14.20] | 6.3× | 3.46s |
| gemma4-e4b-qat | 11.67% | [8.82; 14.98] | 6.3× | 3.50s |
| qwen2-audio-7b | 35.88% | [29.93; 41.57] | 5.3× | 4.30s |
| gemma4-12b (bnb4) | 45.92% | [23.17; 86.83] | 3.1× | 6.39s |
| gemma4-e4b-8bit | 63.43% | [52.66; 72.70] | 4.6× | 5.21s |

### FLEURS n=50 (12b-qat: n=48)

| engine | WER | 95% CI | ×RT | med lat |
|---|---:|---|---:|---:|
| speaches | **4.07%** | [2.74; 5.83] | 12.9× | 0.92s |
| **gemma4-12b-qat** | **4.88%** | [2.93; 7.26] | **7.1×** | 1.59s |
| gigaam-local | 5.11% | [3.39; 7.24] | 34.1× | 0.36s |
| gemma4-e2b | 5.96% | [3.98; 8.39] | 12.4× | 0.94s |
| gemma4-e4b-qat | 7.10% | [4.62; 10.03] | 6.8× | 1.68s |
| gemma4-e4b-4bit | 7.28% | [5.05; 10.03] | 7.1× | 1.61s |
| qwen2-audio-7b | 31.50% | [25.77; 38.05] | 5.3× | 2.15s |
| gemma4-12b (bnb4) | 34.15% | [23.70; 46.70] | 3.7× | 2.68s |
| gemma4-e4b-8bit FLEURS | — | — | — | **skipped** (Podlodka already shows broken ASR; dataset hang on parallel try) |

† offload — ignore speed.

## Takeaways

- **Best multimodal: `gemma4-12b-qat` (Ollama QAT Q4_0)** — FLEURS ≈ Speaches/GigaAM; Podlodka ~+3 п.п. vs GigaAM.
- **E4B 4-bit PTQ ≈ E4B QAT** (~11% / ~7%); both usable with ~6.5 GiB headroom. Prefer 4-bit over 8-bit.
- **E4B bnb 8-bit is a dead end** for ASR (63% WER) despite fitting VRAM.
- **HF bnb4 on non-QAT 12B** fails; official QAT GGUF does not.
- Production: **GigaAM** (speed + Podlodka) / **Speaches** (FLEURS). QAT 12B is the first multimodal close enough for a single-stack experiment.
