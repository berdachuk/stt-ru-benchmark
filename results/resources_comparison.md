# Resource snapshots — ASR comparison on .88 GPU1 (RTX 5060 Ti 16 GB)

Sequential runs: Speaches/GigaAM stopped while Gemma/Qwen occupy GPU1.

## Resources

| engine | VRAM GPU1 | host RAM (docker) | notes |
|---|---:|---:|---|
| speaches (CT2 turbo) | ~2208 MiB | ~418 MiB | baseline |
| gigaam-local (v3_e2e_rnnt) | ~674 MiB | ~2.14 GiB | baseline |
| Speaches+GigaAM together | ~2910 MiB | — | fits easily |
| gemma4-e2b (bf16) | ~10129 MiB | ~3.01 GiB | port 8004 |
| gemma4-e4b | GPU+CPU offload | high host RAM | 16 GB: full GPU residency failed → device_map offload; RTF≈4.7 on Podlodka |
| qwen2-audio-7b Instruct 4-bit | ~6615–7777 MiB | — | port 8005; fits only with load_in_4bit |
| gemma4-12b Instruct 4-bit | ~8907 MiB | ~3.36 GiB | port 8004; bf16 would not fit 16 GB → 4-bit |

Inference: all multimodal candidates via HF Transformers + OpenAI-compatible wrappers on LAN (.88:8004 Gemma / :8005 Qwen). Speaches CT2 :8002, GigaAM :8003.

## WER + speed (same harness)

### Podlodka n=20

| engine | WER | 95% CI | ×RT | med lat |
|---|---:|---|---:|---:|
| gigaam-local | **7.30%** | [4.80; 9.91] | 40.5× | 0.51s |
| speaches | 7.38% | [5.28; 9.95] | 9.1× | 1.66s |
| gemma4-e4b | 9.27% | [6.98; 11.69] | 0.21×† | 89s |
| gemma4-e2b | 9.70% | [7.34; 12.58] | 10.9× | 2.03s |
| qwen2-audio-7b | 35.88% | [29.93; 41.57] | 5.3× | 4.30s |
| gemma4-12b (4-bit) | 45.92% | [23.17; 86.83] | 3.1× | 6.39s |

† E4B CPU-offloaded — ignore speed.

### FLEURS n=50

| engine | WER | 95% CI | ×RT | med lat |
|---|---:|---|---:|---:|
| speaches | **4.07%** | [2.74; 5.83] | 12.9× | 0.92s |
| gigaam-local | 5.11% | [3.39; 7.24] | 34.1× | 0.36s |
| gemma4-e2b | 5.96% | [3.98; 8.39] | 12.4× | 0.94s |
| qwen2-audio-7b | 31.50% | [25.77; 38.05] | 5.3× | 2.15s |
| gemma4-12b (4-bit) | 34.15% | [23.70; 46.70] | 3.7× | 2.68s |
| gemma4-e4b | — | — | — | **skipped**: 16 GB forced CPU offload (Podlodka med lat ~89 s / ×0.21 RT); FLEURS n=50 would be hours and not comparable |

## Takeaways

- For RU ASR on 5060 Ti: **GigaAM / Speaches** remain the practical winners (WER + speed).
- **Gemma E2B** is the only multimodal option close on FLEURS (CI overlaps Speaches/GigaAM); on Podlodka it is ~+2 п.п. worse (significant).
- **E4B** ≈ E2B WER but was unusable speed-wise under memory pressure.
- **Qwen2-Audio-7B-Instruct** and **Gemma 12B 4-bit** are far behind (~30–46% WER); usable as capability checks, not production ASR.
- Qwen needed `audio=` (not `audios=`) for Transformers; otherwise it ignored the waveform.
