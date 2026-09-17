# Resource snapshots — ASR comparison on .88 GPU1 (RTX 5060 Ti 16 GB)

Sequential runs: Speaches/GigaAM stopped while Gemma/Qwen/Ollama/Omni occupy GPU1.

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
| **qwen25-omni-7b** (bnb4, talker on) | **~7000–8340 MiB** | port 8007; GPTQ-Int4 load failed → fallback bnb4 |

## WER + speed

### Podlodka n=20

| engine | WER | 95% CI | ×RT | med lat |
|---|---:|---|---:|---:|
| gigaam-local | **7.30%** | [4.68; 9.96] | 40.5× | 0.51s |
| speaches | 7.38% | [5.04; 10.03] | 9.1× | 1.66s |
| gemma4-e4b (offload) | 9.27% | [7.00; 11.67] | 0.21×† | 89s |
| gemma4-e2b | 9.70% | [7.29; 12.51] | 10.9× | 2.03s |
| **gemma4-12b-qat** | **10.73%** | [7.69; 14.95] | **8.0×** | 2.85s |
| gemma4-e4b-4bit | 11.07% | [8.51; 14.05] | 6.3× | 3.46s |
| gemma4-e4b-qat | 11.67% | [8.61; 14.84] | 6.3× | 3.50s |
| qwen25-omni-7b (bnb4) | 23.52% | [13.04; 33.87] | 13.5× | 1.67s |
| qwen2-audio-7b | 35.88% | [29.82; 41.34] | 5.3× | 4.30s |
| gemma4-12b (bnb4) | 45.92% | [22.41; 88.97] | 3.1× | 6.39s |
| gemma4-e4b-8bit | 63.43% | [53.01; 72.55] | 4.6× | 5.21s |

### FLEURS n=50

| engine | WER | 95% CI | ×RT | med lat |
|---|---:|---|---:|---:|
| speaches | **4.07%** | [2.63; 5.75] | 12.9× | 0.92s |
| gigaam-local | 5.11% | [3.32; 7.20] | 34.1× | 0.36s |
| **gemma4-12b-qat** | **5.39%** | [3.31; 8.10] | **6.7×** | 1.60s |
| gemma4-e2b | 5.96% | [3.99; 8.22] | 12.4× | 0.94s |
| gemma4-e4b-qat | 7.10% | [4.69; 9.95] | 6.8× | 1.68s |
| qwen25-omni-7b (bnb4) | 7.19% | [4.87; 9.81] | 11.3× | 1.06s |
| gemma4-e4b-4bit | 7.28% | [5.12; 9.88] | 7.1× | 1.61s |
| qwen2-audio-7b | 31.50% | [25.82; 37.89] | 5.3× | 2.15s |
| gemma4-12b (bnb4) | 34.15% | [23.55; 46.13] | 3.7× | 2.68s |
| gemma4-e4b-8bit | 75.59% | [65.62; 85.03] | 5.4× | 1.66s |

† offload — ignore speed.  
`gemma4-e4b` (bf16) FLEURS not run: same offload path as Podlodka (~0.21×RT), not useful for speed ranking.

## Takeaways

- **Best multimodal: `gemma4-12b-qat` (Ollama QAT Q4_0)** — FLEURS 5.39% (full n=50); Podlodka ~+3.4 п.п. vs GigaAM.
- **E4B 4-bit PTQ ≈ E4B QAT** (~11% / ~7%); both usable with ~6.5 GiB headroom. Prefer 4-bit over 8-bit.
- **E4B bnb 8-bit is a dead end** for ASR (Podlodka 63% / FLEURS **76%**) despite fitting VRAM — confirmed on both domains.
- **HF bnb4 on non-QAT 12B** fails; official QAT GGUF does not.
- **Qwen2.5-Omni-7B bnb4**: FLEURS OK (~7.2%, ≈ E4B-QAT); **Podlodka weak (23.5%)** — worse than all Gemma QAT and far behind Speaches/GigaAM. Faster than Gemma (~11–13×RT). Official GPTQ-Int4 needs custom low-VRAM loader; simple `from_pretrained` failed → used bnb4.
- Production: **GigaAM** (speed + Podlodka) / **Speaches** (FLEURS). QAT 12B remains the multimodal pick; Omni not competitive on spontaneous RU.
