# Resource snapshots — ASR comparison on .88 GPU1 (RTX 5060 Ti 16 GB)
# Collected during sequential model runs (Speaches/GigaAM stopped while Gemma/Qwen load).

## Baselines (warm, both running earlier)
| engine | VRAM GPU1 | host RAM (docker) | image |
|---|---:|---:|---:|
| speaches (CT2 turbo) | ~2208 MiB | ~418 MiB | ~8.3 GB |
| gigaam-local (v3_e2e_rnnt) | ~674 MiB | ~2.14 GiB | ~13 GB |
| combined Speaches+GigaAM | ~2910 MiB | — | — |

## gemma4-e2b (google/gemma-4-E2B-it, bf16, port 8004)
| metric | value |
|---|---|
| VRAM GPU1 | ~10129 MiB |
| host RAM | ~3.01 GiB |
| image | 13.7 GB |
| idle CPU | ~0.3% |

## WER (same harness)
### Podlodka n=20
- gigaam-local 7.30%
- speaches 7.38%
- gemma4-e2b 9.70% (worse than both, paired CI significant)

### FLEURS n=50
- speaches 4.07%
- gigaam-local 5.11%
- gemma4-e2b 5.96% (CI overlap with both)
