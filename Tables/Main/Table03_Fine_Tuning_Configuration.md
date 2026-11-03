# Table 3: Reported transformer fine-tuning configuration and verified training-seed scope.

| Setting | Value |
| --- | --- |
| Backbones | BanglaBERT, multilingual BERT (mBERT) |
| Optimizer | AdamW |
| Learning rate | 5e-5 |
| Weight decay | 0.01 |
| Scheduler | Linear decay, 500 warmup steps (reported) |
| Epoch budget | 3 |
| Batch size (per device) | 16 |
| Maximum sequence length | 256 Fake News / 128 Hate Speech |
| Precision | FP16 (reported) |
| Checkpoint selection | Highest validation macro-F1 |
| Independent random seeds | 42, 52, 62 |

**Notes.** Seeds, best-epoch selection and full-test metrics were checked in the archived run summaries. Optimizer/scheduler/hardware entries follow the submitted method and should be verified against the original training logs before manuscript final sign-off.
