# KcatNet — emulator_bench report

## Method
KcatNet (Pan et al., Genome Biology 2026) predicts log10(kcat) from an enzyme sequence and a substrate SMILES.
The enzyme is a residue graph. Node features are ProtT5 (1024-d) and ESM2-650M (1280-d) per-residue embeddings.
Edges come from the ESM2 contact map (threshold 0.5) and from sequence neighbours at offsets 1 and 2.
The substrate has atom features (type embedding and 43-d properties) and a 1024-d SMILES Transformer vector.
A GCN with mincut partitioning groups residues into regions.
A cross-attention module, repeated 3 times, links substrate atoms to enzyme regions.
An MLP gives the scalar output.

## Original default settings
The paper (Methods, p.19-20) gives data and architecture only, so the code is the source.

| Setting | Value | Source |
|---|---|---|
| Epochs / batch / seed | 80 / 16 / 666 | `train.py:21-23` |
| Optimiser | AdamW, betas (0.9, 0.999), eps 1e-8, no amsgrad | `models/model_kcat.py:106` |
| Weight decay | 1e-2 (PyTorch default) | `models/model_kcat.py:106` |
| Learning rate | cosine per iteration from 1e-4 to 0, no warmup | `utils/trainer.py` `get_lr`; `schedule_lr=True` default |
| Gradient clip | 1.0 | `config_KcatNet.json`, `train.py` |
| Loss | 0.9 MSE + 0.1 cluster loss | `utils/trainer.py` |
| Validation / early stopping | every epoch, best by val RMSE / none | `utils/trainer.py` |
| Precision | fp32 (only ESM2 embeddings are stored as fp16, as in the original) | `utils/protein_init.py`, `utils/trainer.py` |
| Model sizes | hidden 200, heads 5, K [3,10,30], total_layer 3, pre 2, post 1, dropout 0, attention dropout 0.2 | `config_KcatNet.json` |
| Sequence length | first 1000 residues | `utils/protein_init.py` |

Feature extraction follows the original code:
- The SMILES Transformer runs on CPU, one molecule at a time. The original positional encoding depends on batch size, so batching changes the output.
- ProtT5 runs in fp16 with the original batching rule: longest first, at most 100 sequences and 4000 residues per batch. The result depends on batch composition, as in the original, so bit-level equality needs the same set of sequences.
- ProtT5 embeddings and contact-map edge weights are stored as fp32. ESM2 embeddings are stored as fp16.

Paper and code differences:
- The paper splits randomly 80/20. The wrapper uses the EMULaToR train/val/test files.
- `config_KcatNet.json` lists weight decay 1e-4 and `schedule_lr: false`. The code ignores both, so the wrapper follows the code.

## What the wrapper does now
It trains KcatNet with the fixed original recipe for 3 seeds (666, 777, 888) on every kcat split folder.
It writes predictions and metrics to `--out_dir`, then an aggregator summarises them.

## Files removed or changed (in `emulator_bench/`)
- Removed: `tune_optuna.py`, `launch_parallel_optuna.py`, `launch_parallel_retrain_from_optuna.py`,
  `kcatnet_repo_defaults_hparams.json`, `predict_single_target.py` (training already writes predictions).
- `train_single_target_tvt.py`: rewritten with the fixed recipe above. DDP, scheduler options and mixed precision are gone.
- `run_split_benchmarks.py`: seeds default to 666 777 888. Hyperparameter flags are gone. Results go to `--out_dir`.
- `common.py`: EMULaToR layout defaults built from the repo path. `uniprot_time_splits` added. `group_shuffle_splits` dropped.
- `aggregate_kcatnet_results.py`: rewritten for the new output layout.
- `README.md`, `commands.txt`: rewritten, with no Optuna text.
- No original code outside `emulator_bench` was edited.

## Deviations (speed-ups only)
- The embedding cache is shared across splits and seeds. It uses the original feature code.
- The PNA degree histogram is cached per train file.
- `--num_workers` is optional. Data loading order is the same for any worker count.
- Training uses `fast_models.FastKcatNet`, which has the same parameters and outputs as `models/model_kcat.py` (original code unchanged). Cache files are written uncompressed. Old compressed files still load.

## Commands
From the KcatNet root, with `PY=/home/adhil/.conda/envs/mldb/bin/python`:
```
$PY emulator_bench/run_split_benchmarks.py --device cuda:0 --cache_device cuda:0   # 3 seeds x all kcat splits
$PY emulator_bench/aggregate_kcatnet_results.py
```
Only the kcat task applies to KcatNet.

## Smoke test
- Ran on GPU 1. Outputs only in the session scratchpad (`bench_smoke/KcatNet`).
- 3 tiny split folders (random grouped sequence, time, enzyme sequence), 2 seeds, 1 epoch.
- All 6 runs trained, and the aggregator summarised them.
- ProtT5 and ESM2 weights (about 14 GB) are not on disk. The protein cache held random embeddings in the real format, so the protein caching step did not run. Ligand embeddings came from the real SMILES Transformer. The metric values mean nothing.

## Changes after review

- `get_or_compute_pna_degrees` writes the cache to a temp file in the same folder, then `os.replace`.
- Re-smoke (random grouped sequence, seed 666, 1 epoch, GPU 1, fresh PNA cache): cache written, no temp file left, training ran. Output: `bench_smoke/KcatNet/out2`.

## Open issues
- The paper appendix says fp16 for KcatNet on V100. The original code uses fp32, so the wrapper uses fp32.
- The first real cache run downloads ProtT5 (public Rostlab checkpoint; the original used a private path) and ESM2. This step is not tested.
- Memory at full scale with batch size 16 is not measured.
- The real ProtT5 caching pass is untested.
