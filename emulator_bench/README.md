# emulator_bench

Multi-seed evaluation of KcatNet with its original published settings on the default EMULaToR kcat splits.
The wrapper does not tune hyperparameters. See `REPORT.md` for the settings and their sources.

## Data layout

Split root (default): `<EMULaToR>/data/processed/datasets/kinetic_dataset/kcat`

```
random_splits_grouped_sequence/{train,val,test}.parquet
random_splits_grouped_smiles/{train,val,test}.parquet
uniprot_time_splits/{train,val,test}.parquet
enzyme_sequence_splits/threshold_<t>/{train,val,test}.parquet
substrate_splits/threshold_<t>/{train,val,test}.parquet
```

Columns used: `sequence`, `smiles`, `log10_value`.
Thresholds are discovered from disk. The alias `random_splits` expands to both `random_splits_grouped_*` folders.

Outputs (default `<EMULaToR>/outputs/baselines/KcatNet`, set with `--out_dir`):

- `embeddings/`: shared protein (ProtT5 + ESM2) and ligand (SMILES Transformer) cache
- `<split_group>/<split_name>/seed_<n>/`: `bestmodel.pth`, `final_results_{val,test}.csv`, `pred_label_{val,test}.csv`, `logfile.csv`, `run_summary.csv`
- `aggregate_kcatnet_results.csv`: mean and variance over seeds

## Run

Use the EMULaToR env, from the KcatNet submodule root.

```bash
PY=/home/adhil/.conda/envs/mldb/bin/python
$PY emulator_bench/run_split_benchmarks.py --dry_run            # list jobs
$PY emulator_bench/run_split_benchmarks.py --device cuda:0      # cache, then all splits x seeds 666 777 888
$PY emulator_bench/aggregate_kcatnet_results.py                 # summarize seeds
```

Useful flags of `run_split_benchmarks.py`: `--base_dir`, `--out_dir`, `--embeddings_dir`, `--split_groups`,
`--thresholds`, `--seeds`, `--epochs` (80), `--batch_size` (16), `--num_workers`, `--skip_cache`, `--overwrite`.
Finished runs are skipped. All commands are in `commands.txt`.

## Files

- `common.py`: paths, split discovery, cache keys
- `cache_embeddings.py`, `feature_pipeline.py`: build the embedding cache with the original feature code
- `dataset.py`: cached dataset and PNA degree histogram cache
- `train_single_target_tvt.py`: one split, one seed, original training recipe
- `run_split_benchmarks.py`: sweep over splits and seeds
- `aggregate_kcatnet_results.py`: seed mean and variance
