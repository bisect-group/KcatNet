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
enzyme_structure_splits/threshold_<t>/{train,val,test}.parquet
substrate_splits/threshold_<t>/{train,val,test}.parquet
conformer_cosine_splits/threshold_<t>/{train,val,test}.parquet
```

Columns used: `sequence`, `smiles`, `log10_value`.
Thresholds are discovered from disk. The alias `random_splits` expands to both `random_splits_grouped_*` folders.

Outputs (default `<EMULaToR>/outputs/baselines/KcatNet`, set with `--out_dir`):

- `embeddings/`: shared cache: `proteins_v3/` (ProtT5 + ESM2), `ligands_v3/` (SMILES Transformer), `pna_degrees_v3/`.
  Protein keys cover the ProtT5 and ESM2 models, ESM layer, ProtT5 dtype and length cap.
  Change `CACHE_VERSION` in `common.py` to force a rebuild.
- `<task>/<split_group>/<split_name>/seed_<n>/` (`<task>` is the name of the `--base_dir` folder, `kcat`; `<split_name>` is `all` for flat split groups): `bestmodel.pth`, `final_results_{val,test}.csv`, `pred_label_{val,test}.csv`, `logfile.csv`, `run_summary.csv`
- `aggregate_kcatnet_results.csv`: mean and std over seeds

## Run

Use the EMULaToR env, from the KcatNet submodule root.

```bash
PY=/home/adhil/.conda/envs/mldb/bin/python
$PY emulator_bench/run_split_benchmarks.py --dry_run            # list jobs
$PY emulator_bench/run_split_benchmarks.py --device cuda:0      # cache, then all splits x seeds 666 777 888
$PY emulator_bench/run_split_benchmarks.py --gpus 0 1 2 --runs_per_gpu 2 --amp   # runs spread over 3 GPUs, bf16
$PY emulator_bench/aggregate_kcatnet_results.py                 # summarize seeds
```

Useful flags of `run_split_benchmarks.py`: `--base_dir`, `--out_dir`, `--embeddings_dir`, `--split_groups`,
`--thresholds`, `--seeds`, `--epochs` (80), `--batch_size` (16), `--num_workers`, `--skip_cache`, `--overwrite`, `--gpus` (one GPU per run), `--runs_per_gpu`,
`--amp` (bf16, or fp16 on pre-Ampere GPUs; default is the original fp32).
Finished runs are skipped. All commands are in `commands.txt`.

## Verify equivalence

```bash
emulator_bench/verify_equivalence.sh        # CPU, synthetic graphs, fails on any skipped test
emulator_bench/verify_equivalence.sh --gpu  # also runs the Triton test; needs a CUDA device
```

Run it in the `mldb` env. It needs no data or download. Do not `pip install fair-esm`: it overwrites the
Biohub `esm` in `mldb`. `esm_deps.py` makes fair-esm importable for the embedding cache, and these tests do not need it.

`tests/test_original_equivalence.py` compares `FastKcatNet` with the authors' `models/model_kcat.py` on the same
weights and batch. Both losses and all gradients are checked.

- CPU float64 with plain-torch kernels: max error below 1e-6 of the largest output, 1e-7 for gradients.
  The cluster loss sums in float32 in both models, so this is the best bound.
- CPU float32: 1e-5 for outputs, 2e-3 for gradients.
- GPU float32 with the Triton kernels against the plain-torch original: the prediction gap (step 0) must be at most 10 times its noise gap. Every output, loss and gradient must stay within the CPU tolerance (1e-4 for outputs and losses, 1e-3 for gradients, relative to the largest value of each). A tensor whose own noise floor is above that tolerance is held to 10 times its own floor instead. The floor is the gap between the original and the original with all inputs perturbed by 1e-6 (worst of 8 draws). Some gradients of the toy graphs are discontinuous in the inputs: `inter_convs.0.residue_mlp1.weight` changes by about 5% when a 1e-6 input change moves a tie, in the original as well as with Triton. Only that tensor gets the larger budget.
- The sorted edge order of the dataset does not change the original output. The PNA degree histogram is exact.

Production precision is fp32 with `--tf32` and `--amp` off (`amp: false`, `tf32: false` in each `run_state.json`).
That is the path the GPU test checks. bf16 and TF32 give no gain and are not tested.
`emulator_bench/verify_equivalence.sh --negative-control` scales the weights of the Triton model by 1.05 and runs only the GPU test. It exits 0 only if that test fails, so a pass would show a bound too loose to catch a 5% error.
With `EMULATOR_SYNTHETIC_FIXTURE=1` the tests use synthetic graphs. Without it they also use real cached
graphs from `outputs/baselines/KcatNet/embeddings` when present.
The parent repo has more checks: `tests/baselines/test_kcatnet_official_equivalence.py` and
`tests/baselines/test_kcatnet_triton_kernels.py`. They are not part of this script.

## ESM package

The mldb env holds the Biohub `esm` fork. It has no `esm.pretrained.load_model_and_alphabet`, which KcatNet needs.
The launchers call `setup_esm(out_dir)`. If fair-esm is missing, it downloads the pinned wheel (sha256 checked)
into `<out_dir>/pydeps` and puts it first on the path. Children inherit `PYTHONPATH`.
`TORCH_HOME` defaults to `<out_dir>/torch_home`, so the ESM2 weights stay under `out_dir`. No env var needs setting by hand.

## Files

- `esm_deps.py`: `setup_esm(out_root)` makes fair-esm 2.0.0 importable (see below)
- `common.py`: paths, split discovery, cache keys
- `cache_embeddings.py`, `feature_pipeline.py`: build the embedding cache with the original feature code
- `dataset.py`: cached dataset and PNA degree histogram cache
- `fast_models.py`: `KcatNet` subclasses with the same parameters and outputs but fewer host syncs and copies
- `train_single_target_tvt.py`: one split, one seed, original training recipe
- `run_split_benchmarks.py`: sweep over splits and seeds
- `aggregate_kcatnet_results.py`: seed mean and std
