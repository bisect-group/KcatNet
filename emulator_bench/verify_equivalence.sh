#!/usr/bin/env bash
# Checks that FastKcatNet matches the authors' KcatNet. Usage: verify_equivalence.sh [--gpu | --negative-control]
# Expected env: mldb (torch, torch_geometric, torch_scatter, pandas, pytest). The script does not create envs.
# Never pip install fair-esm: it overwrites the esm package of mldb. These tests do not need ESM.
set -euo pipefail
cd "$(dirname "$0")/.."
echo "Expected conda env: mldb"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export EMULATOR_SYNTHETIC_FIXTURE=1
files=(tests/test_original_equivalence.py)
if [[ ${1:-} == --negative-control ]]; then
    # Scales the candidate weights by 1.05: every gpu test must fail. A passing gpu test has a vacuous bound.
    export EQUIV_NEGATIVE_CONTROL=1
    status=0
    out=$(python -m pytest "${files[@]}" -q -rs -p no:cacheprovider --color=no -o addopts="" -m gpu -s 2>&1) || status=$?
    echo "$out"
    if [[ $status -ne 1 ]] || grep -Eq "[0-9]+ (passed|skipped|errors?)" <<<"$out"; then
        echo "FAIL: a gpu test did not fail under the negative control"
        exit 1
    fi
    echo "OK: every gpu test failed under the negative control"
    exit 0
fi
marker=(-m "not gpu")
extra=()
allow_skip=
if [[ ${1:-} == --gpu ]]; then
    marker=()
    extra=(-s)  # shows the printed gap and floor
    if ! python -c "import torch,sys; sys.exit(not torch.cuda.is_available())"; then
        echo "No CUDA device visible: gpu tests will be skipped."
        allow_skip=1
    fi
else
    export CUDA_VISIBLE_DEVICES=""
fi
out=$(python -m pytest "${files[@]}" -q -rs -p no:cacheprovider --color=no -o addopts="" "${marker[@]}" "${extra[@]}" 2>&1) || { echo "$out"; exit 1; }
echo "$out"
if [[ -z ${allow_skip:-} ]] && grep -Eq "[0-9]+ skipped" <<<"$out"; then
    echo "FAIL: tests were skipped"
    exit 1
fi
