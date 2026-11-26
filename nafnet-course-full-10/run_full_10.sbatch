#!/bin/bash
#SBATCH --job-name=nafnet-full
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=slurm-full-%j.out
#SBATCH --error=slurm-full-%j.err

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit the job from the extracted project directory.}"
source /etc/profile
if command -v module >/dev/null 2>&1 && ! command -v conda >/dev/null 2>&1; then
    module load miniforge3
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate nafnet-pilot

export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export PYTHONUNBUFFERED=1

epochs="${NAFNET_EPOCHS:-10}"
run_dir="${1:-runs/full-${epochs}-${SLURM_JOB_ID}}"
resume=()
if [[ -f "$run_dir/nafnet_last.pt" || -f "$run_dir/nafnet_final.pt" ]]; then
    resume=(--resume)
fi
fast=()
if [[ "${NAFNET_FAST:-0}" == 1 ]]; then
    fast=(--fast)
fi
echo "Job ${SLURM_JOB_ID} on $(hostname) at $(date); outputs: $run_dir"
nvidia-smi
python -u train_pilot.py \
    --manifest data/nafnet_20000/open-images-v6/train/training_photos.json \
    --expected-photos 20000 --epochs "$epochs" --batch-size 8 --workers 8 \
    --output "$run_dir" "${resume[@]}" "${fast[@]}"

python -u evaluate_color.py \
    --checkpoint "$run_dir/nafnet_final.pt" \
    --test-dir data/color_test_burstormer_protocol \
    --expected-cases-per-gain 100 \
    --output "$run_dir/evaluation"
echo "Finished at $(date). Results: $run_dir"
