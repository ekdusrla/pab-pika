#!/bin/bash
#SBATCH --job-name=pb-train
#SBATCH --partition=batch_ugrad
#SBATCH --gres=gpu:normal:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Step 3: Ridge + MLP x features F1..F4, GroupKFold-5 outer, inner problem split for
# layer / HP / early stop, MLP seeds 0 1 2. Writes data_v2/probe_report/ and
# probe_artifacts/{probe.pt,metadata.json} (the committed probe: Ridge F3, layer 12, alpha 1e4)
source experiments/probe_budget/slurm/env.sh
conda activate "$CONDA_PROBE_ENV"
export OMP_NUM_THREADS=4
cd experiments/probe_budget
python -u train_probe_mlp.py --seeds 0 1 2
