#!/bin/bash
#SBATCH --account=def-assem
#SBATCH --job-name=km-test
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=km-test-%j.out

module load StdEnv/2023 python scipy-stack
source "$SLURM_TMPDIR/env/bin/activate" 2>/dev/null || source env/bin/activate

python3 extra/tests-slurm/cluster_test.py
