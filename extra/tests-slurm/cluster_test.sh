#!/bin/bash
#SBATCH --job-name=km-test
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --output=km-test-%j.out
# #SBATCH --partition=<your-partition>

python3 extra/tests-slurm/cluster_test.py
