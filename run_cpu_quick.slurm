#!/bin/bash
#SBATCH --job-name=tta_quick
#SBATCH --output=quick_%j.out
#SBATCH --error=quick_%j.err
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00

source /work/axl220095/envs/beta_env/bin/activate
cd /work/axl220095/BETA

python main.py --algorithm baseline --device cpu > out_baseline.txt 2>&1
python main.py --algorithm lame --device cpu > out_lame.txt 2>&1
python main.py --algorithm beta --device cpu > out_beta.txt 2>&1
