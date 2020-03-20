#!/bin/bash

#SBATCH --array=0-1
#SBATCH --ntasks=1
#SBATCH --output=../log/slurm/slurm_dfts_dh_%A_%a.log
#SBATCH --job-name=dfts
#SBATCH -A collabrobogroup
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=udc-ba25-[23,27,28],udc-ba26-[23,24,25],udc-ba27-[23,24]

#SBATCH -t 72:00:00

module load anaconda
source activate pytorch13

menh=(1 2)
mmnh=(1 2)

python3 ../deepfake_training.py \
-lr 0.0003 \
-ng 1 \
-cdn -1 \
-sml 300 \
-bs 2 \
-ep 6 \
-enl 2 \
-fes 128 \
-lhs 128 \
-menh ${menh[${SLURM_ARRAY_TASK_ID}]} \
-mmnh ${mmnh[${SLURM_ARRAY_TASK_ID}]} \
-lld 0.2 \
-uld 0.2 \
-cout 64 \
-mmattn_type 'concat' \
-logbd 'log' \
-logf 'exe_dfts_h'${menh[${SLURM_ARRAY_TASK_ID}]}''${mmnh[${SLURM_ARRAY_TASK_ID}]}'_dp2_'${SLURM_ARRAY_TASK_ID}'_'${SLURM_ARRAY_JOB_ID}'.log' \
-tbl \
-tb_wn 'tb_runs/tb_dfts/h'${menh[${SLURM_ARRAY_TASK_ID}]}''${mmnh[${SLURM_ARRAY_TASK_ID}]}'_dp2_'${SLURM_ARRAY_TASK_ID}'_'${SLURM_ARRAY_JOB_ID}'' \
-mcp 'dfts_'${SLURM_ARRAY_TASK_ID}'_'${SLURM_ARRAY_JOB_ID}'' \
-dfp '../dfdc_datasets/dfdc_train_data/' \
-is_guiding \
-cm 2 \
-cl 4

