#!/bin/bash
#Batch Job Paremeters
#SBATCH --mail-user=XXX@mail
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --job-name=SLEET_run
#SBATCH --account=XXXXXXXXXX
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH -o slurm_out/slurm-%j.out

export HYDRA_FULL_ERROR=1

### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

source ~/anaconda3/etc/profile.d/conda.sh
conda activate SLEET
srun python3 scripts/test.py