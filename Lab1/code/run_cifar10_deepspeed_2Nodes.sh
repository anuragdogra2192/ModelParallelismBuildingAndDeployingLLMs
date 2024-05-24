#!/bin/bash
#SBATCH --job-name=dli_ds
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1       
#SBATCH --cpus-per-task=32 ### Number of threads per task (OMP threads)
#SBATCH -o /dli/nemo/logs/%j.out
#SBATCH -e /dli/nemo/logs/%j.err

# Number of nodes
NUM_NODES=2
# Number of GPUs per node
NUM_GPUS=2


deepspeed --num_nodes=${NUM_NODES} --hostfile /dli/code/moe/hostfile --num_gpus=${NUM_GPUS} /dli/code/moe/cifar10_deepspeed.py \
    --deepspeed \
    --deepspeed_config /dli/code/moe/ds_config.json \
    --profile-execution=True \
    --profile-name='zero0_sbatch'
