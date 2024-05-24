#!/bin/bash
#SBATCH --job-name=dli_2nodes
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2       
#SBATCH --cpus-per-task=32 ### Number of threads per task (OMP threads)
#SBATCH -o /dli/nemo/logs/%j.out
#SBATCH -e /dli/nemo/logs/%j.err

set -x -e

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Distributed training args
NNODES=2
GPUS_PER_NODE=2
TP_SIZE=1
PP_SIZE=1

# Distributed training 
MICRO_BATCH_SIZE=16     
GLOBAL_BATCH_SIZE=64    

# Model architecture 
NLAYERS=12
NHIDDEN=768
NHEADS=32
SEQ_LEN=1024

# Data Paths
VOCAB_FILE=/dli/data/GPT-2_assets/gpt2-vocab.json
MERGE_FILE=/dli/data/GPT-2_assets/gpt2-merges.txt
DATA_PATH=[1.0,/dli/data/GPT-2_assets/my-gpt2_text_document]

OUTPUT_PATH=/dli/nemo
LOGS_PATH=/dli/nemo/logs
NAME="2Nodes4GPUS_increase_MBS_fp16_activation_checkpoiting"        


OPTIMIZER_ARGS=" \
            model.optim.name=fused_adam \
            model.optim.betas=[0.9,0.95] \
            model.optim.lr=6e-5 \
            model.optim.sched.min_lr=6e-6 \
            model.optim.sched.name=CosineAnnealing \
            +model.optim.sched.max_steps=800 \
            model.optim.sched.warmup_steps=80 \
            model.optim.weight_decay=1e-1 \
        "

TRAINER_ARGS=" \
            trainer.gradient_clip_val=1.0 \
            trainer.precision=16 \
            trainer.devices=$GPUS_PER_NODE \
            trainer.num_nodes=$NNODES \
            trainer.max_steps=100 \
            trainer.enable_model_summary=true \
            trainer.log_every_n_steps=10 \
            trainer.val_check_interval=20 \
            trainer.limit_val_batches=10 \
            +trainer.use_profiler=true \
        "

GPT_ARGS=" \
            model.num_layers=$NLAYERS \
            model.hidden_size=$NHIDDEN \
            model.num_attention_heads=$NHEADS \
            model.encoder_seq_length=$SEQ_LEN \
            model.data.seq_length=$SEQ_LEN \
            model.max_position_embeddings=$SEQ_LEN \
            model.micro_batch_size=$MICRO_BATCH_SIZE \
            model.global_batch_size=$GLOBAL_BATCH_SIZE \
            model.tokenizer.vocab_file=$VOCAB_FILE \
            model.tokenizer.merge_file=$MERGE_FILE \
            model.init_method_std=0.006 \
            model.activations_checkpoint_method=uniform \
            $OPTIMIZER_ARGS \
        "

OUTPUT_ARGS=" \
            exp_manager.explicit_log_dir=$OUTPUT_PATH/$NAME \
            exp_manager.resume_if_exists=false \
            exp_manager.name=$NAME \
        "

PARALLEL_ARGS=" \
            model.tensor_model_parallel_size=$TP_SIZE \
            model.pipeline_model_parallel_size=$PP_SIZE \
        "

export CMD=" \
            python /dli/code/NeMo/examples/nlp/language_modeling/megatron_gpt_pretraining.py \
            --config-path=/dli/code/NeMo/examples/nlp/language_modeling/conf/ \
            --config-name=megatron_gpt_config.yaml \
            $TRAINER_ARGS \
            $PARALLEL_ARGS \
            $GPT_ARGS \
            $OUTPUT_ARGS \
            model.data.data_prefix=$DATA_PATH \
            model.data.data_impl=mmap \
            model.data.splits_string=\"949,50,1\" \
        "

clear; srun --jobid $SLURM_JOBID bash -c 'NCCL_DEBUG=INFO $CMD' 2>&1 | tee -a $LOGS_PATH/$NAME.txt