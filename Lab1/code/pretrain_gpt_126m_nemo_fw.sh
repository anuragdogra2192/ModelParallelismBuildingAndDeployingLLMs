
# Distributed training args
NNODES=2
GPUS_PER_NODE=2
TP_SIZE=1
PP_SIZE=1

# Distributed training 
MICRO_BATCH_SIZE=4    
GLOBAL_BATCH_SIZE=64

# Data Paths
VOCAB_FILE=/dli/data/GPT-2_assets/gpt2-vocab.json
MERGE_FILE=/dli/data/GPT-2_assets/gpt2-merges.txt
DATA_PATH=[1.0,/dli/data/GPT-2_assets/my-gpt2_text_document]

OUTPUT_PATH=/dli/nemo
LOGS_PATH=/dli/nemo/logs
NAME="GPT_126m_NeMo_FW"      


OPTIMIZER_ARGS=" \
            training.model.optim.name=fused_adam \
            training.model.optim.betas=[0.9,0.95] \
            training.model.optim.lr=6e-5 \
            training.model.optim.sched.min_lr=6e-6 \
            training.model.optim.sched.name=CosineAnnealing \
            +training.model.optim.sched.max_steps=800 \
            training.model.optim.sched.warmup_steps=80 \
            training.model.optim.weight_decay=1e-1 \
        "

# NeMo Framework Launcher arguments
LAUNCHER_ARGS=" \
            cluster_type=bcm \
            stages=[training] \
            training=gpt3/126m \
            training_config=gpt3/126m \
            launcher_scripts_path=/dli/code/NeMo-Megatron-Launcher/launcher_scripts \
            "

# Search path for NeMo example configs
HYDRA_ARGS=" \
            training.hydra.searchpath=[file:///dli/code/NeMo/examples/nlp/language_modeling/conf]
        "

# Trainer arguments
TRAINER_ARGS=" \
            training.trainer.devices=$GPUS_PER_NODE \
            training.trainer.num_nodes=$NNODES \
            training.trainer.max_steps=1000 \
            +training.trainer.enable_model_summary=true \
            training.trainer.log_every_n_steps=10 \
            training.trainer.val_check_interval=20 \
            training.trainer.limit_val_batches=10 \
            +training.trainer.use_profiler=true \
        "

GPT_ARGS=" \
            training.model.micro_batch_size=$MICRO_BATCH_SIZE \
            training.model.global_batch_size=$GLOBAL_BATCH_SIZE \
            training.model.tokenizer.vocab_file=$VOCAB_FILE \
            training.model.tokenizer.merge_file=$MERGE_FILE \
            $OPTIMIZER_ARGS \
        "

OUTPUT_ARGS=" \
            training.run.results_dir=$OUTPUT_PATH/$NAME \
            training.exp_manager.explicit_log_dir=$OUTPUT_PATH/$NAME \
            training.exp_manager.resume_if_exists=false \
            training.exp_manager.name=$NAME \
        "

PARALLEL_ARGS=" \
            training.model.tensor_model_parallel_size=$TP_SIZE \
            training.model.pipeline_model_parallel_size=$PP_SIZE \
        "

CMD=" \
            python /dli/code/NeMo-Megatron-Launcher/launcher_scripts/main.py \
            $LAUNCHER_ARGS \
            $HYDRA_ARGS \
            $TRAINER_ARGS \
            $GPT_ARGS \
            $OUTPUT_ARGS \
            $PARALLEL_ARGS \
            training.model.data.data_prefix=$DATA_PATH \
            training.model.data.data_impl=mmap \
            training.model.data.splits_string=\"949,50,1\" \
        "

$CMD
