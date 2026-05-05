#!/bin/bash

MODEL_TYPE="svs"

# ============================================
# 多GPU配置 - 根据您的硬件修改
# ============================================
NUM_GPUS=3  # 修改为您的GPU数量（2, 4, 8等）
export CUDA_VISIBLE_DEVICES="3,5,7"  # 修改为您要使用的GPU ID

# ============================================
# 训练参数
# ============================================
MAX_EPOCH=50
MODEL_NAME_OR_MODEL_DIR="/code/FunASR/models/svs"

TRAIN_DATA="/asr-corpus/train/mess_v1_0/train_clean.jsonl"
VAL_DATA="/asr-corpus/train/mess_v1_0/val_clean.jsonl"
OUTPUT_DIR="/traindata/outputs/mixed/svs-mess-v1_2"
MASTER_PORT=57329

# ============================================
# Batch Size 和学习率配置
# ============================================
# 单卡配置
BATCH_SIZE_PER_GPU=60000  # 每张卡的batch size
NUM_WORKERS=8

# 多卡配置 - 自动计算
EFFECTIVE_BATCH_SIZE=$((BATCH_SIZE_PER_GPU * NUM_GPUS))
BASE_LR=0.00005

# 学习率缩放策略（使用Python计算，避免bc依赖）
# 选项1: 平方根缩放（推荐，更保守）
SCALED_LR=$(python3 -c "import math; print($BASE_LR * math.sqrt($NUM_GPUS))")

# 选项2: 线性缩放（如需使用，注释掉上面一行，取消注释下面一行）
# SCALED_LR=$(python3 -c "print($BASE_LR * $NUM_GPUS)")

# 选项3: 不缩放（如需使用，注释掉上面的SCALED_LR行，取消注释下面一行）
# SCALED_LR=$BASE_LR

# ============================================
# 分布式训练配置
# ============================================
GPU_NUM=${NUM_GPUS}
LOG_FILE="${OUTPUT_DIR}/log.txt"

mkdir -p ${OUTPUT_DIR}

DISTRIBUTED_ARGS="
    --nnodes 1 \
    --nproc_per_node ${NUM_GPUS} \
    --master_addr 127.0.0.1 \
    --master_port ${MASTER_PORT}
"

TRAIN_TOOL="/code/FunASR/funasr/bin/train_ds.py"

# ============================================
# 启动TensorBoard
# ============================================
pkill -f "tensorboard"
rm -rf "${OUTPUT_DIR}/tb.log"
nohup bash -c "sleep 15 && tensorboard --host=0.0.0.0 --logdir='${OUTPUT_DIR}/tensorboard/'" > "${OUTPUT_DIR}/tb.log" 2>&1 &

# ============================================
# 开始训练
# ============================================
if [ "$MODEL_TYPE" = "svs" ]; then
    torchrun $DISTRIBUTED_ARGS \
    ${TRAIN_TOOL} \
    ++model="${MODEL_NAME_OR_MODEL_DIR}" \
    ++disable_update=True \
    ++train_data_set_list="${TRAIN_DATA}" \
    ++valid_data_set_list="${VAL_DATA}" \
    \
    ++dataset_conf.data_split_num=1 \
    ++dataset_conf.batch_sampler="BatchSampler" \
    ++dataset_conf.batch_type="token" \
    ++dataset_conf.batch_size=${BATCH_SIZE_PER_GPU} \
    ++dataset_conf.sort_size=1024 \
    ++dataset_conf.min_token_length=300 \
    ++dataset_conf.max_token_length=4000 \
    ++dataset_conf.num_workers=${NUM_WORKERS} \
    ++dataset_conf.shuffle=true \
    \
    ++train_conf.max_epoch=${MAX_EPOCH} \
    ++train_conf.log_interval=50 \
    ++train_conf.resume=true \
    ++train_conf.validate_interval=1000 \
    ++train_conf.save_checkpoint_interval=1000 \
    ++train_conf.keep_nbest_models=40 \
    ++train_conf.avg_nbest_model=30 \
    ++train_conf.avg_keep_nbest_models_type="acc" \
    ++train_conf.grad_clip=5.0 \
    ++train_conf.accum_grad=1 \
    ++train_conf.early_stopping_patience=10 \
    ++train_conf.use_deepspeed=false \
    \
    ++optim="adam" \
    ++optim_conf.lr=${SCALED_LR} \
    ++optim_conf.weight_decay=0.0 \
    \
    ++scheduler="warmuplr" \
    ++scheduler_conf.warmup_steps=2000 \
    \
    ++output_dir="${OUTPUT_DIR}" &> ${LOG_FILE} &
    
    echo "✅ Multi-GPU Training started!"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🎯 Configuration:"
    echo "  - Number of GPUs: ${NUM_GPUS}"
    echo "  - GPU IDs: ${CUDA_VISIBLE_DEVICES}"
    echo "  - Batch size per GPU: ${BATCH_SIZE_PER_GPU} tokens"
    echo "  - Effective batch size: ${EFFECTIVE_BATCH_SIZE} tokens"
    echo "  - Base learning rate: ${BASE_LR}"
    echo "  - Scaled learning rate: ${SCALED_LR}"
    echo "  - Model averaging: 30 best models"
    echo "  - Keep best: 40 models"
    echo ""
    echo "📊 Monitoring:"
    echo "  - Log file: tail -f ${LOG_FILE}"
    echo "  - TensorBoard: http://0.0.0.0:6006"
    echo ""
    echo "💡 Tips:"
    echo "  - Monitor GPU usage: watch -n 1 nvidia-smi"
    echo "  - Check training progress: grep 'loss' ${LOG_FILE}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
fi