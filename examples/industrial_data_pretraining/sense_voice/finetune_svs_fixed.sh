#!/bin/bash

MODEL_TYPE="svs"
export CUDA_VISIBLE_DEVICES="0"
MAX_EPOCH=50
MODEL_NAME_OR_MODEL_DIR="/code/FunASR/models/svs"

TRAIN_DATA="/asr-corpus/train/teochew/convert_5th/train.jsonl"
VAL_DATA="/asr-corpus/train/teochew/convert_5th/val.jsonl"
OUTPUT_DIR="/traindata/outputs/mixed/svs-teo-aggressive-v4"
MASTER_PORT=57329

BATCH_SIZE=60000  # Back to original
NUM_WORKERS=8

GPU_NUM=1
LOG_FILE="${OUTPUT_DIR}/log.txt"

mkdir -p ${OUTPUT_DIR}

DISTRIBUTED_ARGS="
    --nnodes 1 \
    --nproc_per_node 1 \
    --master_addr 127.0.0.1 \
    --master_port ${MASTER_PORT}
"

TRAIN_TOOL="/code/FunASR/funasr/bin/train_ds.py"

pkill -f "tensorboard"
rm -rf "${OUTPUT_DIR}/tb.log"
nohup bash -c "sleep 15 && tensorboard --host=0.0.0.0 --logdir='${OUTPUT_DIR}/tensorboard/'" > "${OUTPUT_DIR}/tb.log" 2>&1 &

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
    ++dataset_conf.batch_size=${BATCH_SIZE} \
    ++dataset_conf.sort_size=4096 \
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
    ++optim_conf.lr=0.00005 \
    ++optim_conf.weight_decay=0.0 \
    \
    ++scheduler="warmuplr" \
    ++scheduler_conf.warmup_steps=2000 \
    \
    ++output_dir="${OUTPUT_DIR}" &> ${LOG_FILE} &
    
    echo "✅ Training started with conservative optimization!"
    echo "📊 Monitor: tail -f ${LOG_FILE}"
    echo "📈 TensorBoard: http://0.0.0.0:6006"
    echo ""
    echo "🎯 Changes from baseline:"
    echo "  - Model averaging: 30 best models (from 20)"
    echo "  - Keep best: 40 models (from 30)"
    echo "  - All other params: same as your stable config"
fi
