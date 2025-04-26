#!/usr/bin/env bash
# chmod +x ./scripts/UDA/DANN.sh
# ./scripts/UDA/DANN.sh
# ps aux | grep "python run_UDA.py"
# kill -9 [pid]
#---运行信息----
APPROACH="DANN"
# --- 模型参数 ---
MODEL_SRC_NAME='ResNet'
MODEL_SRC_BOTTLENECK=1
MODEL_SRC_BLOCKS="[2,2,2,2]"
DROP_LAST=True
SRC_EPOCH=50
GPU_ID=2
use_gpu=True
declare -a DATASET_=("Gearbox" "PU" "CWRU" "PHM")
for DATASET in "${DATASET_[@]}"; do
    if [ "$DATASET" = "Gearbox" ]; then
        OPT_SRC_NAME='adamw'
        OPT_SRC_LR=1e-3
        DANN=0.5
        ADV_WARMUP=True
    elif [ "$DATASET" = "CWRU" ]; then
        OPT_SRC_NAME='adamw'
        OPT_SRC_LR=1e-3
        DANN=1.0
        ADV_WARMUP=False
    elif [ "$DATASET" = "PU" ]; then
        OPT_SRC_NAME='adamw'
        OPT_SRC_LR=1e-3
        DANN=0.1
        ADV_WARMUP=False
    elif [ "$DATASET" = "PHM" ]; then
        OPT_SRC_NAME='sgd'
        OPT_SRC_LR=1e-2 
        DANN=0.5
        ADV_WARMUP=False
    fi
    # --- 动态时间戳---
    LOG_DIR="./output_logs/UDA_${DATASET}/${APPROACH}"
    mkdir -p "$LOG_DIR"
    TRAIN_TIME=$(date +"%m-%d-%H-%M") #月-日-时:分
    GROUP_1="${TRAIN_TIME}_${MODEL_SRC_NAME}B${MODEL_SRC_BOTTLENECK}_DANN${DANN}"
    GROUP_2="_W${ADV_WARMUP}_${OPT_SRC_NAME}_lr${OPT_SRC_LR}"
    GROUP_NAME="${GROUP_1}${GROUP_2}"
    LOG_FILE="${LOG_DIR}/${GROUP_NAME}.log"
    python run_UDA.py -m \
        Dataset="$DATASET" \
        UDA="$APPROACH" \
        UDA.dann="$DANN" \
        UDA.adv_warmup="$ADV_WARMUP" \
        Model_src.bottleneck_type="$MODEL_SRC_BOTTLENECK" \
        Model_src.name="$MODEL_SRC_NAME" \
        Model_src.blocks="$MODEL_SRC_BLOCKS" \
        Opt_src.name="$OPT_SRC_NAME" \
        Opt_src.lr="$OPT_SRC_LR" \
        Training.drop_last="$DROP_LAST" \
        Training.src_epoch="$SRC_EPOCH" \
        Wandb.setup.group="$GROUP_NAME" \
        gpu_id="$GPU_ID" \
        use_gpu="$use_gpu"\
        > "$LOG_FILE" 2>&1 &
    echo "任务PID: $!"
    echo "Wandb 组名: $GROUP_NAME"
    echo "日志文件: $LOG_FILE"
done        

