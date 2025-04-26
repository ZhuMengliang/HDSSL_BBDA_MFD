#!/usr/bin/env bash
# chmod +x ./scripts/UDA/DAN.sh
# ./scripts/UDA/DAN.sh
# ps aux | grep "python run_UDA.py"
# kill -9 [pid]
#---运行信息----
APPROACH="DAN"
# --- 模型参数 ---
MODEL_SRC_NAME='ResNet'
MODEL_SRC_BOTTLENECK=1
MODEL_SRC_BLOCKS="[2,2,2,2]"
DROP_LAST=True
SRC_EPOCH=50
GPU_ID=1
use_gpu=True
# --- 待遍历参数定义 ---
declare -a DATASET_=("Gearbox" "PU" "CWRU" "PHM")
for DATASET in "${DATASET_[@]}"; do
    if [ "$DATASET" = "Gearbox" ]; then
        OPT_SRC_NAME='adamw'
        OPT_SRC_LR=1e-3
        DAN=0.1
        NON_LINEAR=True
    elif [ "$DATASET" = "CWRU" ]; then
        OPT_SRC_NAME='adamw'
        OPT_SRC_LR=1e-3
        DAN=0.1
        NON_LINEAR=True
    elif [ "$DATASET" = "PU" ]; then
        OPT_SRC_NAME='adamw'
        OPT_SRC_LR=1e-3
        DAN=0.1
        NON_LINEAR=False
    elif [ "$DATASET" = "PHM" ]; then
        OPT_SRC_NAME='sgd'
        OPT_SRC_LR=1e-2 
        DAN=1.0
        NON_LINEAR=True
    fi
    # --- 动态时间戳---
    LOG_DIR="./output_logs/UDA_${DATASET}/${APPROACH}"
    mkdir -p "$LOG_DIR"
    TRAIN_TIME=$(date +"%m-%d-%H-%M") #月-日-时:分
    GROUP_1="${TRAIN_TIME}_${MODEL_SRC_NAME}B${MODEL_SRC_BOTTLENECK}"
    GROUP_2="_DAN${DAN}_${NON_LINEAR}_${OPT_SRC_NAME}_lr${OPT_SRC_LR}"
    GROUP_NAME="${GROUP_1}${GROUP_2}"
    LOG_FILE="${LOG_DIR}/${GROUP_NAME}.log"
    python run_UDA.py -m \
        Dataset="$DATASET" \
        UDA="$APPROACH" \
        UDA.dan="$DAN" \
        UDA.non_linear="$NON_LINEAR" \
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

