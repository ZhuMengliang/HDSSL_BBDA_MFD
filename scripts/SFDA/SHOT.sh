#!/usr/bin/env bash
# chmod +x ./scripts/SFDA/SHOT.sh
# ./scripts/SFDA/SHOT.sh
# ps aux | grep "python run_SFDA.py"
# kill -9 [pid]
#---运行信息----
APPROACH="SHOT"
# --- 模型参数 ---
MODEL_TAR_NAME='ResNet'
MODEL_TAR_BOTTLENECK=1
MODEL_TAR_BLOCKS="[2,2,2,2]"
DROP_LAST=False
GPU_ID=0
use_gpu=True
declare -a DATASET_=("Gearbox" "PU" "CWRU" "PHM")
for DATASET in "${DATASET_[@]}"; do
    if [ "$DATASET" = "Gearbox" ]; then
        OPT_TAR_NAME='adamw'
        OPT_TAR_LR=1e-5
        IM=1.0
        ST=0.1
        TAR_EPOCH=20
    elif [ "$DATASET" = "CWRU" ]; then
        OPT_TAR_NAME='adamw'
        OPT_TAR_LR=1e-5
        IM=1.0
        ST=1.0
        TAR_EPOCH=20
    elif [ "$DATASET" = "PU" ]; then
        OPT_TAR_NAME='adamw'
        OPT_TAR_LR=1e-5
        IM=1.0
        ST=0.1
        TAR_EPOCH=20
    elif [ "$DATASET" = "PHM" ]; then
        OPT_TAR_NAME='sgd'
        OPT_TAR_LR=1e-4
        IM=1.0
        ST=0.5
        TAR_EPOCH=50
    fi
    # --- 动态时间戳---
    LOG_DIR="./output_logs/SFDA_${DATASET}/${APPROACH}"
    mkdir -p "$LOG_DIR"
    TRAIN_TIME=$(date +"%m-%d-%H-%M") #月-日-时:分
    GROUP_1="${TRAIN_TIME}_${MODEL_TAR_NAME}B${MODEL_TAR_BOTTLENECK}_${APPROACH}"
    GROUP_2="_IM${IM}_ST${ST}_${OPT_TAR_NAME}_lr${OPT_TAR_LR}"
    GROUP_NAME="${GROUP_1}${GROUP_2}"
    LOG_FILE="${LOG_DIR}/${GROUP_NAME}.log"
    python run_SFDA.py -m \
        Dataset="$DATASET" \
        SFDA="$APPROACH" \
        SFDA.im="$IM" \
        SFDA.st="$ST" \
        Model_tar.bottleneck_type="$MODEL_TAR_BOTTLENECK" \
        Model_tar.name="$MODEL_TAR_NAME" \
        Model_tar.blocks="$MODEL_TAR_BLOCKS" \
        Opt_tar.name="$OPT_TAR_NAME" \
        Opt_tar.lr="$OPT_TAR_LR" \
        Training.drop_last="$DROP_LAST" \
        Training.tar_epoch="$TAR_EPOCH" \
        Wandb.setup.group="$GROUP_NAME" \
        gpu_id="$GPU_ID" \
        use_gpu="$use_gpu"\
        > "$LOG_FILE" 2>&1 #&

    echo "任务PID: $!"
    echo "Wandb 组名: $GROUP_NAME"
    echo "日志文件: $LOG_FILE"
done        

