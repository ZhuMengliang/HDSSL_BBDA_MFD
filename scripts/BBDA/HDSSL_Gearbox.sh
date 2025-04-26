#!/usr/bin/env bash
# chmod +x ./scripts/BBDA/HDSSL_Gearbox.sh
# ps aux | grep "python run_BBDA.py"
# kill -9 [pid]

#---运行信息----
APPROACH="HDSSL"
DATASET="Gearbox"
LOG_DIR="./output_logs/BBDA_${DATASET}/${APPROACH}"
mkdir -p $LOG_DIR
###----**---
use_gpu=True
GPU_ID=0
# --- 模型参数 ---
MODEL_TAR_NAME='ResNet'
MODEL_SRC_NAME='ResNet'
MODEL_TAR_BOTTLENECK=1
MODEL_SRC_BOTTLENECK=1
MODEL_SRC_BLOCKS="[2,2,2,2]"
MODEL_TAR_BLOCKS="[2,2,2,2]"
OPT_SRC_NAME='adamw'
OPT_SRC_LR=1e-3
OPT_TAR_NAME='adamw'
OPT_TAR_LR=1e-3
## --- BBDA 固定参数 ---
WARM_UP_EPOCH=5
TAR_EPOCH=20
BALANCED=True
TEMP=1.0
DPST=1.0
HNA=1.0
EMA=0.8
NEIGHBORHOOD=3

TRAIN_TIME=$(date +"%m-%d-%H-%M") #月-日-时:分
GROUP_1="${TRAIN_TIME}_${MODEL_TAR_NAME}B${MODEL_TAR_BOTTLENECK}_EMA${EMA}"
GROUP_2="_N${NEIGHBORHOOD}_W${WARM_UP_EPOCH}"
#GROUP_2="adaptive data division"
GROUP_NAME="${GROUP_1}${GROUP_2}"
LOG_FILE="${LOG_DIR}/${GROUP_NAME}.log"

python run_BBDA.py -m \
    Dataset="$DATASET" \
    BBDA="$APPROACH" \
    BBDA.neighborhood="$NEIGHBORHOOD" \
    BBDA.dkd_ema="$EMA" \
    BBDA.warm_up_epoch="$WARM_UP_EPOCH" \
    BBDA.balanced="$BALANCED" \
    BBDA.temp="$TEMP" \
    BBDA.dpst="$DPST" \
    BBDA.hna="$HNA" \
    Model_tar.bottleneck_type="$MODEL_TAR_BOTTLENECK" \
    Model_src.bottleneck_type="$MODEL_SRC_BOTTLENECK" \
    Model_tar.name="$MODEL_TAR_NAME" \
    Model_src.name="$MODEL_SRC_NAME" \
    Model_tar.blocks="$MODEL_TAR_BLOCKS" \
    Model_src.blocks="$MODEL_SRC_BLOCKS" \
    Opt_tar.lr="$OPT_TAR_LR" \
    Opt_tar.name="$OPT_TAR_NAME" \
    Opt_src.lr="$OPT_SRC_LR" \
    Opt_src.name="$OPT_SRC_NAME" \
    Training.tar_epoch="$TAR_EPOCH" \
    Wandb.setup.group="$GROUP_NAME" \
    gpu_id="$GPU_ID" \
    use_gpu="$use_gpu"\
    > "$LOG_FILE" 2>&1 #&

echo "任务PID: $!"
echo "Wandb 组名: $GROUP_NAME"
echo "日志文件: $LOG_FILE"

