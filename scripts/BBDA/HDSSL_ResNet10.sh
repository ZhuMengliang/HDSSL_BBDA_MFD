#!/usr/bin/env bash
# chmod +x ./scripts/BBDA/HDSSL_ResNet10.sh
# ps aux | grep "python run_BBDA.py"
# kill -9 [pid]
# --- 模型参数 ---
MODEL_TAR_NAME='ResNet'
MODEL_SRC_NAME='ResNet'
MODEL_TAR_BOTTLENECK=1
MODEL_SRC_BOTTLENECK=1
MODEL_SRC_BLOCKS="[2,2,2,2]"
MODEL_TAR_BLOCKS="[1,1,1,1]"
## BBDA固定参数
BALANCED=True
TEMP=1.0
DPST=1.0
HNA=1.0
# 删除所有非数字字符（保留逗号用于分割）
CLEANED_STR=$(echo "$MODEL_TAR_BLOCKS" | sed 's/[][]//g')  # 删除所有 [ 和 ]
# 验证清洗结果格式是否为 "num,num,..."
if [[ ! "$CLEANED_STR" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "错误: 输入格式非法 '$MODEL_TAR_BLOCKS' → 清洗后 '$CLEANED_STR'"
    exit 1
fi
IFS=',' read -ra BLOCKS_ARR <<< "$CLEANED_STR"  # 按逗号分割为数组
SUM=0
for num in "${BLOCKS_ARR[@]}"; do
    # 验证元素是否为整数
    if [[ ! "$num" =~ ^[0-9]+$ ]]; then
        echo "错误: 检测到非数字元素 '$num'"
        exit 1
    fi
    SUM=$((SUM + num))
done
SUM=$((SUM * 2 + 2))
DEEPTH="$SUM"
echo "DEEPTH=\"$DEEPTH\""  # 输出: DEEPTH="8"
# --- 待遍历参数定义 ---
declare -a DATASET_=('Gearbox' 'CWRU' 'PU' 'PHM')
GPU_ID=1
use_gpu=True
for DATASET in "${DATASET_[@]}"; do
    # --- 模型参数 ---
    if [ "$DATASET" = "Gearbox" ]; then
        OPT_SRC_NAME='adamw'
        OPT_SRC_LR=1e-3
        OPT_TAR_NAME='adamw'
        OPT_TAR_LR=1e-3
        WARM_UP_EPOCH=5
        TAR_EPOCH=20
        EMA=0.8
        NEIGHBORHOOD=3
    elif [ "$DATASET" = "CWRU" ]; then
        OPT_SRC_NAME='adamw'
        OPT_SRC_LR=1e-3
        OPT_TAR_NAME='adamw'
        OPT_TAR_LR=1e-3
        WARM_UP_EPOCH=5
        TAR_EPOCH=20
        EMA=0.9
        NEIGHBORHOOD=2
    elif [ "$DATASET" = "PU" ]; then
        OPT_SRC_NAME='adamw'
        OPT_SRC_LR=1e-3
        OPT_TAR_NAME='adamw'
        OPT_TAR_LR=1e-3
        WARM_UP_EPOCH=5
        TAR_EPOCH=20
        EMA=0.8
        NEIGHBORHOOD=2
    elif [ "$DATASET" = "PHM" ]; then
        OPT_SRC_NAME='sgd'
        OPT_SRC_LR=1e-2 
        OPT_TAR_NAME='sgd'
        OPT_TAR_LR=1e-2
        WARM_UP_EPOCH=10
        TAR_EPOCH=50
        EMA=0.99
        NEIGHBORHOOD=2
    fi

    APPROACH="HDSSL"
    LOG_DIR="./output_logs/BBDA_${DATASET}/${APPROACH}"
    mkdir -p $LOG_DIR
    TRAIN_TIME=$(date +"%m-%d-%H-%M") #月-日-时:分
    GROUP_1="${TRAIN_TIME}_${MODEL_TAR_NAME}${SUM}B${MODEL_TAR_BOTTLENECK}_EMA${EMA}"
    GROUP_2="_N${NEIGHBORHOOD}_W${WARM_UP_EPOCH}"
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
done        

