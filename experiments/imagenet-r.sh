#!/bin/bash

# experiment settings
DATASET=ImageNet_R
N_CLASS=200

# hard coded inputs
GPUID='0'  # Đổi thành 0 cho Kaggle nhé
CONFIG=configs/imagenet-r_prompt.yaml
REPEAT=1
OVERWRITE=0

# hyperparameter arrays
LR=0.02
SCHEDULE=25
EMA_COEFF=0.7
SEED_LIST=(1 2 3)

DELAY_BETWEEN_EXPERIMENTS=10 
LOG_DIR="logs"
mkdir -p $LOG_DIR

for seed in "${SEED_LIST[@]}"
    do
        OUTDIR="./checkpoints/${DATASET}/seed${seed}"
        mkdir -p $OUTDIR
        LOG_FILE="${LOG_DIR}/${DATASET}_seed${seed}.log"

        echo "Starting experiment with seed=$seed"
        
        nohup python -u run.py \
            --config $CONFIG \
            --gpuid $GPUID \
            --repeat $REPEAT \
            --overwrite $OVERWRITE \
            --learner_type prompt \
            --learner_name APT_Learner \
            --prompt_param 0.01 \
            --lr $LR \
            --seed $seed \
            --ema_coeff $EMA_COEFF \
            --schedule $SCHEDULE \
            --dataroot /kaggle/working/data \
            --log_dir ${OUTDIR} > "$LOG_FILE" 2>&1 &

        PID=$!
        wait $PID
        
        if [ $? -eq 0 ]; then
            echo "Experiment completed successfully"
        else
            echo "Experiment failed"
        fi

        rm -rf ${OUTDIR}/models
        echo "----------------------------------------"
        
        if [ $current -lt $total_experiments ]; then
            echo "Waiting for $DELAY_BETWEEN_EXPERIMENTS seconds before next experiment..."
            sleep $DELAY_BETWEEN_EXPERIMENTS
        fi
    done

echo "All experiments completed!"
exit 0
