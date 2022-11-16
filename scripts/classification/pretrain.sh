#!/bin/bash
# python ${PATH-TO-FAIRSEQ_ROOT}/fairseq_cli/train.py ${args}.
# bash train_genre.sh topmagd 13 0 checkpoints/checkpoint_last_musicbert_base.pt
# bash train_xai.sh xai 28 0 checkpoints/checkpoint_last_musicbert_base.pt
export CUDA_VISIBLE_DEVICES=0

# cd checkpoints
# wget https://msramllasc.blob.core.windows.net/modelrelease/checkpoint_last_musicbert_small.pt
# wget https://msramllasc.blob.core.windows.net/modelrelease/checkpoint_last_musicbert_base.pt
# cd -

TOTAL_NUM_UPDATES=5000
WARMUP_UPDATES=300
PEAK_LRS=(5e-5)
TOKENS_PER_SAMPLE=8192
MAX_POSITIONS=8192
BATCH_SIZE=64
MAX_SENTENCES=64
subset=xai
UPDATE_FREQ=$((${BATCH_SIZE} / ${MAX_SENTENCES} / 1))
HEAD_NAME=xai_head

# LOSS_TYPE=("xai_pretrain_loss" "xai_pretrain_loss_unimodal" "xai_pretrain_loss_ntxent")
LOSS_TYPE=("xai_pretrain_loss_unimodal" "xai_pretrain_loss_ntxent")

SIZES=("base")
for size in "${SIZES[@]}"
do
    
    MUSICBERT_PATH=checkpoints/checkpoint_last_musicbert_${size}

    for lr in "${PEAK_LRS[@]}"
        do
        for loss in "${LOSS_TYPE[@]}"
            do
            CHECKPOINT_SUFFIX=${loss}_${lr}_${size}_released
            fairseq-train xai_data_bin_apex_reg_cls/0 --user-dir musicbert \
                --max-update $TOTAL_NUM_UPDATES \
                --batch-size $MAX_SENTENCES --update-freq $UPDATE_FREQ \
                --max-positions $MAX_POSITIONS \
                --max-tokens $((${TOKENS_PER_SAMPLE} * ${MAX_SENTENCES})) \
                --task xai_pretrain_task \
                --reset-optimizer --reset-dataloader --reset-meters \
                --required-batch-size-multiple 1 \
                --num-workers 0 \
                --seed 7 \
                --init-token 0 --separator-token 2 \
                --arch xai_pretrain_arch \
                --criterion $loss \
                --classification-head-name $HEAD_NAME \
                --num-cls-classes 13 \
                --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
                --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-6 --clip-norm 0.0 \
                --lr-scheduler cosine --lr $lr --warmup-updates $WARMUP_UPDATES \
                --warmup-init-lr 1e-5 --min-lr 1e-5 --lr-period-updates 800 --lr-shrink 0.9\
                --log-format json --log-interval 100 \
                --tensorboard-logdir checkpoints/board_${CHECKPOINT_SUFFIX} \
                --best-checkpoint-metric loss \
                --shorten-method "truncate" \
                --checkpoint-suffix _${CHECKPOINT_SUFFIX} \
                --no-epoch-checkpoints \
                --find-unused-parameters \
                --fp16
            done
        done
done