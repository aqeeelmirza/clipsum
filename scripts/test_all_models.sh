#!/bin/bash
# test_all_models.sh

export CUDA_VISIBLE_DEVICES=0

TEST_SRC="./dataset/youcook2_summarization/sum_cv/tran.tok.txt"
TEST_TGT="./dataset/youcook2_summarization/sum_cv/desc.tok.txt"

test_model() {
    local MODEL_NAME=$1
    local FEATURE_PATH=$2
    local VISUAL_DIM=$3
    
    # Find best checkpoint
    CKPT=$(find lightning_logs/${MODEL_NAME}/ -name "*.ckpt" | grep -v last | tail -1)
    
    echo "Testing $MODEL_NAME with $CKPT"
    
    python src/run.py \
        -model multi_modal_bart \
        -test_src_path $TEST_SRC \
        -test_tgt_path $TEST_TGT \
        -image_feature_path ${FEATURE_PATH}/ \
        -visual_hidden_size $VISUAL_DIM \
        -checkpoint $CKPT \
        -do_test True \
        -do_train False \
        -gpus 1 \
        -test_save_file ./evaluation/results/${MODEL_NAME}_test.txt
}

test_model "resnet152" "./extracted_features/resnet152" 2048
test_model "clip_vit_b32" "./extracted_features/clip_vit_b32" 512
test_model "clip_vit_b16" "./extracted_features/clip_vit_b16" 512
test_model "blip_base" "./extracted_features/blip_base" 768