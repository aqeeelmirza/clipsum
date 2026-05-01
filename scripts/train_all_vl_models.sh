#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# Base paths
TRAIN_SRC="./dataset/youcook2_summarization/sum_train/tran.tok.txt"
TRAIN_TGT="./dataset/youcook2_summarization/sum_train/desc.tok.txt"
VAL_SRC="./dataset/youcook2_summarization/sum_cv/tran.tok.txt"
VAL_TGT="./dataset/youcook2_summarization/sum_cv/desc.tok.txt"
TEST_SRC="./dataset/youcook2_summarization/sum_cv/tran.tok.txt"
TEST_TGT="./dataset/youcook2_summarization/sum_cv/desc.tok.txt"

# Common hyperparameters
BATCH_SIZE=16
LEARNING_RATE=3e-5
NUM_EPOCHS=200
MAX_INPUT_LEN=512
MAX_OUTPUT_LEN=128
MAX_IMG_LEN=50
N_BEAMS=5
NO_REPEAT_NGRAM=3
GRAD_ACCUMULATE=4

# Fusion settings
FUSION_LAYER=5
CROSS_ATTN_TYPE=0
DIM_COMMON=256
N_ATTN_HEADS=1

# ============================================================================
# TRAIN FUNCTION
# ============================================================================

train_model() {
    local MODEL_NAME=$1
    local FEATURE_PATH=$2
    local VISUAL_DIM=$3
    
    echo ""
    echo "========================================================================"
    echo "TRAINING: $MODEL_NAME"
    echo "========================================================================"
    echo "Features: $FEATURE_PATH"
    echo "Visual dim: $VISUAL_DIM"
    echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    # Train
    python src/run.py \
        -model multi_modal_bart \
        -train_src_path $TRAIN_SRC \
        -train_tgt_path $TRAIN_TGT \
        -val_src_path $VAL_SRC \
        -val_tgt_path $VAL_TGT \
        -test_src_path $TEST_SRC \
        -test_tgt_path $TEST_TGT \
        -image_feature_path $FEATURE_PATH/ \
        -visual_hidden_size $VISUAL_DIM \
        -fusion_layer $FUSION_LAYER \
        -cross_attn_type $CROSS_ATTN_TYPE \
        -dim_common $DIM_COMMON \
        -n_attn_heads $N_ATTN_HEADS \
        -batch_size $BATCH_SIZE \
        -learning_rate $LEARNING_RATE \
        -num_epochs $NUM_EPOCHS \
        -max_input_len $MAX_INPUT_LEN \
        -max_output_len $MAX_OUTPUT_LEN \
        -max_img_len $MAX_IMG_LEN \
        -n_beams $N_BEAMS \
        -no_repeat_ngram_size $NO_REPEAT_NGRAM \
        -do_train True \
        -do_test False \
        -checkpoint None \
        -log_name $MODEL_NAME \
        -gpus 1 \
        -grad_accumulate $GRAD_ACCUMULATE \
        -val_save_file ./evaluation/${MODEL_NAME}_valid \
        -test_save_file ./evaluation/results/${MODEL_NAME}_test.txt
    
    if [ $? -eq 0 ]; then
        echo "✅ $MODEL_NAME training completed!"
    else
        echo "❌ $MODEL_NAME training FAILED!"
    fi
}

# ============================================================================
# TRAIN ALL MODELS
# ============================================================================

START_TIME=$(date +%s)

echo "========================================================================"
echo "TRAINING ALL VL MODELS"
echo "========================================================================"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# # 1. ResNet-152 (CNN Baseline)
# train_model "resnet152" "./extracted_features/resnet152" 2048

# 2. CLIP ViT-B/32 (VL Baseline)
train_model "clip_vit_b32" "./extracted_features/clip_vit_b32" 512

# 3. CLIP ViT-B/16
train_model "clip_vit_b16" "./extracted_features/clip_vit_b16" 512

# 4. BLIP Base
train_model "blip_base" "./extracted_features/blip_base" 768

# # 5. SigLIP Base
# train_model "siglip_base" "./extracted_features/siglip_base" 768

# ============================================================================
# SUMMARY
# ============================================================================

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))

echo ""
echo "========================================================================"
echo "ALL TRAINING COMPLETE!"
echo "========================================================================"
echo "Total time: ${HOURS}h ${MINUTES}m"
echo "Completed: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "Checkpoints saved in:"
echo "  lightning_logs/resnet152/"
echo "  lightning_logs/clip_vit_b32/"
echo "  lightning_logs/clip_vit_b16/"
echo "  lightning_logs/blip_base/"
echo "  lightning_logs/siglip_base/"
echo ""
echo "Next: Evaluate all models on test set"
echo "========================================================================"