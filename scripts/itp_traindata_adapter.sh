cd ..

TASK=glue-mnli
DATA_DIR=./data
TUNE_METHOD=adapter
SAVE_PATH=./models_itp_result
IDENTIFIER=itp_on_traindata
PRETRAINED_MODEL_PATH=./pretrained_models
STEP=10
SEED1=20
SEED2=30
GPU=3

echo "SEED1: $SEED1, SEED2: $SEED2, Task: $TASK, Identifier: $IDENTIFIER"

CUDA_VISIBLE_DEVICES=${GPU} \
python task_interpolation.py \
--task_dir ${DATA_DIR}/${TASK} \
--model ${PRETRAINED_MODEL_PATH}/t5.1.1.lm100k.base \
--tokenizer_path ${PRETRAINED_MODEL_PATH}/t5-v1_1-base \
--output_dir ${SAVE_PATH}/${IDENTIFIER}/${TASK}_${TUNE_METHOD}_diff-seed/SEED${SEED1}_and_SEED${SEED2}_STEP${STEP} \
--tune_method ${TUNE_METHOD} \
--one_prefix \
--apply_adapter \
--adapter_type houlsby \
--adapter_size 12 \
--load_PET_path_1 ./models/PET_full_data_${TUNE_METHOD}/${TASK}-${TUNE_METHOD}-seed_${SEED1}/lr_5e-05_bsz_32_seed_${SEED1}/checkpoint@${STEP}.pt \
--load_PET_path_2 ./models/PET_full_data_${TUNE_METHOD}/${TASK}-${TUNE_METHOD}-seed_${SEED2}/lr_5e-05_bsz_32_seed_${SEED2}/checkpoint@${STEP}.pt \
--itpl_points 6 \
--itp_on_train \
