cd ..

TASK=superglue-record
DATA_DIR=./data
TUNE_METHOD=finetune
SAVE_PATH=./models_itp_result
IDENTIFIER=itp_diff_step
PRETRAINED_MODEL_PATH=./pretrained_models
STEP1=10000
STEP2=50000
SEED1=20
SEED2=30
GPU=2

echo "SEED1: $SEED1, SEED2: $SEED2, STEP1: $STEP1, STEP2: $STEP2, Task: $TASK, Identifier: $IDENTIFIER"

CUDA_VISIBLE_DEVICES=${GPU} \
python task_interpolation.py \
--task_dir ${DATA_DIR}/${TASK} \
--do_valid \
--do_predict \
--model ${PRETRAINED_MODEL_PATH}/t5.1.1.lm100k.base \
--tokenizer_path ${PRETRAINED_MODEL_PATH}/t5-v1_1-base \
--output_dir ${SAVE_PATH}/${IDENTIFIER}/${TASK}_${TUNE_METHOD}_diff-step/SEED${SEED1}_STEP${STEP1}_and_SEED${SEED2}_STEP${STEP} \
--tune_method model \
--one_prefix \
--load_PET_path_1 ./models/PET_full_data_${TUNE_METHOD}/${TASK}-${TUNE_METHOD}-seed_${SEED1}/lr_0.0001_bsz_32_seed_${SEED1}/checkpoint@${STEP1}.pt \
--load_PET_path_2 ./models/PET_full_data_${TUNE_METHOD}/${TASK}-${TUNE_METHOD}-seed_${SEED2}/lr_0.0001_bsz_32_seed_${SEED2}/checkpoint@${STEP2}.pt \
--itpl_points 26 \
