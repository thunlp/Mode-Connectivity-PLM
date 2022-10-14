cd ..

TASK=glue-mnli
DATA_DIR=./data
TUNE_METHOD=finetune
SAVE_PATH=./models_itp_result
IDENTIFIER=itp_data_split
PRETRAINED_MODEL_PATH=./pretrained_models
STEPS="10000 30000 50000"
SEEDS="20 30"
GPU=1

for SEED in $SEEDS
do

for STEP in $STEPS
do

echo "SEED: $SEED, Task: $TASK, STEP: $STEP, Identifier: $IDENTIFIER"

CUDA_VISIBLE_DEVICES=${GPU} \
python task_interpolation.py \
--task_dir ${DATA_DIR}/${TASK} \
--do_valid \
--do_predict \
--model ${PRETRAINED_MODEL_PATH}/t5.1.1.lm100k.base \
--tokenizer_path ${PRETRAINED_MODEL_PATH}/t5-v1_1-base \
--output_dir ${SAVE_PATH}/${IDENTIFIER}/${TASK}_${TUNE_METHOD}_datasplit/seed${SEED}_STEP${STEP} \
--tune_method model \
--one_prefix \
--load_PET_path_1 ./models/PET_datasplit_${TUNE_METHOD}/${TASK}-datasplit/seed_${SEED}-split1/lr_5e-05_bsz_32_seed_${SEED}/checkpoint@${STEP}.pt \
--load_PET_path_2 ./models/PET_datasplit_${TUNE_METHOD}/${TASK}-datasplit/seed_${SEED}-split2/lr_5e-05_bsz_32_seed_${SEED}/checkpoint@${STEP}.pt \
--itpl_points 26 \

done

done