cd ..

TASK1=glue-mnli
TASK2=anli
TASKS="glue-mnli anli"
DATA_DIR=./data
TUNE_METHOD=finetune
SAVE_PATH=./models_itp_result
IDENTIFIER=itp_same_domain
PRETRAINED_MODEL_PATH=./pretrained_models
STEPS="10000 30000 50000"
SEEDS="20 30"
GPU=1

for TASK in $TASKS
do

for STEP in $STEPS
do

for SEED in $SEEDS
do

echo "SEED: $SEED, STEP: $STEP, Test On: $TASK, Identifier: $IDENTIFIER"

CUDA_VISIBLE_DEVICES=${GPU} \
python task_interpolation.py \
--task_dir ${DATA_DIR}/${TASK} \
--do_valid \
--do_predict \
--model ${PRETRAINED_MODEL_PATH}/t5.1.1.lm100k.base \
--tokenizer_path ${PRETRAINED_MODEL_PATH}/t5-v1_1-base \
--output_dir ${SAVE_PATH}/${IDENTIFIER}/TestUse_${TASK}_${TASK1}+${TASK2}_samedomain_finetune/SEED${SEED}_STEP${STEP} \
--tune_method ${TUNE_METHOD} \
--one_prefix \
--load_PET_path_1 ./models/PET_full_data_finetune/${TASK1}-finetune-seed_${SEED}/lr_5e-05_bsz_32_seed_${SEED}/checkpoint@${STEP}.pt \
--load_PET_path_2 ./models/PET_full_data_finetune/${TASK2}-finetune-seed_${SEED}/lr_5e-05_bsz_32_seed_${SEED}/checkpoint@${STEP}.pt \
--itpl_points 26 \

done

done

done