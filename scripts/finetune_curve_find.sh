cd ..
GPU=0


TASKS="glue-mnli"
STEPS="50000"
DATA_DIR=./data
TUNE_METHOD=curve_find
PET_NAME=finetune
BEND_NUM=1
SAVE_PATH=./models
IDENTIFIER=full_data_curve_find_${PET_NAME}
PRETRAINED_MODEL_PATH=./pretrained_models
LOAD_PET_PATH=./models
SOURCE_SEED=20


for TASK in $TASKS
do
for TARGET_SEED in "30"
do
for STEP in $STEPS
do

echo "Task: $TASK, Identifier: $IDENTIFIER, Source_seed: $SOURCE_SEED, Target_seed: $TARGET_SEED, itpl_step: $STEP"

CUDA_VISIBLE_DEVICES=${GPU} \
MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python tune_hps_singletask_PET_curve_find_finetune.py \
--task_dir ${DATA_DIR}/${TASK} \
--do_predict \
--learning_rate_list 1e-4 \
--bsz_list 8 \
--train_iters 5000 \
--model ${PRETRAINED_MODEL_PATH}/t5.1.1.lm100k.base \
--tokenizer_path ${PRETRAINED_MODEL_PATH}/t5-v1_1-base \
--output_dir ${SAVE_PATH}/${IDENTIFIER}/${TASK}_${PET_NAME}-diff-perterb/SEED_${SOURCE_SEED}_and_SEED_${TARGET_SEED}_STEP_${STEP} \
--predict_batch_size 100 \
--tune_method ${TUNE_METHOD} \
--valid_interval 100 \
--output_interval 10000 \
--log_interval 10 \
--one_prefix \
--quiet \
--load_stage1_pet_path_list ${LOAD_PET_PATH}/PET_perterb_${PET_NAME}/${TASK}-${PET_NAME}-seed_${SOURCE_SEED}/seed_${SOURCE_SEED}/checkpoint@${STEP}.pt ${LOAD_PET_PATH}/PET_perterb_${PET_NAME}/${TASK}-${PET_NAME}-seed_${TARGET_SEED}/seed_${TARGET_SEED}/checkpoint@${STEP}.pt \
--bend_num ${BEND_NUM} \
--choose_valid \
--choose_valid_lines 1000 \

done
done
done

