cd ..

TASKS="glue-mnli"
DATA_DIR=./data
TUNE_METHOD=curve_find
ADAPTER_SIZE=12
LORA_SIZE=10
PREFIX_R=24
PREFIX_NUM=120
BEND_NUM=1
SAVE_PATH=./models
IDENTIFIER=full_data_curve_find_adapter
PRETRAINED_MODEL_PATH=./pretrained_models
LOAD_PET_PATH=./models
SEED=20
GPU=1

for TASK in $TASKS
do

echo "Task: $TASK, Identifier: $IDENTIFIER"

CUDA_VISIBLE_DEVICES=${GPU} \
MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python tune_hps_singletask_PET_curve_find.py \
--task_dir ${DATA_DIR}/${TASK} \
--do_train \
--do_predict \
--learning_rate_list 1e-4 \
--bsz_list 16 \
--train_iters 10000 \
--model ${PRETRAINED_MODEL_PATH}/t5.1.1.lm100k.base \
--tokenizer_path ${PRETRAINED_MODEL_PATH}/t5-v1_1-base \
--output_dir ${SAVE_PATH}/${IDENTIFIER}/${TASK}_${TUNE_METHOD}-seed_20_std_0.02_and_0.08_5w-bend_${BEND_NUM} \
--predict_batch_size 100 \
--tune_method ${TUNE_METHOD} \
--valid_interval 100 \
--output_interval 10000 \
--log_interval 100 \
--one_prefix \
--quiet \
--apply_adapter \
--adapter_type houlsby \
--adapter_size ${ADAPTER_SIZE} \
--load_stage1_pet_path_list ${LOAD_PET_PATH}/PET_full_data_adapter/${TASK}-adapter_size_12-seed_20-r_std_0.02/lr_0.0005_bsz_16_seed_20/checkpoint@2-50000.pt ${LOAD_PET_PATH}/PET_full_data_adapter/${TASK}-adapter_size_12-seed_20-r_std_0.08/lr_0.0005_bsz_16_seed_20/checkpoint@2-50000.pt \
--bend_num ${BEND_NUM} \

done
