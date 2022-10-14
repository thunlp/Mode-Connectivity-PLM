cd ..

TASKS="superglue-record glue-mnli anli"
DATA_DIR=./data
TUNE_METHOD=adapter
SAVE_PATH=./models
IDENTIFIER=PET_full_data_adapter
PRETRAINED_MODEL_PATH=./pretrained_models
SEEDS="20 30 40 50"
GPU=1

for TASK in $TASKS
do

for SEED in $SEEDS
do

echo "Seed: $SEED, Task: $TASK, Identifier: $IDENTIFIER"

CUDA_VISIBLE_DEVICES=${GPU} \
python tune_singletask.py \
--task_dir ${DATA_DIR}/${TASK} \
--do_train \
--do_predict \
--learning_rate_list 5e-5 \
--bsz_list 32 \
--train_iters 50000 \
--model ${PRETRAINED_MODEL_PATH}/t5.1.1.lm100k.base \
--tokenizer_path ${PRETRAINED_MODEL_PATH}/t5-v1_1-base \
--output_dir ${SAVE_PATH}/${IDENTIFIER}/${TASK}-${TUNE_METHOD}-seed_${SEED} \
--predict_batch_size 32 \
--tune_method ${TUNE_METHOD} \
--valid_interval 5000 \
--output_interval 10000 \
--log_interval 2 \
--one_prefix \
--seed ${SEED} \
--apply_adapter \
--adapter_type houlsby \
--adapter_size 12 \

done

done
