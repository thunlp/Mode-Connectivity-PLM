cd ..

tasks="mnli sst2"
pre_steps="15k"

for task in $tasks
do

for pre_step in $pre_steps
do

    echo "Dealing Task $task"
    TOKENIZERS_PARALLELISM=false \
    CUDA_VISIBLE_DEVICES=1 \
    python3 itp_task_boundary_finetune.py configs/roberta-base/itp_boundary_TestUse_${task}_pre${pre_step}.json

done

done
