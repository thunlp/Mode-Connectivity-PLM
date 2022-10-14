cd ..

tasks="mnli"
pre_steps="15k"

for task in $tasks
do

for pre_step in $pre_steps
do

    echo "Dealing Task $task"
    TOKENIZERS_PARALLELISM=false \
    CUDA_VISIBLE_DEVICES=1 \
    python3 itp_pretrain_adapter.py configs/adapter_roberta-base/${task}_itp_pretrain_pre${pre_step}.json

done

done
