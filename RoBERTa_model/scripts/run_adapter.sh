cd ..

tasks="mnli sst2"
for task in $tasks
do
    echo "Dealing Task $task"
    TOKENIZERS_PARALLELISM=false \
    CUDA_VISIBLE_DEVICES=0 \
    python3 run_glue.py configs/adapter_roberta-base/${task}.json
done
