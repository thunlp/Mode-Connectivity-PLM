cd ..

python utils/process_cartography.py \
--cartography_path [path to cartography dict containing raw confidence] \
--analysis_path ./models_itp_result/itp_on_traindata/[path to target valid data analysis dict] \
--save_path [path to save the result]/result.txt \