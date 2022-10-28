# Exploring Mode Connectivity for Pre-trained Language Models

This is the implementation of our EMNLP 2022 paper [Exploring Mode Connectivity for Pre-trained Language Models](https://arxiv.org/pdf/2210.14102.pdf).



## Installation

This code requires PyTorch and Python 3+. Other dependencies can be installed by

```bash
pip install -r requirements.txt
```



## Dataset and Checkpoint Preparation

#### Prepare Datasets

We mainly use GLUE and SuperGLUE datasets in experiments, and the data are specially prepared instead of using raw dataset. The datasets of related tasks can be downloaded [here](https://drive.google.com/drive/folders/1kxGRapy43Gf_0-Nuo1QgYmyo2cRg3NmT?usp=sharing). Please put the datasets under `./data`. We have prepared some data as example, but they are not complete, just for showing the format. Make sure `[task_name]_train.tsv`, `[task_name]_dev.tsv` and `[task_name]_test.tsv` can be found under each task folder.

#### Prepare $T5_{Base}$ Checkpoint

Besides, the pretrained models of $T5_{Base}$ and $RoBERTa_{Base}$ are also needed to conduct the experiments. For $T5_{Base}$, the pre-trained model checkpoints can be downloaded [here](https://drive.google.com/drive/folders/1OQLOcbgi61TI6idzKqlU47EJ30eQ29Lt?usp=sharing). Please put the directory `pretrained_models` downloaded directly under the root folder.

#### Prepare $RoBERTa_{Base}$ Checkpoint

For $RoBERTa_{Base}$, please download checkpoints of different pre-trained steps from [here](https://drive.google.com/drive/folders/1P4H73T2VcNej0IGUbnkmZd78OGEwnM9o?usp=sharing). They should be put under `./RoBERTa_model/checkpoint` which is now empty. Keep the folder's name as `ckpt_{STEP}` and make sure `config.json` and `pytorch_model.bin` can be found under each folder.



## Experiments on $T5_{Base}$

All the scripts for conducting experiments can be found in `./scripts`, including fine-tuning, adapter-tuning and interpolation.

All the default parameters and explanations can be found in `./utils/options.py`, and the parameters in scripts can be changed accordingly.

#### Full data tuning

The parameters of example script can be found in `./scripts/finetune_full_data.sh`. This is the basic version tuning. The seeds, tasks, saving intervals, learning rate, etc can be changed as wanted. If you want to do the adapter tuning, use the script `adapter_full_data.sh`. The main differences are the three lines added:

```bash
--apply_adapter \
--adapter_type houlsby \
--adapter_size 12 \
```

These arguments will change the $T5_{Base}$ to adapter mode, with most of the parameters frozen during adapter-tuning. The `--method` should also change from `model` to `adapter`.

The checkpoints saved can be used to conduct interpolation on different initialization / different training step later.

#### Split data tuning

The example script given is `finetune_split_data.sh`. Under this tuning mode, the training dataset of target task will be divided evenly into two splits, and the model will be tuned respectively on these two splits. `--datasplit [split1/split2]` will control which split your model is currently tuning on.

The adapter-tuning version can also be trained with a few lines of revision to this script. The checkpoints saved can be used to conduct interpolation on different data splits.

#### Cartography Tuning

To normally get the result, please first put the `transformers` package under the root file, and change the file `generate_utils.py` in the package to the one in `./utils`.

Cartography tuning mode is used for recording the raw confidence for each piece of training data. The scripts for adapter-tuning version is given in  `adapter_cartography.sh`. `--cartography` will control whether the tuning is under cartography mode.

The fine-tuning version can also be trained with a few lines of revision to this script. The `cartography.json` file saved will be used as indices later to analyse knowledge change along the interpolation path.

#### Interpolation

All the scripts for interpolation starts with `itp`. Five example scripts are given in `./scripts`. In all these five scripts, there are three special arguments:

```bash
--load_PET_path_1 [path to the checkpoint used as the left end of interpolation path] \
--load_PET_path_2 [path to the checkpoint used as the right end of interpolation path] \
--itpl_points [total number of points interpolated (including both ends)] \
```

The `--task` here means the task used to valid and test along the interpolation path. Whether to valid or test are controlled by `--do_valid` and `--do_predict`.

Scripts `itp_diff_seed_adapter.sh` and `itp_diff_step_adapter.sh` are the basic examples, with the former one used to explore the mode connectivity between different adapter initializations, and the later one used to explore the mode connectivity between different fine-tuning steps. The interpolation result will be saved in `result.csv`.

Script `itp_same_domain_finetune.sh` is an example which uses two different tasks from same domain as two end points. These two tasks will also be used for interpolation respectively, and finally two curves will be got. This example is used to explore the mode connectivity of tasks in the same data domain. 

Script `itp_split_data_finetune.sh` explores the effect of data overlap on mode connectivity, with two end points come from models tuned on different training data splits from same task. The adapter-tuning checkpoints can also be interpolated by changing `--method` to `adapter` and changing the two paths accordingly.

Script `itp_traindata_adapter.sh` uses training data to test the knowledge change along the interpolation path, and records the whether each piece of training data is forgotten or not at each interpolated point. `--itp_on_train` controls whether training data will be used along the path instead of valid and test data. The result will be saved in `valid_data_analysis.json` for later processing. To change the ends points of your interpolation, please modify 2 load_PET_paths to the checkpoint file.

#### Auxiliary Scripts

Two auxiliary scripts are in `./utils`.

`process_cartography.sh` combines the raw indices from `cartography.json` and forgetting facts from `valid_data_analysis.json` and forms the final result.

```bash
python process_cartography.py \
--cartography_path [path to cartography dict containing raw confidence] \
--analysis_path [path to target valid data analysis dict] \
--save_path [path to save the result] \
```

`calculate_dist.sh` is used to calculate the L2 distance of any two checkpoints (but must be from the same tuning method!).

```
python calculate_dist.py \
--input1 [checkpoint path 1] \
--input2 [checkpoint path 2] \
--type [adapter/model] \
--path [path to save the result] \
```



## Experiments on $RoBERTa_{Base}$ 

All experiments on $RoBERTa_{Base}$ are under file `./RoBERTa_model`. The scripts are in `./RoBERTa_model/scripts`.

`run_glue.py` is the core file to generate fine-tuned or adapter-tuned models, and python files start with `itp` are used for interpolation.

#### Full data tuning

Two example scripts are `run_adapter.sh` and `run_finetune.sh`. The parameters can be found in the configuration file in `./RoBERTa_model/configs`. The function of some keys  parameters are list as following:

```
delta_type: [adapter/none]
task_name: [task to tune]
model_name_or_path: [use the default model 'roberta-base' or load the checkpoints]
unfrozen_modules: [applied when doing adapter-tuning, decides the modules that can be tuned]
```

The detailed information can be checked in `run_glue.py`.

#### Interpolation

Script `itp_diff_pretrain_adapter.sh` explores the mode connectivity of checkpoints adapter-tuned from different pre-train steps. Two new arguments added are:

```
path1: [path to the checkpoint as the left end of interpolation path]
path2: [path to the checkpoint as the right end of interpolation path]
```

They needs to be further changed in order to get more results of different interpolation paths. The number of interpolation points are by default set to 26. The fine-tuning version can also be interpolated with `delta_type` changed to `none` and paths changed accordingly.

Script `itp_task_boundary_finetune.sh` explores the task boundaries between two different tasks, with the end points of interpolation path tuned on tasks from different domains. All the other default settings are same as those in `itp_diff_pretrain_adapter.sh`.



## Code Structure

`tune_singletask.py`: The entry point for all tuning to save the checkpoints for interpolation.

`task_interpolation.py`: The entry point for all interpolation tasks on $T5_{Base}$.

`T5_model`: Including the trainer and model structure for $T5$ model.

`dataloader`: Processes the raw input data, and defines the metrics for evaluation.

`module`: Modules that aid the training of $T5$ model, including the adapter modules.

`utils`: To calculate the distance of checkpoints, help synthesizing the cartography results, and give the training parameters for $T5_{Base}$.

`RoBERTa_model`: Including the codes, configs and scripts for training and interpolating $RoBERTa_{Base}$ model.


## Acknowledgments

The authors would like to thank anonymous reviewers for their valuable feedback. The ``finding'' box in the paper is borrowed from [this paper](https://arxiv.org/abs/2104.08835).












