# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import random
import numpy as np
import json

import pandas as pd
from sympy import *
import torch

from T5_model.modeling_t5 import (
    T5ForConditionalGeneration,
)
from T5_model.configuration_t5 import T5Config
from transformers import T5Tokenizer

from T5_model.t5_trainer import Trainer
from utils.options import option


def model_provider(args):
    # only the master process download model

    config = T5Config.from_pretrained(
        args.model,
        apply_adapter=args.apply_adapter,
        adapter_type=args.adapter_type,
        adapter_size=args.adapter_size,
        r_mean=args.r_mean,
        r_std=args.r_std,
    )
    tokenizer = T5Tokenizer.from_pretrained(args.tokenizer_path)
    model = T5ForConditionalGeneration.from_pretrained(
        args.model, config=config)

    return model, config, tokenizer


def write_result(output_dir, result_name, dev_performance, best_dev_performance, valid_loss, test_performance, test_loss, args, df, prefix, metric, x):
    best_config = None
    if os.path.exists(os.path.join(output_dir, result_name)):
        df_load = pd.read_csv(os.path.join(output_dir, result_name), sep=',')
        if 'best' in df_load.prefix[len(df_load)-1]:
            best_dev_performance = df_load.dev_performance.iloc[-1]
            best_config = df_load.tail(1).values.tolist()[0]
            df_load.drop(len(df_load)-1, inplace=True)
        else:
            max_iloc = df_load['dev_performance'].argmax()
            best_config = df_load.iloc[[max_iloc]].values.tolist()[0]
            best_dev_performance = max(df_load.dev_performance)
        df = df_load

    if dev_performance > best_dev_performance:
        best_dev_performance = dev_performance
        best_valid_loss = valid_loss
        best_test_performance = test_performance
        best_test_loss = test_loss
        best_config = [prefix, metric, x, best_dev_performance,
                       best_valid_loss, best_test_performance, best_test_loss]
        if args.tune_method == 'model':
            pass

    df.loc[len(df.index)] = [prefix, metric, x, dev_performance,
                             valid_loss, test_performance, test_loss]
    df.to_csv(os.path.join(output_dir, result_name), sep=',',
              index=False, header=True, float_format='%.4f')
    return best_config, best_dev_performance, df


def main():
    args = option().parse()

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        print("Output directory () already exists and is not empty.")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    output_dir = args.output_dir

    # Start writing logs
    log_filename = "{}log.txt".format("" if args.do_train else "eval_")

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        handlers=[logging.FileHandler(os.path.join(args.output_dir, log_filename)),
                                  logging.StreamHandler()])
    logger = logging.getLogger(__name__)
    logger.info(args)
    logger.info(args.output_dir)

    args.seed = int(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.n_gpu = torch.cuda.device_count()

    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if args.do_train:
        if not args.train_file:
            raise ValueError(
                "If `do_train` is True, then `train_dir` must be specified.")
        if not args.dev_file:
            raise ValueError(
                "If `do_train` is True, then `predict_dir` must be specified.")

    if args.do_predict:
        if not args.test_file:
            raise ValueError(
                "If `do_predict` is True, then `predict_dir` must be specified.")

    logger.info("Using {} gpus".format(args.n_gpu))

    files = sorted(os.listdir(args.task_dir))
    prefixes = []
    for filename in files:
        if not filename.endswith(".tsv"):
            continue
        prefix = "_".join(filename.split("_")[:-1])
        if prefix not in prefixes:
            prefixes.append(prefix)

    logger.info("Fine-tuning the following samples: {}".format(prefixes))

    df = pd.DataFrame(columns=["prefix", "metric", "x",
                      "dev_performance", "dev_loss", "test_performance", "test_loss"])

    for prefix in prefixes:
        args.train_file = os.path.join(args.task_dir, prefix + "_train.tsv")
        if args.itp_on_train:
            args.dev_file = os.path.join(args.task_dir, prefix + "_train.tsv")
        else:
            args.dev_file = os.path.join(args.task_dir, prefix + "_dev.tsv")
        args.test_file = os.path.join(args.task_dir, prefix + "_test.tsv")

        best_dev_performance = -1.0
        best_config = None

        trainer = Trainer(args, logger, model_provider)
        model_dict = {k: v for (k, v) in trainer.model.state_dict().items()}

        def load_PET_from_path(path):
            load_PET = torch.load(path)
            if args.tune_method in load_PET:
                PET_state_dict = load_PET[args.tune_method]
            else:
                PET_state_dict = load_PET
            return PET_state_dict
        PET_state_dict_1 = load_PET_from_path(args.load_PET_path_1)
        PET_state_dict_2 = load_PET_from_path(args.load_PET_path_2)

        assert PET_state_dict_1.keys() == PET_state_dict_2.keys()

        last_raw_score_test = []
        last_raw_score_valid = []

        better_valid = []
        better_test = []
        better_valid_test = []
        total_valid_loss = 0
        total_test_loss = 0
        total_valid_performance = 0
        total_test_performance = 0
        valid_data_dict = {}
        test_data_dict = {}

        left_dev_loss = 0
        left_dev = 0
        right_dev_loss = 0
        right_dev = 0

        # get the results of end points first
        x = 0
        logger.info("calculating the performance on the left side")
        model_dict_to_update = {key: ((1-x)*PET_state_dict_1[key].cuda() + x*PET_state_dict_2[key].cuda())
                                for key in PET_state_dict_1.keys()}
        model_dict.update(model_dict_to_update)
        trainer.model.load_state_dict(model_dict)
        metric, dev_performance, valid_loss, raw_score = trainer.itp_valid(x=x)
        left_dev = dev_performance
        left_dev_loss = valid_loss
        metric, test_performance, test_loss, raw_score = trainer.itp_test(
            args=trainer.args, model=trainer.model, x=x)
        left_test = test_performance
        left_dev_loss = test_loss

        x = 1
        logger.info("calculating the performance on the right side")
        model_dict_to_update = {key: ((1-x)*PET_state_dict_1[key].cuda() + x*PET_state_dict_2[key].cuda())
                                for key in PET_state_dict_1.keys()}
        model_dict.update(model_dict_to_update)
        trainer.model.load_state_dict(model_dict)
        metric, dev_performance, valid_loss, raw_score = trainer.itp_valid(x=x)
        right_dev = dev_performance
        right_dev_loss = valid_loss
        metric, test_performance, test_loss, raw_score = trainer.itp_test(
            args=trainer.args, model=trainer.model, x=x)
        right_test = test_performance
        right_dev_loss = test_loss

        logger.info(
            "===========================START NOW=======================")

        for x in np.linspace(0, 1, args.itpl_points):
            x = round(x, 3)
            model_dict_to_update = {key: ((1-x)*PET_state_dict_1[key].cuda() + x*PET_state_dict_2[key].cuda())
                                    for key in PET_state_dict_1.keys()}
            model_dict.update(model_dict_to_update)
            trainer.model.load_state_dict(model_dict)

            dev_performance = 0
            test_performance = 0
            valid_loss = 0
            test_loss = 0

            if args.itp_on_train:
                logger.info("Valid ... prefix={}, x={}...".format(prefix, x))
                metric, dev_performance, valid_loss, raw_score = trainer.itp_valid(x=x)
                
                zero_list = []
                one_list = []
                with open(args.output_dir + "/data_analyze.txt", "a") as f:
                    f.write("total number of data is {} \n".format(len(raw_score)))
                    f.write("TEST ON TRAIN: round x = {}, in dev_data: \n".format(x))
                    for i in range(len(raw_score)):
                        if raw_score[i] == 0:
                            zero_list.append(i)
                        if raw_score[i] == 1:
                            one_list.append(i)
                    f.write("zero_list_num: {}\n".format(len(zero_list)))
                    f.write("one_list_num: {}\n".format(len(one_list)))

                    data_dict = {}
                    data_dict["zero_list"] = zero_list
                    data_dict["one_list"] = one_list
                    valid_data_dict[x] = data_dict 

            if not args.itp_on_train and args.do_valid:
                logger.info("Valid ... prefix={}, x={}...".format(prefix, x))
                metric, dev_performance, valid_loss, raw_score = trainer.itp_valid(x=x)
                total_valid_loss += valid_loss
                total_valid_performance += dev_performance
                if dev_performance > left_dev and dev_performance > right_dev:
                    better_valid.append(x)

                zero_list = []
                one_list = []
                up_list = []
                down_list = []
                down_to_zero = []
                up_to_one = []
                with open(args.output_dir + "/data_analyze.txt", "a") as f:
                    f.write("total number of data is {} \n".format(
                        len(raw_score)))
                    f.write("DEV: round x = {}, in dev_data: \n".format(x))
                    if x == 0:
                        for i in range(len(raw_score)):
                            if raw_score[i] == 0:
                                zero_list.append(i)
                            if raw_score[i] == 1:
                                one_list.append(i)
                                up_to_one.append(i)
                            if raw_score[i] > 0:
                                up_list.append(i)
                    else:
                        for i in range(len(raw_score)):
                            if raw_score[i] == 0:
                                zero_list.append(i)
                                if last_raw_score_valid[i] > 0:
                                    down_to_zero.append(i)
                            if raw_score[i] == 1:
                                one_list.append(i)
                                if last_raw_score_valid[i] < 1:
                                    up_to_one.append(i)
                            if raw_score[i] > last_raw_score_valid[i]:
                                up_list.append(i)
                            if raw_score[i] < last_raw_score_valid[i]:
                                down_list.append(i)
                    f.write("zero_list_num: {}\n".format(len(zero_list)))
                    f.write("one_list_num: {}\n".format(len(one_list)))
                    f.write("up_list_num: {}\n".format(len(up_list)))
                    f.write("up_to_one_num: {}\n".format(len(up_to_one)))
                    f.write("down_list_num: {}\n".format(len(down_list)))
                    f.write("down_to_zero_num: {}\n\n".format(
                        len(down_to_zero)))
                    last_raw_score_valid = raw_score

                    data_dict = {}
                    data_dict["zero_list"] = zero_list
                    data_dict["one_list"] = one_list
                    data_dict["up_list"] = up_list
                    data_dict["down_list"] = down_list
                    valid_data_dict[x] = data_dict

            if not args.itp_on_train and args.do_predict:
                logger.info("Test ... prefix={}, x={}...".format(prefix, x))
                metric, test_performance, test_loss, raw_score = trainer.itp_test(
                    args=trainer.args, model=trainer.model, x=x)
                total_test_loss += test_loss
                total_test_performance += total_test_performance
                if test_performance > left_test and test_performance > right_test:
                    better_test.append(x)
                    if len(better_valid) > 0 and better_valid[-1] == x:
                        better_valid_test.append(x)

                zero_list = []
                one_list = []
                up_list = []
                down_list = []
                down_to_zero = []
                up_to_one = []
                with open(args.output_dir + "/data_analyze.txt", "a") as f:
                    f.write("total number of data is {} \n".format(
                        len(raw_score)))
                    f.write("TEST: round x = {}, in test_data: \n".format(x))
                    if x == 0:
                        for i in range(len(raw_score)):
                            if raw_score[i] == 0:
                                zero_list.append(i)
                            if raw_score[i] == 1:
                                one_list.append(i)
                                up_to_one.append(i)
                            if raw_score[i] > 0:
                                up_list.append(i)
                    else:
                        for i in range(len(raw_score)):
                            if raw_score[i] == 0:
                                zero_list.append(i)
                                if last_raw_score_test[i] > 0:
                                    down_to_zero.append(i)
                            if raw_score[i] == 1:
                                one_list.append(i)
                                if last_raw_score_test[i] < 1:
                                    up_to_one.append(i)
                            if raw_score[i] > last_raw_score_test[i]:
                                up_list.append(i)
                            if raw_score[i] < last_raw_score_test[i]:
                                down_list.append(i)
                    f.write("zero_list_num: {}\n".format(len(zero_list)))
                    f.write("one_list_num: {}\n".format(len(one_list)))
                    f.write("up_list_num: {}\n".format(len(up_list)))
                    f.write("up_to_one_num: {}\n".format(len(up_to_one)))
                    f.write("down_list_num: {}\n".format(len(down_list)))
                    f.write("down_to_zero_num: {}\n\n".format(
                        len(down_to_zero)))
                    last_raw_score_test = raw_score

                    data_dict = {}
                    data_dict["zero_list"] = zero_list
                    data_dict["one_list"] = one_list
                    data_dict["up_list"] = up_list
                    data_dict["down_list"] = down_list
                    test_data_dict[x] = data_dict
            
            if not args.itp_on_train:
                result_name = "result.csv"
                best_config, best_dev_performance, df = write_result(
                    output_dir, result_name, dev_performance, best_dev_performance, valid_loss, test_performance, test_loss, args, df, prefix, metric, x)

        if args.itp_on_train:
            with open(output_dir + "/valid_data_analysis.json","w") as f:
                json.dump(valid_data_dict, f)
        else:
            with open(output_dir + "/valid_data_analysis.json", "w") as f:
                json.dump(valid_data_dict, f)
            with open(output_dir + "/test_data_analysis.json", "w") as f:
                json.dump(test_data_dict, f)

            with open(args.output_dir + "/solution_analysis.txt", "a") as f:
                f.write("better valid number: {}, which are {} \n".format(
                    len(better_valid), better_valid))
                f.write("better test number: {}, which are {} \n".format(
                    len(better_test), better_test))
                f.write("better valid & test number: {}, which are {} \n\n".format(
                    len(better_valid_test), better_valid_test))
                f.write("best valid performance: {} \n".format(best_dev_performance))
                f.write("average valid performance: {} \n".format(
                    total_valid_performance / (args.itpl_points)))
                f.write("average valid loss: {} \n".format(
                    total_valid_loss / (args.itpl_points)))
                f.write("average test performance: {} \n".format(
                    total_test_performance / (args.itpl_points)))
                f.write("average test loss: {} \n\n".format(
                    total_test_loss / (args.itpl_points)))

            result_name = "result.csv"
            write_result(output_dir, result_name, total_valid_performance / (args.itpl_points), best_dev_performance, total_valid_loss /
                        (args.itpl_points), total_test_performance / (args.itpl_points), total_test_loss / (args.itpl_points), args, df, "average", metric, x)

        if args.one_prefix:
            break


if __name__ == '__main__':
    main()
