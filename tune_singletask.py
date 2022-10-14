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
import shutil

import random
import numpy as np
import torch

import pandas as pd

from T5_model.modeling_t5 import T5ForConditionalGeneration
from T5_model.configuration_t5 import T5Config
from transformers import T5Tokenizer

# from run_singletask_t5 import run
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

    df = pd.DataFrame(columns=["prefix", "metric", "lr",
                      "bsz", "dev_performance", "test_performance"])

    for prefix in prefixes:
        args.train_file = os.path.join(args.task_dir, prefix + "_train.tsv")
        args.dev_file = os.path.join(args.task_dir, prefix + "_dev.tsv")
        args.test_file = os.path.join(args.task_dir, prefix + "_test.tsv")

        best_dev_performance = -1.0
        for bsz in args.bsz_list:
            for lr in args.learning_rate_list:

                args.learning_rate = lr
                if bsz > 16:
                    args.train_batch_size = 16
                    args.gradient_accumulation_steps = int(bsz // 16)
                else:
                    args.train_batch_size = bsz
                    args.gradient_accumulation_steps = 1

                args.output_dir = output_dir + '/lr_' + \
                    str(lr)+'_bsz_'+str(bsz)+'_seed_'+str(args.seed)
                if os.path.exists(f"{args.output_dir}/checkpoint-last.pt"):
                    logger.info(
                        "Done ... prefix={}, lr={}, bsz={} ...!!!".format(prefix, lr, bsz))
                    continue
                logger.info(
                    "Running ... prefix={}, lr={}, bsz={} ...".format(prefix, lr, bsz))
                trainer = Trainer(args, logger, model_provider)

                dev_performance = None
                test_performance = None
                if args.do_train:
                    dev_performance = trainer.train()
                if args.cartography:
                    continue
                if args.do_predict:
                    metrics = trainer.test()
                    for i, j in metrics.items():
                        metric = i
                        test_performance = j

                if os.path.exists(os.path.join(output_dir, "result.csv")):
                    df_load = pd.read_csv(os.path.join(
                        output_dir, "result.csv"), sep=',')
                    if 'best' in df_load.prefix[len(df_load)-1]:
                        best_dev_performance = df_load.dev_performance.iloc[-1]
                        df_load.drop(len(df_load)-1, inplace=True)
                    else:
                        max_iloc = df_load['dev_performance'].argmax()
                        best_dev_performance = max(df_load.dev_performance)
                    df = df_load

                if dev_performance > best_dev_performance:
                    best_dev_performance = dev_performance
                    best_test_performance = test_performance
                    best_output_dir = args.output_dir
                    best_config = [prefix, metric, lr, bsz,
                                   best_dev_performance, best_test_performance]

                    if args.tune_method == 'model':
                        os.remove(os.path.join(
                            best_output_dir, 'checkpoint-best.pt'))
                    else:
                        if os.path.exists(os.path.join(best_output_dir, 'checkpoint-best.pt')):
                            shutil.copy(
                                os.path.join(best_output_dir,
                                             'checkpoint-best.pt'),
                                os.path.join(output_dir, 'checkpoint-best.pt')
                            )

                logger.info("prefix={}, lr={}, bsz={}, dev_performance={}, test_performance={}".format(
                    prefix, lr, bsz, dev_performance, test_performance))
                df.loc[len(df.index)] = [prefix, metric, lr, bsz,
                                         dev_performance, test_performance]
                df.to_csv(os.path.join(output_dir, "result.csv"),
                          sep=',', index=False, header=True)

        if args.one_prefix:
            break


if __name__ == '__main__':
    main()
