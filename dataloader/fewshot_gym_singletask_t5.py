import os
import torch
from .utils import MyQADataset, MyDataLoader
from collections import OrderedDict
from .metrics_t5 import METRICS, evaluate
from .itp_metrics_t5 import itp_evaluate


class NLPFewshotGymSingleTaskData(object):

    def __init__(self, logger, args, data_path, data_type, is_training):
        # should give the tasks used in this split in the var "tasks"
        self.data_path = data_path
        self.data_type = data_type
        self.split = args.datasplit
        self.data = []
        self.ori_outputs = []

        self.task_name = args.task_dir.split("/")[-1]

        with open(data_path) as fin:
            lines = fin.readlines()

        if args.cartography:
            for line in lines:
                d = line.strip().split("\t")
                new_ans = []
                # change the label for cartography analysis
                for ans in d[1:]:
                    if ans == "entailment":
                        new_ans.append("a")
                    if ans == "neutral":
                        new_ans.append("b")
                    if ans == "contradiction":
                        new_ans.append("c")
                self.data.append((d[0], new_ans))
                self.ori_outputs.append(new_ans[0])
        
        elif data_type == "train" and args.datasplit == "split1":
            length = int(len(lines) / 2)
            for line in lines[:length]:
                d = line.strip().split("\t")
                self.data.append((d[0], d[1:]))

        elif data_type == "train" and args.datasplit == "split2":
            length = int(len(lines) / 2)
            for line in lines[length:]:
                d = line.strip().split("\t")
                self.data.append((d[0], d[1:]))
        else:
            for line in lines:
                d = line.strip().split("\t")
                self.data.append((d[0], d[1:]))

        self.is_training = is_training
        self.logger = logger
        self.args = args

        self.metric = METRICS[self.task_name]
        self.tokenizer = None
        self.dataset = None
        self.dataloader = None
        self.cache = None

        self.gen_early_stop = False
        self.extra_id_0 = '<extra_id_0>'

    def __len__(self):
        return len(self.data)

    def flatten(self, answers):
        new_answers, metadata = [], []
        for answer in answers:
            metadata.append((len(new_answers), len(new_answers)+len(answer)))
            new_answers += answer
        return new_answers, metadata

    def load_dataset(self, tokenizer, do_return=False):
        self.tokenizer = tokenizer
        postfix = tokenizer.__class__.__name__.replace("zer", "zed")

        if self.args.cartography:
            preprocessed_path = os.path.join(
                "/".join(self.data_path.split("/")[:-1]),
                self.data_path.split("/")[-1].replace(".tsv", "-cartography-{}.pth".format(postfix)))
        else:
            preprocessed_path = os.path.join(
                "/".join(self.data_path.split("/")[:-1]),
                self.data_path.split("/")[-1].replace(".tsv", "-{}{}.pth".format(postfix, self.split)))

        if os.path.exists(preprocessed_path) and self.split == "":
            # load preprocessed input
            self.logger.info(
                "Loading pre-tokenized data from {}".format(preprocessed_path))
            preprocessed_data = torch.load(preprocessed_path)
            input_ids = preprocessed_data['input_ids']
            attention_mask = preprocessed_data['attention_mask']
            decoder_input_ids = preprocessed_data['decoder_input_ids']
            decoder_attention_mask = preprocessed_data['decoder_attention_mask']
            metadata = preprocessed_data['metadata']
            print("load from existed path")

        else:
            self.logger.info(
                "Start tokenizing ... {} instances".format(len(self.data)))

            inputs = []
            outputs = []

            for dp in self.data:
                inputs.append(" [{}] {}".format(self.task_name, dp[0]))
                output = []
                for d in dp[1]:
                    output.append(self.extra_id_0+d)
                outputs.append(output)  # is a list

            self.logger.info("Printing 3 examples")
            for i in range(3):
                self.logger.info(inputs[i])
                self.logger.info(outputs[i])

            outputs, metadata = self.flatten(outputs)

            if self.args.do_lowercase:
                inputs = [input0.lower() for input0 in inputs]
                outputs = [output0.lower() for output0 in outputs]
            if self.args.append_another_bos:
                inputs = ["<s> "+input0 for input0 in inputs]
                outputs = ["<s> " + output0 for output0 in outputs]

            self.logger.info("Tokenizing Input ...")
            tokenized_input = tokenizer.batch_encode_plus(inputs,
                                                          padding='max_length',
                                                          truncation=True,
                                                          return_tensors="pt",
                                                          max_length=self.args.max_input_length)
            self.logger.info("Tokenizing Output ...")
            tokenized_output = tokenizer.batch_encode_plus(outputs,
                                                           padding='max_length',
                                                           truncation=True,
                                                           return_tensors="pt",
                                                           max_length=self.args.max_output_length)

            input_ids, attention_mask = tokenized_input["input_ids"], tokenized_input["attention_mask"]
            decoder_input_ids, decoder_attention_mask = tokenized_output['input_ids'].masked_fill_(
                tokenized_output['input_ids'] == self.tokenizer.pad_token_id, -100), tokenized_output["attention_mask"]

            preprocessed_data = {}
            preprocessed_data['input_ids'] = input_ids
            preprocessed_data['attention_mask'] = attention_mask
            preprocessed_data['decoder_input_ids'] = decoder_input_ids
            preprocessed_data['decoder_attention_mask'] = decoder_attention_mask
            preprocessed_data['metadata'] = metadata
            torch.save(preprocessed_data, preprocessed_path)

        if self.args.choose_dev_1000 and self.data_type == 'dev' and len(input_ids) > 1000:
            self.data = self.data[:1000]
            input_ids = input_ids[:1000]
            attention_mask = attention_mask[:1000]
            len_of_decoder_input_ids = metadata[999][-1]
            decoder_input_ids = decoder_input_ids[:len_of_decoder_input_ids]
            decoder_attention_mask = decoder_attention_mask[:len_of_decoder_input_ids]
            metadata = metadata[:1000]
            print("Choose 1000 lines of dev dataset")
        if self.args.choose_test_1000 and self.data_type == 'test' and len(input_ids) > 1000:
            self.data = self.data[:1000]
            input_ids = input_ids[:1000]
            attention_mask = attention_mask[:1000]
            len_of_decoder_input_ids = metadata[999][-1]
            decoder_input_ids = decoder_input_ids[:len_of_decoder_input_ids]
            decoder_attention_mask = decoder_attention_mask[:len_of_decoder_input_ids]
            metadata = metadata[:1000]
            print("Choose 1000 lines of test dataset")
    
        self.dataset = MyQADataset(input_ids, attention_mask,
                                decoder_input_ids, decoder_attention_mask,
                                in_metadata=None, out_metadata=metadata,
                                is_training=self.is_training,
                                data_ans=self.ori_outputs)
        self.logger.info("Loaded {} examples from {} data".format(
            len(self.dataset), self.data_type))

        if do_return:
            return self.dataset

    def load_dataloader(self, do_return=False):
        self.dataloader = MyDataLoader(
            self.args, self.dataset, self.is_training)
        if do_return:
            return self.dataloader

    def evaluate(self, predictions, verbose=False):
        assert len(predictions) == len(self), (len(predictions), len(self))
        predictions = [prediction.strip() for prediction in predictions]
        self.logger.info("======print five predictions examples=====")
        self.logger.info(predictions[0])
        self.logger.info(predictions[1])
        self.logger.info(predictions[2])
        self.logger.info(predictions[3])
        self.logger.info(predictions[4])
        metrics = evaluate(self.logger, predictions, self.data, self.metric, self.args.cartography)
        if self.args.cartography:
            new_metrics = OrderedDict()
            new_metrics[self.metric] = metrics[self.metric]
            acc_list = metrics["cartography_acc"]
            return new_metrics, acc_list
        else:
            return metrics
    
    def itp_evaluate(self, predictions, verbose=False):
        assert len(predictions) == len(self), (len(predictions), len(self))
        predictions = [prediction.strip() for prediction in predictions]
        self.logger.info("======print five predictions examples=====")
        self.logger.info(predictions[0])
        self.logger.info(predictions[1])
        self.logger.info(predictions[2])
        self.logger.info(predictions[3])
        self.logger.info(predictions[4])
        metric_return = OrderedDict()
        return_dict = itp_evaluate(
            self.logger, predictions, self.data, self.metric)
        metric_return[self.metric] = return_dict[self.metric]
        list_return = return_dict["raw_score"]
        return metric_return, list_return
