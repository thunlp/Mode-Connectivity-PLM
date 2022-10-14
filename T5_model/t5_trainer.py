import os
import numpy as np
import torch
import random
import warnings
import json

from transformers import (
    Adafactor,
    get_scheduler,
    is_torch_available,
)
from dataloader.fewshot_gym_singletask_t5 import NLPFewshotGymSingleTaskData

from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings("ignore")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # ^^ safe to call this function even if cuda is not available


class Trainer:
    def __init__(self, args, logger, model_provider):
        self.args = args
        self.logger = logger
        logger.info("Loading model ...")

        self.model, self.config, self.tokenizer = model_provider(args)

        # save some random init checkpoints
        # if args.tune_method == "adapter" or args.tune_method == "model":
        #     path = './models/PET_{}_random_init/{}_init_seed_{}.pth'.format(args.tune_method, args.tune_method, args.seed)
        #     my_state_dict = self.model.state_dict()
        #     if args.tune_method == "adapter":
        #         state_dict = {k: my_state_dict[k] for k in my_state_dict if 'adapter_' in k}
        #     elif args.tune_method == "model":
        #         state_dict = my_state_dict
        #     torch.save(state_dict, path)
        # exit()

        if args.cartography:
            self.cartography_dict = OrderedDict()
        
        logger.info("Loading Dataset ...")
        self.train_data = NLPFewshotGymSingleTaskData(
            logger, args, args.train_file, data_type="train", is_training=True)
        self.train_data.load_dataset(self.tokenizer)
        self.train_data.load_dataloader()
        # cartography analysis uses training data to calculate indices at the end of every epoch
        # with the help of evaluation
        if args.cartography:
            self.dev_data = NLPFewshotGymSingleTaskData(
                logger, args, args.train_file, data_type="dev", is_training=False)
        else:
            self.dev_data = NLPFewshotGymSingleTaskData(
                logger, args, args.dev_file, data_type="dev", is_training=False)
        self.dev_data.load_dataset(self.tokenizer)
        self.dev_data.load_dataloader()
        self.test_data = NLPFewshotGymSingleTaskData(
            logger, args, args.test_file, data_type="test", is_training=False)
        self.test_data.load_dataset(self.tokenizer)
        self.test_data.load_dataloader()

        self.device = self.init_device()
        self.model = self.model.to(self.device)
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self.init_tensorboard(args)

        if args.seed is not None:
            set_seed(args.seed)
        self.prepare_data = self.prepare_model_data

        if args.tune_method == 'adapter' and args.SGD_noise:
            adapter_path = './models/PET_adapter_random_init/adapter_init_seed_{}.pth'.format(args.load_init_seed)
            state_dict = torch.load(adapter_path)
            model_dict = {k: v for (k, v) in self.model.state_dict().items()}
            model_dict.update({k: v.cuda() for (k, v) in state_dict.items()})
            self.model.load_state_dict(model_dict)
            logger.info(
                f'To keep the same adapter parameters as seed {args.load_init_seed}, loaded them from {adapter_path}')
        elif args.tune_method == 'model' and args.SGD_noise:
            model_path = './models/PET_model_random_init/model_init_seed_{}.pth'.format(args.load_init_seed)
            state_dict = torch.load(model_path)
            model_dict = {k: v for (k, v) in self.model.state_dict().items()}
            model_dict.update({k: v.cuda() for (k, v) in state_dict.items()})
            self.model.load_state_dict(model_dict)
            logger.info(
                f'To keep the same model parameters as seed {args.load_init_seed}, loaded them from {model_path}')

    def init_device(self):
        if (not torch.cuda.is_available()):
            print('no gpu can be used!')
            assert torch.cuda.is_available()
        else:
            return torch.device('cuda:0')

    def init_tensorboard(self, args):
        self.tensorboard = None
        args.tensorboard_dir = args.output_dir + '/tensorboard'
        self.tensorboard = SummaryWriter(log_dir=args.tensorboard_dir)

    def get_optimzied_group(self):
        if self.args.tune_method == 'model':
            no_decay = ["bias", "layer_norm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            to_update = self.model.parameters()
        elif self.args.tune_method == 'adapter':
            no_decay = ["bias", "layer_norm.weight"]
            for n, p in self.model.named_parameters():
                p.requires_grad = False
                if "adapter" in n:
                    p.requires_grad = True
            optimizer_grouped_parameters = []
            for n, p in self.model.named_parameters():
                if "adapter" in n:
                    optimizer_grouped_parameters.append({'params': [p]})
            to_update = self.model.parameters()
            sum = 0
            for group in optimizer_grouped_parameters:
                for param in group['params']:
                    sum += param.numel()
        return optimizer_grouped_parameters, to_update

    def train(self):
        if not os.path.exists(self.args.output_dir):
            os.mkdir(self.args.output_dir)
        self.model.train()

        train_dataloader = self.train_data.dataloader

        if self.args.train_iters is None:
            self.args.train_iters = (
                len(train_dataloader) // self.gradient_accumulation_steps
                * float(self.args.train_epochs)
            )
        if self.args.train_epochs is None:
            self.args.train_epochs = (self.args.train_iters * self.gradient_accumulation_steps) \
                // len(train_dataloader) + 1

        # setup optimizer
        optimizer_grouped_parameters, to_update_parameters = self.get_optimzied_group()
        self.optimizer = self.build_optimizer(
            self.args, optimizer_grouped_parameters)
        warm_up_steps = int(self.args.train_iters) * self.args.warmup_rate
        self.scheduler = get_scheduler(
            'constant', self.optimizer, warm_up_steps, self.args.train_iters)

        num_updates = 0
        log_dict = OrderedDict()
        best_metric = 0
        best_num_updates = 0
        early_stop = 0
        self.logger.info(
            f"Train {len(train_dataloader) // self.gradient_accumulation_steps} steps every epoch")

        for epoch in range(self.args.train_epochs):
            self.optimizer.zero_grad()
            self.reset_logging(log_dict)
            for local_step, batch in enumerate(train_dataloader):
                loss = self.train_step(batch)
                self.add_logging(log_dict, 'loss', loss.item()
                                 * self.gradient_accumulation_steps)
                if local_step % self.gradient_accumulation_steps == 0:
                    # update model parameter
                    updated, old_scale = self.optimizer_step(
                        self.model.parameters())
                    if updated:
                        num_updates += 1
                    else:
                        self.logger.info("Inf or NaN detected in grad. Change scale from {:.1f} to {:.1f}"
                                         .format(old_scale, self.scaler.get_scale()))
                    if num_updates % self.args.log_interval == 0:
                        # to log
                        train_loss_mean = self.log_step(log_dict, tensorboard_suffix='train', epoch=epoch, num_updates=num_updates,
                                                        lr=self.scheduler.get_last_lr()[0])
                    self.reset_logging(log_dict)
                    if self.args.valid_interval is not None and not self.args.cartography and \
                            num_updates % self.args.valid_interval == 0:
                        current_metrics = self.valid(epoch, num_updates)
                        best_update, average_score = self.early_stop(
                            current_metrics, best_metric, epoch, num_updates)
                        if not best_update or train_loss_mean < 1e-7:
                            early_stop += 1
                            self.logger.info(f"Early stop + 1 = {early_stop}. "
                                             f"Best averate score = {best_metric} at {best_num_updates}.")
                        else:
                            early_stop = 0
                            best_metric = average_score
                            best_num_updates = num_updates
                        if self.args.early_stop > 0 and early_stop >= self.args.early_stop:
                            break
                    if self.args.output_interval is not None and not self.args.cartography and \
                            num_updates % self.args.output_interval == 0:
                        save_path = f"{self.args.output_dir}/checkpoint@{num_updates}.pt"
                        self.save_checkpoint(save_path, epoch, num_updates)

                    if num_updates >= self.args.train_iters:
                        break
            
            if self.args.cartography:
                current_metrics = self.valid(epoch, num_updates)

            if self.args.early_stop > 0 and early_stop >= self.args.early_stop:
                self.logger.info(
                    f"Stop training. Best averate score = {best_metric:.3f} at {best_num_updates}.")
                break

            if num_updates >= self.args.train_iters:
                break

        if self.args.tune_method != 'model':
            save_path = f"{self.args.output_dir}/checkpoint-last.pt"
            self.save_checkpoint(save_path, epoch, num_updates)
        return best_metric

    def early_stop(self, metrics, best_metric, epoch, num_updates):
        current_metric = 0
        update = True
        for key in metrics:
            current_metric += metrics[key]
        current_metric = current_metric / len(metrics)  # compare average
        if best_metric > current_metric:
            update = False
        else:
            save_path = f"{self.args.output_dir}/checkpoint-best.pt"
            self.save_checkpoint(save_path, epoch, num_updates)
        return update, current_metric

    def valid(self, epoch=0, num_updates=0):
        self.model.eval()
        valid_dataloader = self.dev_data.dataloader
        my_prediction = []
        raw_confidence=[]
        valid_log_dict = OrderedDict()
        self.logger.info("Begin validation on {:d} samples ...".format(
            len(self.dev_data.dataset)))
        metrics = {}

        with torch.no_grad():
            for local_step, batch in enumerate(valid_dataloader):
                if self.args.cartography:
                    std_ans = batch[-1]
                    std_ans = list(std_ans)

                all_input = self.prepare_data(batch)
                output = self.model(**all_input)
                valid_loss = output[0]
                self.add_logging(valid_log_dict, 'loss', valid_loss.item())

                decoder_input_ids = self.get_decoder_input_ids(
                    all_input["inputs_embeds"])
                if self.args.cartography:
                    generated_ids, p_acc = self.model.generate(
                        inputs_embeds=all_input["inputs_embeds"],
                        attention_mask=all_input["attention_mask"],
                        decoder_input_ids=decoder_input_ids,
                        max_length=self.args.max_output_length,
                        early_stopping=True,
                        compute_acc=True,
                        std_ans=std_ans
                    )
                    raw_confidence.extend(p_acc)
                else:
                    generated_ids = self.model.generate(
                        inputs_embeds=all_input["inputs_embeds"],
                        attention_mask=all_input["attention_mask"],
                        decoder_input_ids=decoder_input_ids,
                        max_length=self.args.max_output_length,
                        early_stopping=True
                    )
                gen_text = self.tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                gen_text = list(map(str.strip, gen_text))
                my_prediction.extend(gen_text)
        
        # begin to record the raw confidence in this epoch
        if self.args.cartography:
            metrics, acc_list = self.dev_data.evaluate(my_prediction, verbose=False)
            for i in range(len(self.dev_data.dataset)):
                if i not in self.cartography_dict:
                    self.cartography_dict[i] = {"acc":[float(acc_list[i])], "confidence":[float(raw_confidence[i])]}
                else:
                    self.cartography_dict[i]["acc"].append(float(acc_list[i]))
                    self.cartography_dict[i]["confidence"].append(float(raw_confidence[i]))
            
            with open(self.args.output_dir + "/../cartography.json", 'w') as f:
                json.dump(self.cartography_dict, f)
                self.logger.info("write into json file successfully!")
        else:
            metrics = self.dev_data.evaluate(my_prediction, verbose=False)

        valid_loss = self.log_step(valid_log_dict, suffix="Valid |", tensorboard_suffix='valid',
                                   epoch=epoch, num_updates=num_updates, **metrics)

        self.model.train()
        return metrics

    def test(self, epoch=0, num_updates=0):
        load_best_path = f"{self.args.output_dir}/checkpoint-best.pt"
        self.load_checkpoint(load_best_path)
        self.model.eval()

        test_dataloader = self.test_data.dataloader
        my_prediction = []
        test_log_dict = OrderedDict()
        self.logger.info("Begin test on {:d} samples ...".format(
            len(self.test_data.dataset)))
        metrics = {}

        with torch.no_grad():
            for local_step, batch in enumerate(test_dataloader):
                all_input = self.prepare_data(batch)
                output = self.model(**all_input)
                test_loss = output[0]
                self.add_logging(test_log_dict, 'loss', test_loss.item())

                decoder_input_ids = self.get_decoder_input_ids(
                    all_input["inputs_embeds"])
                generated_ids = self.model.generate(
                    inputs_embeds=all_input["inputs_embeds"],
                    attention_mask=all_input["attention_mask"],
                    decoder_input_ids=decoder_input_ids,
                    max_length=self.args.max_output_length,
                    early_stopping=True
                )
                gen_text = self.tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                gen_text = list(map(str.strip, gen_text))
                my_prediction.extend(gen_text)

        if len(my_prediction) != 0:
            metrics = self.test_data.evaluate(my_prediction, verbose=False)

        test_loss = self.log_step(test_log_dict, suffix="Test |", tensorboard_suffix='test',
                                  epoch=epoch, num_updates=num_updates, **metrics)

        self.model.train()
        return metrics

    def itp_valid(self, epoch=0, num_updates=0, **kwargs):
        self.model.eval()
        valid_dataloader = self.dev_data.dataloader
        my_prediction = []
        valid_log_dict = OrderedDict()
        self.logger.info("Begin validation on {:d} samples ...".format(
            len(self.dev_data.dataset)))
        metrics = {}

        with torch.no_grad():
            for local_step, batch in enumerate(valid_dataloader):
                all_input = self.prepare_data(batch)
                output = self.model(**all_input)
                valid_loss = output[0]
                self.add_logging(valid_log_dict, 'loss', valid_loss.item())

                decoder_input_ids = self.get_decoder_input_ids(
                    all_input["inputs_embeds"])
                generated_ids = self.model.generate(
                    inputs_embeds=all_input["inputs_embeds"],
                    attention_mask=all_input["attention_mask"],
                    decoder_input_ids=decoder_input_ids,
                    max_length=self.args.max_output_length,
                    early_stopping=True
                )
                gen_text = self.tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                gen_text = list(map(str.strip, gen_text))
                my_prediction.extend(gen_text)

        if len(my_prediction) != 0:
            metrics, raw_score = self.dev_data.itp_evaluate(
                my_prediction, verbose=False)

        valid_loss = self.log_step(valid_log_dict, suffix="Valid |", tensorboard_suffix='valid',
                                   epoch=epoch, num_updates=num_updates, **metrics)

        for i, j in metrics.items():
            metric = i
            valid_performance = j

        return metric, valid_performance, valid_loss, raw_score

    def itp_test(self, args=None, model=None, **kwargs):
        model.eval()
        test_dataloader = self.test_data.dataloader
        my_prediction = []
        test_log_dict = OrderedDict()
        self.logger.info("Begin test on {:d} samples ...".format(
            len(self.test_data.dataset)))
        metrics = {}

        with torch.no_grad():
            for local_step, batch in enumerate(test_dataloader):
                all_input = self.prepare_data(batch)
                output = model(**all_input)
                test_loss = output[0]
                self.add_logging(test_log_dict, 'loss', test_loss.item())

                decoder_input_ids = self.get_decoder_input_ids(
                    all_input["inputs_embeds"])
                generated_ids = model.generate(
                    inputs_embeds=all_input["inputs_embeds"],
                    attention_mask=all_input["attention_mask"],
                    decoder_input_ids=decoder_input_ids,
                    max_length=args.max_output_length,
                    early_stopping=True
                )
                gen_text = self.tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                my_prediction.extend(gen_text)

        if len(my_prediction) != 0:
            metrics, raw_score = self.test_data.itp_evaluate(
                my_prediction, verbose=False)

        test_loss = self.log_step(test_log_dict, suffix="Test |", tensorboard_suffix='test',
                                  **metrics, **kwargs)

        for i, j in metrics.items():
            metric = i
            test_performance = j

        return metric, test_performance, test_loss, raw_score

    def get_decoder_input_ids(self, inputs_embeds):
        decoder_start_token_id = self.config.decoder_start_token_id
        decoder_input_ids = (
            torch.ones((inputs_embeds.shape[0], 1), dtype=torch.long,
                       device=inputs_embeds.device) * decoder_start_token_id
        )
        return decoder_input_ids

    def save_checkpoint(self, path, epoch, num_updates):
        state_dict = OrderedDict()
        if self.args.tune_method == 'model':
            state_dict['model'] = self.model.state_dict()
        elif self.args.tune_method == 'adapter':
            my_state_dict = self.model.state_dict()
            state_dict['adapter'] = {k: my_state_dict[k]
                                     for k in my_state_dict if 'adapter_' in k}
        state_dict['optimizer'] = self.optimizer.state_dict()
        state_dict['scheduler'] = self.scheduler.state_dict()
        state_dict['config'] = self.config
        state_dict['args'] = vars(self.args)
        state_dict['current_state'] = {
            'epoch': epoch, 'num_updates': num_updates}
        torch.save(state_dict, path)
        self.logger.info(
            f"epoch: {epoch} num_updates: {num_updates} Save {self.args.tune_method} to {path}.")

    def load_checkpoint(self, path):
        state_dict = torch.load(path)
        if state_dict['args']['tune_method'] == 'model':
            self.model.load_state_dict(state_dict['model'])
        elif state_dict['args']['tune_method'] == 'adapter':
            model_dict = {k: v for (k, v) in self.model.state_dict().items()}
            model_dict.update({k: v.cuda()
                              for (k, v) in state_dict['adapter'].items()})
            self.model.load_state_dict(model_dict)
        current_state = state_dict['current_state']
        self.logger.info(
            f"Load {state_dict['args']['tune_method']} from {path}.")
        return current_state

    def build_optimizer(self, args, params):
        optimizer = Adafactor(params, lr=args.learning_rate,
                              scale_parameter=False, relative_step=False, warmup_init=False)
        return optimizer

    def prepare_model_data(self, batch):
        all_input = {
            'input_ids': batch[0].to(self.device),
            'attention_mask': batch[1].to(self.device),
            'labels': batch[2].to(self.device),
            'decoder_attention_mask': batch[3].to(self.device),
        }
        input_ids = all_input.pop('input_ids')
        input_embeds = self.model.get_input_embeddings()(input_ids)
        all_input['inputs_embeds'] = input_embeds
        return all_input

    def train_step(self, batch):
        all_input = self.prepare_data(batch)
        output = self.model(**all_input)
        loss = output[0] / self.gradient_accumulation_steps
        loss.backward()
        return loss

    def optimizer_step(self, parameters):
        updated = True
        scale = 0
        torch.nn.utils.clip_grad_norm_(parameters, self.args.max_grad_norm)
        self.optimizer.step()
        if updated:
            self.scheduler.step()
        self.optimizer.zero_grad()
        return updated, scale

    def log_step(self, log_dict, suffix='', tensorboard_suffix=None, **kwargs):
        new_log_dict = OrderedDict()
        for key, value in kwargs.items():
            new_log_dict[key] = value
        for key in log_dict:
            key_tensor = torch.tensor(log_dict[key], device=self.device)

            key_value = key_tensor.mean().item()
            new_log_dict[key] = key_value
        message = '' + suffix
        for key, value in new_log_dict.items():
            if isinstance(value, float):
                message += ' {:s}: {:.5f}'.format(key, value)
            else:
                message += ' {:s}: {}'.format(key, value)
        self.logger.info(message)
        if self.tensorboard is not None:
            for key, value in new_log_dict.items():
                if key in ['epoch', 'num_updates']:
                    continue
                tag = f'{tensorboard_suffix}/{key}' if tensorboard_suffix is not None else key
                global_step = kwargs.get('num_updates', None)
                self.tensorboard.add_scalar(
                    tag, value, global_step=global_step)
        return new_log_dict.get('loss', None)

    def add_logging(self, log_dict, key, value):
        if key not in log_dict:
            log_dict[key] = []
        log_dict[key].append(value)

    def reset_logging(self, log_dict):
        for key in log_dict:
            log_dict[key] = []
