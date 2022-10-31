from scipy.special import comb
import torch
import torch.nn.functional as F
from torch import Tensor, nn
import numpy as np
import random
import math

from modeling_t5_multiHyper_flatten_pet import T5PreTrainedModel, T5ForConditionalGeneration

class MyT5_pet_CF(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config
        self.bend_num = args.bend_num
        
        self.end_w = self.faltten(args.load_stage1_pet_path_list, args.pet_type)
        self.end_w.requires_grad = False
                
        self.flatten_size = self.end_w.size()[1]
        self.train_theta = nn.Parameter(torch.zeros(args.bend_num, self.flatten_size))

        self.model_AL = T5ForConditionalGeneration.from_pretrained(args.model, config=self.config) #网络结构同时存在，但是不同时forwad
        self.init_weight()
        self.train_theta.requires_grad = True
    
    def init_weight(self):
        
        # 是否需要根据lora的kaiming_uniform初始化
        if self.args.apply_adapter:
            self.train_theta.data.normal_(mean=0.0, std=0.02)
        elif self.args.apply_lora:
            
            lora_flatten_len = self.config.lora_r * self.config.d_model
            for i in range(0,144,2):
                nn.init.kaiming_uniform_(self.train_theta[:,i*lora_flatten_len:(i+1)*lora_flatten_len], a=math.sqrt(5))
                nn.init.zeros_(self.train_theta[:,(i+1)*lora_flatten_len:(i+2)*lora_flatten_len])
        elif self.args.apply_prefix:
            prefix_len = self.model_AL.encoder.prefix.prefix.numel()
                        
            self.train_theta.data[:,0:prefix_len].normal_(mean=0.0, std=0.02)
            nn.init.kaiming_uniform_(self.train_theta[:,prefix_len:int(self.flatten_size/2)], a=math.sqrt(5))
            self.train_theta.data[:,int(self.flatten_size/2):(int(self.flatten_size/2)+prefix_len)].normal_(mean=0.0, std=0.02)
            nn.init.kaiming_uniform_(self.train_theta[:,(int(self.flatten_size/2)+prefix_len):self.flatten_size], a=math.sqrt(5))
        
    def faltten(self, ckpt_path_list, pet_type):
        def faltten_from_path(ckpt_path_list, pet_type):
            flatten_all = []
            for ckpt_path in ckpt_path_list:
                ckpt = torch.load(ckpt_path)
                if pet_type in ckpt:
                    pet_dict = ckpt[pet_type]
                else:
                    pet_dict = ckpt
                pet_name_modules = list(pet_dict.keys())
                flatten = torch.Tensor([]).cuda()
                for pet_name_module in pet_name_modules:
                    flatten = torch.cat((flatten, pet_dict[pet_name_module].flatten().cuda()),dim=0)
                flatten_all.append(flatten)
            flatten_all = torch.stack(flatten_all,dim=0)
            return flatten_all, len(pet_name_modules)
        
        if pet_type=='adapter':
            flatten_adapter_all, num_module = faltten_from_path(ckpt_path_list, pet_type)
            assert num_module==120, f"Num of {pet_type} modules should be 120!"
            return flatten_adapter_all
        if pet_type=='lora':
            flatten_lora_all, num_module = faltten_from_path(ckpt_path_list, pet_type)
            assert num_module==144, f"Num of {pet_type} modules should be 144!"
            return flatten_lora_all
        if pet_type=='prefix':
            flatten_prefix_all, num_module = faltten_from_path(ckpt_path_list, pet_type)
            assert num_module==6, f"Num of {pet_type} modules should be 6!"
            return flatten_prefix_all
        if pet_type=='model':
            flatten_model_all, num_module = faltten_from_path(ckpt_path_list, pet_type)
            assert num_module==284, f"Num of {pet_type} modules should be 284!"
            return flatten_model_all
        
                
    def forward(self, all_input):
        t = random.uniform(0,1)
        loss = torch.Tensor([0]).cuda()
        n = self.bend_num
        P_alpha = pow(1-t, n+1) * self.end_w[0] + pow(t, n+1) * self.end_w[1]
        coeff = []
        for i in range(n):
            coeff.append(pow(t, i+1) * pow(1-t, n-i) * comb(n+1,i+1))
        flatten_pet = P_alpha + torch.Tensor(coeff).cuda() @ self.train_theta
        output = self.model_AL(**all_input, only_adapter=self.args.apply_adapter, only_lora=self.args.apply_lora, only_prefix=self.args.apply_prefix, flatten_pet=flatten_pet)
        loss = output[0]
        return loss