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
        # t_list = [0.5]
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

def get_group_layer_name(name_modules, layer_filter):
    name_prefix_group = []
    name_prefix_group_dict = {}
    if layer_filter=='block':
        index=3
    elif layer_filter=='layer':
        index=5
    for name in name_modules:
        name_list = name.split('.')
        if layer_filter in name_list:
            prefix = '.'.join(name_list[:index])
            if prefix in name_prefix_group:
                name_prefix_group_dict[name] = name_prefix_group.index(prefix)
            else:
                name_prefix_group.append(prefix)
                name_prefix_group_dict[name] = name_prefix_group.index(prefix)
        else:
            name_prefix_group.append(name)
            name_prefix_group_dict[name] = name_prefix_group.index(name)
    return name_prefix_group, name_prefix_group_dict

class WrapModelLayerWF:
    def __init__(self, module: nn.Module, args, device="cpu"):
        super().__init__()
        self.args = args
        self.layer_filter = args.layer_filter
        self.device = device
        
        self.name_base_localname = []
        self.end_w = self.faltten(args.load_stage1_pet_path_list, args.pet_type)
        self.end_w.requires_grad = False
        
        self.model_dict_1 = torch.load(args.load_stage1_pet_path_list[0])['model']
        self.model_dict_2 = torch.load(args.load_stage1_pet_path_list[1])['model']
        
        
        pet_name_modules = []
        flatten = torch.Tensor([]).cuda()
        for pet_name_module, params in module.named_parameters():
            flatten = torch.cat((flatten, params.flatten().cuda()),dim=0)
            pet_name_modules.append(pet_name_module)
        self.pet_name_modules = pet_name_modules

        if self.layer_filter in ['block', 'layer']:
            name_prefix_group, name_prefix_group_dict = get_group_layer_name(self.pet_name_modules, self.layer_filter)
        elif self.layer_filter=='weight':
            name_prefix_group = self.pet_name_modules
            name_prefix_group_dict = {}
            for i,name in enumerate(name_prefix_group):
                name_prefix_group_dict[name] = i

        self.name_prefix_group = name_prefix_group
        self.name_prefix_group_dict = name_prefix_group_dict
        self.weight_num = len(name_prefix_group)
        self.layer_weight = nn.Parameter(torch.zeros(self.weight_num).cpu() if device=="cpu" else torch.zeros(self.weight_num).cuda())
        self.layer_weight.data.normal_(mean=0.0, std=args.init_std)
        self.init_state = module.state_dict()
        
        module.register_parameter(
            "layer_weight", self.layer_weight)
        setattr(module, "layer_weight", self.layer_weight)
        
        self.layer_weight.requires_grad = True
        
        for name, param in module.named_parameters():
            if 'adapter' not in name and 'lora' not in name and 'prefix' not in name:
                base, localname = module, name
                while "." in localname:
                    prefix, localname = localname.split(".", 1)
                    base = base.__getattr__(prefix)
                self.name_base_localname.append((name, base, localname))

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
                    if 'embed_tokens' not in pet_name_module:
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
    
    
    def __call__(self, module, inputs, kwargs):
        index = 0
        layer_weight_norm = torch.sigmoid(self.layer_weight)
        with torch.enable_grad():
            for name, base, localname in self.name_base_localname:
                if localname == "layer_weight":
                    continue
                if 'adapter' in name or 'lora' in name or 'prefix' in name:
                    continue
                index = self.name_prefix_group_dict[name]
                param = layer_weight_norm[index] * self.model_dict_1[name] + (1 - layer_weight_norm[index]) * self.model_dict_2[name]
                
                delattr(base, localname)
                setattr(base, localname, param)

    @staticmethod
    def apply(module, args, device="cpu"):
        for k, hook in module._forward_pre_hooks.items():
            assert False
            if isinstance(hook, WrapModelLayerWF) and hook.name == name:
                raise RuntimeError("Cannot register two intrinsic dimension hooks on "
                                   "the same parameter {}".format(name))
        fn = WrapModelLayerWF(
            module, args, device)
        module.register_forward_pre_hook(fn)
        return fn
    
def layer_weight_find_finetune(module, args, device="cpu"):
    ID_wrap = WrapModelLayerWF.apply(
        module, args, device)
    return module, ID_wrap

        