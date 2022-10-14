import argparse
import math
import torch
import torch.nn as nn


def flatten(ckpt_path, pet_type):
    def faltten_from_path(ckpt_path, pet_type):
        ckpt = torch.load(ckpt_path)
        if pet_type in ckpt:
            pet_dict = ckpt[pet_type]
        else:
            pet_dict = ckpt
        pets_name_modules = list(pet_dict.keys())

        flatten_pet = torch.Tensor([]).cuda()
        for pet_name_module in pets_name_modules:
            flatten_pet = torch.cat(
                (flatten_pet, pet_dict[pet_name_module].flatten().cuda()), dim=0)

        return flatten_pet, len(pets_name_modules)

    if pet_type == 'adapter':
        flatten_adapter, num_module = faltten_from_path(ckpt_path, 'adapter')
        print("num_of_models:", num_module)
        return flatten_adapter
    if pet_type == 'model' or pet_type == 'finetune':
        flatten_model, num_module = faltten_from_path(ckpt_path, 'model')
        print("num_of_models:", num_module)
        return flatten_model


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input1', type=str, default="")
    parser.add_argument('--input2', type=str, default="")
    parser.add_argument('--type', type=str, default="")
    parser.add_argument('--path', type=str, default="")
    args = parser.parse_args()

    if args.path == "":
        a = flatten(args.input1, args.type)
        b = flatten(args.input2, args.type)
        cos_sim = nn.CosineSimilarity(dim=0, eps=1e-6)
        sim = cos_sim(a, b)

        print('=====================L2 distance===================')
        dis = torch.dist(a, b).item()
        print(dis)
        print('====================Cosine distance====================')
        print(sim.item())
        _len = len(a)
        average = math.sqrt(dis * dis / _len)
        print('====================num of para====================')
        print(_len)
        print('====================average distance====================')
        print(average)

    else:
        a = flatten(args.input1, args.type)
        b = flatten(args.input2, args.type)
        cos_sim = nn.CosineSimilarity(dim=0, eps=1e-6)
        sim = cos_sim(a, b)

        path = args.path + "/distance.txt"
        dis = torch.dist(a, b).item()
        _len = len(a)
        average = math.sqrt(dis * dis / _len)

        with open(path, "w") as f:
            f.write("L2 distance is: \n")
            f.write(str(dis))
            f.write("\n")
            f.write("num of para is: \n")
            f.write(str(_len))
            f.write("\n")
            f.write("average L2 distance is: \n")
            f.write(str(average))
            f.write("\n")
            f.write("Cos distance is: \n")
            f.write(str(sim.item()))
            f.write("\n")
