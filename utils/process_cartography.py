import json
import numpy as np
from tqdm import tqdm
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--cartography_path', type=str, required=True)
    parser.add_argument('--analysis_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    args = parser.parse_args()

    with open(args.cartography_path, "r") as f:
        c_dict = json.load(f)
    
    with open(args.analysis_path, "r") as f:
        t_dict = json.load(f)

    all_var = []
    all_acc = []
    all_turn = []
    all_conf = []

    for key, value in tqdm(c_dict.items()):
        new_value = np.array(value["confidence"])
        var = np.std(new_value)
        all_var.append(var)

        conf = np.sum(new_value) / len(new_value)
        all_conf.append(conf)

        new_value = np.array(value["acc"])
        acc = np.sum(new_value) / len(new_value)
        all_acc.append(acc)

        turn_cnt = 0
        for i in range(len(value["acc"]) - 1):
            if value["acc"][i + 1] != value["acc"][i]:
                turn_cnt += 1
        all_turn.append(turn_cnt)
    
    
    with open(args.save_path, "w") as f:

        original_list = []
        original_key = "start"
        for key, value in t_dict.items():
            value = value["zero_list"]
            new_value = []
            for i in value:
                if i not in original_list:
                    new_value.append(i)

            print("len of original list ", len(original_list))
            print("len of cur list ", len(value))
            print("len of new ", len(new_value))
            print("remember sth? ", len(new_value) - (len(value) - len(original_list)))
            # input()

            original_list = value
            num_example = len(new_value)

            
            average_acc = 0.0
            average_turn = 0.0
            average_var = 0.0
            average_conf = 0.0
            
            if num_example != 0:
                for i in new_value:
                    average_acc += all_acc[int(i)]
                    average_turn += all_turn[int(i)]
                    average_var += all_var[int(i)]
                    average_conf += all_conf[int(i)]
                average_acc /= num_example
                average_turn /= num_example
                average_var /= num_example
                average_conf /= num_example

            print(average_acc, average_var, average_turn, average_conf)
        
            f.write("this is from interpolation point {} ===> {}\n".format(original_key, key))
            f.write("total number of newly forgotten examples: {}\n".format(num_example))
            f.write("average correctness is: {}\n".format(average_acc))
            f.write("average confidence is: {}\n".format(average_conf))
            f.write("average variance is: {}\n".format(average_var))
            f.write("average turn is: {}\n\n".format(average_turn))
            
            original_key = key