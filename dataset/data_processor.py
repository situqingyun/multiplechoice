import json
import math
import random


def preprocess_data(input_file, split_eval=0.05):
    """
        切分训练集和验证集
    """
    with open(input_file, "r", encoding="utf-8-sig") as f:
        data_list = json.load(f)
    split_index = math.floor(len(data_list) * (1 - split_eval))

    train_list = data_list[:split_index]
    eval_list = data_list[split_index:]

    with open('dataset/haihua/train.json', "w", encoding="utf-8-sig") as f:
        train_list = random.shuffle(train_list)
        json.dump(train_list, f)
    with open('dataset/haihua/dev.json', "w", encoding="utf-8-sig") as f:
        json.dump(eval_list, f)


if __name__ == '__main__':
    preprocess_data('dataset/haihua/raw/train.json')
