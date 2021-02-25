import pandas as pd
import math

LABEL_LIST = ["0", "1", '-1']


def preprocess_test_data(test_file_path):
    df = pd.read_csv(test_file_path)
    df = df['微博中文内容']
    df.to_csv('test.csv', sep='\t', index=False)
    print('the length of test', df.shape)


def preprocess_data(file_path, split_eval=0.05):
    """
        split_eval：验证集切分比例
    """
    df = pd.read_csv(file_path)
    bf = df.loc[df['情感倾向'].isin(LABEL_LIST)]
    # bf.to_csv('train.csv')
    print('df.shape', df.shape)
    print('bf.shape', bf.shape)

    split_index = math.floor(bf.shape[0] * (1 - split_eval))
    bf = bf[['微博中文内容', '情感倾向']]
    bf = bf.sample(frac=1).reset_index(drop=True)
    train_df = bf.loc[:split_index]
    eval_df = bf.loc[split_index:]

    train_df.to_csv('train.csv', sep='\t', index=False)
    eval_df.to_csv('dev.csv', sep='\t', index=False)
    print('the lenght of train_df:', train_df.shape)
    print('the lenght of eval_df:', eval_df.shape)



if __name__ == '__main__':
    preprocess_data('train/nCoV_100k_train.labled.csv')
    preprocess_test_data('test/nCov_10k_test.csv')

# import pandas as pd
# from transformers import *
#
# df = pd.read_csv('train/nCoV_100k_train.labled.csv', encoding='utf-8', engine='python')
#
# print(df.head())
#
# data_dict = dict(df)
# print(data_dict.keys())
#
# print(data_dict.get('微博id'))
#
# print(df['微博id'][0])
#
# # 微博中文内容
# content = df['微博中文内容']
# label = df['情感倾向']
#
# max_length = 0
# for k, i in enumerate(content):
#     print(i)
#     if not type(i)== float:
#         max_length = max(max_length, len(i))
# print('max_length', max_length)
# def preprocess_data(content, label):
#     # tokenizer = BertTokenizer.
#     pass
#
#
# preprocess_data(content, label)
#
# print('-'*10)
# label_set = set()
# for k, i in enumerate(label):
#     if i not in label_set:
#         print(k, i)
#         label_set.add(i)
#     if '·' == i:
#         print(k, i)
#     elif '-'==i:
#         print(k, i)
#
# # print(label)
# # print(set(label))
#
# label_list = list(label_set)
# print(label_list)
