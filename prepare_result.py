import numpy as np
import pandas as pd
import pickle
import json


def prepare_result(predict_result_file, test_file):
    with open(predict_result_file, 'rb') as f:
        predict_data = pickle.load(f)

    with open(test_file, 'r') as f:
        test_data = json.load(f)

    q_ids = []
    for data in test_data:
        for q in data['Questions']:
            q_ids.append(q['Q_id'])

    flat_predictions = np.argmax(predict_data, axis=1).flatten().tolist()

    def convert_id(x):
        if len(str(x)) < 6:
            return '0' * (6 - len(str(x))) + str(x)
        return str(x)

    def convert_label(x):
        res = ['A', 'B', 'C', 'D']
        return res[x]

    sub = pd.DataFrame()
    sub['id'] = q_ids
    sub['label'] = flat_predictions
    sub['label'] = sub['label'].apply(convert_label)

    sub.sort_values('id', inplace=True)
    sub['id'] = sub['id'].apply(convert_id)
    sub.to_csv('outputs/haihua_output/sub.csv', index=False)
    print('Everything Done !!')


if __name__ == '__main__':
    # prepare_result('outputs/haihua_output/longformer-chinese-base-4096/logs/longformer-chinese-base-4096_haihua9716_predict_test_logits.pkl', 'dataset/haihua/validation.json')
    # prepare_result('outputs/haihua_output/dcmn-nezha/nezha-cn-base_haihua13880_predict_test_logits.pkl', 'dataset/haihua/validation.json')
    # prepare_result(
    #     'outputs/haihua_output/nezha-base-www/nezha-cn-base_haihua11104_predict_test_logits.pkl',
    #     'dataset/haihua/validation.json')
    prepare_result('outputs/haihua_output/duma-nezha/nezha-cn-base_haihua15615_predict_test_logits.pkl', 'dataset/haihua/validation.json')

