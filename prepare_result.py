import numpy as np
import pandas as pd
import pickle
import json


def prepare_result(predict_result_file, test_file):
    with open(predict_result_file, 'r') as f:
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
    sub.to_csv('/content/drive/MyDrive/drive/haihua/output/sub.csv', index=False)
    print('Everything Done !!')


if __name__ == '__main__':
    prepare_result('', 'dataset/haihua/validation.json')

