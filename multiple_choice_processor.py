import logging
from torchblocks.processor.base import DataProcessor
from torchblocks.processor.utils import InputFeatures

logger = logging.getLogger(__name__)


class MultipleChoiceProcessor(DataProcessor):
    """
        多选题
    """

    def convert_to_features(self, examples, label_list, max_seq_length):
        '''
        文本分类
        '''
        label_map = {label: i for i, label in enumerate(label_list)} if label_list is not None else {}
        features = []
        for (ex_index, example) in enumerate(examples):
            if ex_index % 10000 == 0:
                logger.info("Writing example %d/%d" % (ex_index, len(examples)))
            texts = example.texts
            if not isinstance(texts, list):
                raise ValueError(" texts type: expected one of (list,)")

            # encode_dic = tokenizer.encode_plus(
            #     text=para,
            #     text_pair=qa,
            #     max_length=max_length,
            #     padding='max_length',
            #     truncation=True,
            #     pad_to_multiple_of=True,
            #     add_special_tokens=True,
            #     return_attention_mask=True,
            #     return_token_type_ids=True,
            #     return_tensors='pt'
            # )
            # input_ids = encode_dic['input_ids'].tolist()[0]
            # attention_mask = encode_dic['attention_mask'].tolist()[0]
            # token_type_ids = encode_dic['token_type_ids'].tolist()[0]

            # inputs = self.encode(texts, max_seq_length)

            input_ids = list()
            attention_masks = list()
            token_type_ids = list()

            for count, text in enumerate(texts):
                inputs = self.encode(text, max_seq_length)
                input_id = inputs['input_ids']
                attention_mask = inputs['attention_mask']
                token_type_id = inputs['token_type_ids']
                input_ids.append(input_id)
                attention_masks.append(attention_mask)
                token_type_ids.append(token_type_id)

            count += 1
            # for count, (input_id, attention_mask, token_type_id) in enumerate(inputs):
            #     input_ids.append(input_id)
            #     attention_masks.append(attention_mask)
            #     token_type_ids.append(token_type_id)

            # processed into four options
            for j in range(4 - count):
                input_id = [0] * max_seq_length
                attention_mask = [0] * max_seq_length
                token_type_id = [0] * max_seq_length

                input_ids.append(input_id)
                attention_masks.append(attention_mask)
                token_type_ids.append(token_type_id)

            inputs['input_ids'] = input_ids
            inputs['attention_mask'] = attention_masks
            inputs['token_type_ids'] = token_type_ids

            inputs['guid'] = example.guid
            if example.label_ids is not None:
                label_ids = [0] * len(label_map)  # 多标签分类
                for i, lb in enumerate(example.label_ids):
                    if isinstance(lb, str):
                        label_ids[label_map[lb]] = 1
                    elif isinstance(lb, (float, int)):
                        label_ids[i] = int(lb)
                    else:
                        raise ValueError("multi label type: expected one of (str,float,int)")
                inputs['label_ids'] = label_ids
            if example.label is not None:
                if isinstance(example.label, (float, int)):
                    label = int(example.label)
                elif isinstance(example.label, str):
                    label = label_map[example.label]
                else:
                    raise ValueError("label type: expected one of (str,float,int)")
                inputs['label'] = label
            if ex_index < 5:
                self.print_examples(**inputs)

            features.append(InputFeatures(**inputs))
        return features
