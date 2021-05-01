import os
from torchblocks.metrics.classification import Accuracy
from torchblocks.callback import TrainLogger
from torchblocks.utils import seed_everything, dict_to_text, build_argparse
from torchblocks.utils import prepare_device, get_checkpoints
from model.modeling_nezha import NeZhaForMultipleChoice, NeZhaForMultipleChoiceWithMatch, NeZhaForMultipleChoiceWithDUMA, NeZhaForMultipleChoiceWithDUMAZuiJia, NeZhaForMultipleChoiceWithDUMA256
from model.configuration_nezha import NeZhaConfig
from transformers import BertForMultipleChoice, BertConfig, BertTokenizer, WEIGHTS_NAME
from transformers import XLNetConfig, XLNetForMultipleChoice, XLNetTokenizer
from torchblocks.processor import InputExample
from torchblocks.trainer.classifier_trainer import TextClassifierTrainer
from processor.multiple_choice_processor import DCMNMultipleChoiceProcessor
from torch.utils.data import random_split
from model.modeling_bert import BertForMultipleChoiceWithDUMA
# from model.modeling_roberta import RobertaForMultipleChoiceWithDUMA
from model.modeling_xlnet import XLNetForMultipleChoiceWithDUMA
from transformers import RobertaTokenizer, RobertaConfig
from transformers import ElectraConfig, ElectraTokenizer
from model.modeling_electra import ElectraForMultipleChoiceDUMA
import torch

import json


class CommonDataProcessor(DCMNMultipleChoiceProcessor):

    def get_labels(self):
        """See base class."""
        return ["A", "B", 'C', "D"]

    def read_data(self, input_file):
        """Reads a json list file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            # reader = csv.reader(f, delimiter="\t", quotechar=None)
            # lines = []
            # for line in reader:
            #     lines.append(line)
            # return lines
            data_list = json.load(f)
            return data_list

    def create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            Id = line['ID']
            context = line['Content']
            for qa in line['Questions']:
                Id = qa['Q_id']
                # answer = qa['Answer']
                question = qa['Question']
                choice = qa['Choices']

                if 'test' != set_type:
                    label = qa['Answer']
                else:
                    label = None

                examples.append(
                    InputExample(
                        guid=int(Id),
                        texts=[[question, i[2:], context] for i in choice],
                        # texts=[[question + ' ' + i[2:], context] for i in choice],
                        label=label
                    )
                )

        return examples


# MODEL_CLASSES = {
#     'nezha': (NeZhaConfig, NeZhaForMultipleChoiceWithDUMA, BertTokenizer),
#     'bert': (BertConfig, BertForMultipleChoiceWithDUMA, BertTokenizer),
#     'roberta': (RobertaConfig, RobertaForMultipleChoiceWithDUMA, BertTokenizer),
#     "electra": (ElectraConfig, ElectraForMultipleChoiceDUMA, ElectraTokenizer)
# }

models_detail = {
    'dcmn': (NeZhaConfig, NeZhaForMultipleChoiceWithMatch, BertTokenizer, 256),
    'duma_zuijia':(NeZhaConfig, NeZhaForMultipleChoiceWithDUMAZuiJia, BertTokenizer, 512),
    'duma_256':(NeZhaConfig, NeZhaForMultipleChoiceWithDUMA256, BertTokenizer, 256),
    'roberta_duma':(BertConfig, BertForMultipleChoiceWithDUMA, BertTokenizer, 256),
}

def main():
    # args = build_argparse().parse_args()
    parser = build_argparse()
    parser.add_argument('--adv_lr', type=float, default=1e-3)
    parser.add_argument('--adv_K', type=int, default=1)
    parser.add_argument('--adv_alpha', default=1.0, type=float)
    parser.add_argument('--adv_var', default=1e-5, type=float)
    parser.add_argument('--adv_gamma', default=1e-6, type=float)
    parser.add_argument('--adv_norm_type', type=str, default="inf", choices=["l2", 'l1', "inf"])
    parser.add_argument('--hidden_dropout_prob', type=float, default=0.1)
    parser.add_argument('--attention_probs_dropout_prob', type=float, default=0)

    parser.add_argument("--do_debug", action="store_true", help="the do_debug only uses the first 10")
    parser.add_argument("--do_fusion", action="store_true", default=False, help="concencate all hidden states")
    # 保存guid
    parser.add_argument("--save_guid", action="store_true", default=False, help="save guid")
    # 加载指定数列切分
    parser.add_argument("--load_guid", action="store_true", default=False, help="load guid")

    args = parser.parse_args()
    if args.model_path is None:
        args.model_path = args.model_name
    if args.model_name is None:
        args.model_name = args.model_path.split("/")[-1]
    args.output_dir = args.output_dir + '{}'.format(args.model_name)
    os.makedirs(args.output_dir, exist_ok=True)

    # output dir
    prefix = "_".join([args.model_name, args.task_name])
    logger = TrainLogger(log_dir=args.output_dir, prefix=prefix)

    # device
    logger.info("initializing device")
    args.device, args.n_gpu = prepare_device(args.gpu, args.local_rank)
    seed_everything(args.seed)
    # args.model_type = args.model_type.lower()
    # config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    # data processor
    logger.info("initializing data processor")
    # AutoModel.from_pretrained('bert-base-uncased', mirror='tuna')
    # tokenizer = tokenizer_class.from_pretrained(args.model_path, do_lower_case=args.do_lower_case)
    # processor = CommonDataProcessor(data_dir=args.data_dir, tokenizer=tokenizer, prefix=prefix)
    # label_list = processor.get_labels()
    # num_labels = len(label_list)
    # args.num_labels = num_labels

    model_paths = {
        'nezha': args.model_path,
        'roberta': 'hfl/chinese-roberta-wwm-ext',
    }

    model_trainers=dict()

    for k, v in model_paths.items():
        logger.info("initializing model and config")

        config_class, model_class, tokenizer_class, tokenize_len = models_detail[k]

        # config = config_class.from_pretrained(args.model_path, num_labels=num_labels,
        #                                       cache_dir=args.cache_dir if args.cache_dir else None)

        tokenizer = tokenizer_class.from_pretrained(args.model_path, do_lower_case=args.do_lower_case)
        processor = CommonDataProcessor(data_dir=args.data_dir, tokenizer=tokenizer, prefix=prefix)
        label_list = processor.get_labels()
        num_labels = len(label_list)
        args.num_labels = num_labels

        # trainer
        logger.info("initializing traniner")
        trainer = TextClassifierTrainer(logger=logger, args=args, collate_fn=processor.collate_fn,
                                        input_keys=processor.get_input_keys(),
                                        metrics=[Accuracy()])

        model_trainers[k] = [processor, trainer]

    # do predict
    if args.do_predict:

        model_checkpoints = {
            'dcmn':'/content/drive/MyDrive/multi-choice/duma/duma/test/model/dcmn/checkpoint-34700',
            'duma_zuijia': '/content/drive/MyDrive/multi-choice/duma/duma/test/model/zuijia_512/checkpoint-74016',
            'duma_256': '/content/drive/MyDrive/multi-choice/duma/duma/test/model/duma_2_diff_256/checkpoint-32965',
            'roberta_duma': '/content/drive/MyDrive/multi-choice/duma/duma/test/model/robert_256_hfl_duma_bert/checkpoint-31230'
        }

        i=0
        for key, checkpoint in model_checkpoints.items():
            config_class, model_class, tokenizer_class, tokenize_len = models_detail[key]
            tokenizer = tokenizer_class.from_pretrained(args.model_path, do_lower_case=args.do_lower_case)
            processor = CommonDataProcessor(data_dir=args.data_dir, tokenizer=tokenizer, prefix=prefix)
            test_dataset = processor.create_dataset(tokenize_len, 'validation.json', 'test')
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            model.load_state_dict() #todo
            trainer.predict(model, test_dataset=test_dataset, prefix=str(i))
            if key=='duma_zuijia':
                for j in range(5):
                    data_file = 'nezha-cn-base_fold_{}.pt'.format(j)
                    model.load_state_dict(torch.load(data_file))
                    trainer.predict(model, test_dataset=test_dataset, prefix=str(i))
                    i+=1
            i+=1

        # test_dataset = processor.create_dataset(args.eval_max_seq_length, 'validation.json', 'test')
        #
        # if args.do_debug:
        #     test_dataset, _ = random_split(test_dataset, [2, len(test_dataset) - 2])    # , generator=torch.Generator().manual_seed(args.seed)
        #
        # if args.checkpoint_number != 0:
        #     checkpoints = get_checkpoints(args.output_dir, args.checkpoint_number, WEIGHTS_NAME)
        #
        # for checkpoint in checkpoints:
        #     global_step = checkpoint.split("/")[-1].split("-")[-1]
        #     model = model_class.from_pretrained(checkpoint)
        #     model.to(args.device)
        #     trainer.predict(model, test_dataset=test_dataset, prefix=str(global_step))


if __name__ == "__main__":
    main()
