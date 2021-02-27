import os
import csv
from torchblocks.metrics.classification import Accuracy
from torchblocks.callback import TrainLogger
from torchblocks.utils import seed_everything, dict_to_text, build_argparse
from torchblocks.utils import prepare_device, get_checkpoints
from model.modeling_nezha import NeZhaForMultipleChoice
from model.configuration_nezha import NeZhaConfig
from transformers import BertForMultipleChoice, BertConfig, BertTokenizer, WEIGHTS_NAME
from torchblocks.processor import TextClassifierProcessor, InputExample
from torchblocks.trainer.classifier_trainer import FreelbTrainer
from multiple_choice_processor import MultipleChoiceProcessor
# from trainer.multiple_choice_trainer import FreelbTrainer

import json


class CommonDataProcessor(MultipleChoiceProcessor):
    # """Base class for processor converters
    #    data_dir: 数据目录
    #    tokenizer: tokenizer
    #    encode_mode: 预处理方式.
    #             ``one``: 表示只有一个inputs
    #             ``pair``: 表示两个inputs，一般针对siamese类型网络
    #             ``triple``: 表示三个inputs，一般针对triple类型网络
    #             (default: ``one``)
    #     add_special_tokens: 是否增加[CLS]XXX[SEP], default: True
    #     pad_to_max_length: 是否padding到最大长度, default: True
    #     truncate_label: 是否label进行阶段，主要在collect_fn函数中，一般针对sequence labeling任务中，default: False
    # """
    #
    # def __init__(self, data_dir, tokenizer,
    #              encode_mode='pair',
    #              add_special_tokens=True,
    #              pad_to_max_length=True,
    #              truncate_label=False,
    #              truncation_strategy="longest_first",
    #              prefix='', **kwargs):
    #
    #     super(MultipleChoiceProcessor, self).__init__(data_dir, tokenizer,
    #                                                   encode_mode,
    #                                                   add_special_tokens,
    #                                                   pad_to_max_length,
    #                                                   truncate_label,
    #                                                   truncation_strategy,
    #                                                   prefix, **kwargs)

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
                        texts=[[question + ' ' + i[2:], context] for i in choice],
                        # texts=[[question + ' ' + i[2:] for i in choice], [context for i in choice]],
                        label=label
                    )
                )

        return examples


MODEL_CLASSES = {
    'nezha': (NeZhaConfig, NeZhaForMultipleChoice, BertTokenizer),
    'bert': (BertConfig, BertForMultipleChoice, BertTokenizer)
}


def main():
    # args = build_argparse().parse_args()
    parser = build_argparse()
    parser.add_argument('--adv_lr', type=float, default=1e-2)
    parser.add_argument('--adv_K', type=int, default=3, help="should be at least 1")
    parser.add_argument('--adv_init_mag', type=float, default=2e-2)
    parser.add_argument('--adv_norm_type', type=str, default="l2", choices=["l2", "linf"])
    parser.add_argument('--adv_max_norm', type=float, default=0, help="set to 0 to be unlimited")
    parser.add_argument('--base_model', default='bert')
    parser.add_argument('--hidden_dropout_prob', type=float, default=0.1)
    parser.add_argument('--attention_probs_dropout_prob', type=float, default=0)
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
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    # data processor
    logger.info("initializing data processor")
    tokenizer = tokenizer_class.from_pretrained(args.model_path, do_lower_case=args.do_lower_case)
    processor = CommonDataProcessor(data_dir=args.data_dir, tokenizer=tokenizer, prefix=prefix)
    label_list = processor.get_labels()
    num_labels = len(label_list)
    args.num_labels = num_labels

    # model
    logger.info("initializing model and config")
    config = config_class.from_pretrained(args.model_path, num_labels=num_labels,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    model = model_class.from_pretrained(args.model_path, config=config)
    model.to(args.device)

    # trainer
    logger.info("initializing traniner")
    trainer = FreelbTrainer(logger=logger, args=args, collate_fn=processor.collate_fn,
                            input_keys=processor.get_input_keys(),
                            metrics=[Accuracy()])
    # do train
    if args.do_train:
        train_dataset = processor.create_dataset(args.train_max_seq_length, 'train.json', 'train')
        # eval_dataset = processor.create_dataset(args.eval_max_seq_length, 'dev.csv', 'dev')
        threshold = 0.1
        train_size = int(threshold * len(train_dataset))
        val_size = len(train_dataset) - train_size
        from torch.utils.data import random_split
        train_dataset, eval_dataset = random_split(train_dataset, [train_size, val_size])
        trainer.train(model, train_dataset=train_dataset, eval_dataset=eval_dataset)
    # do eval
    checkpoint_numbers = list()
    loss_list = list()
    if args.do_eval and args.local_rank in [-1, 0]:
        results = {}
        # eval_dataset = processor.create_dataset(args.eval_max_seq_length, 'dev.csv', 'dev')
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints or args.checkpoint_number > 0:
            checkpoints = get_checkpoints(args.output_dir, args.checkpoint_number, WEIGHTS_NAME)
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("/")[-1].split("-")[-1]
            model = model_class.from_pretrained(checkpoint, config=config)
            model.to(args.device)
            trainer.evaluate(model, eval_dataset, save_preds=True, prefix=str(global_step))
            if global_step:
                result = {"{}_{}".format(global_step, k): v for k, v in trainer.records['result'].items()}
                results.update(result)
            # 筛选出最好的三个
            loss_list.append(trainer.records['result']['eval_loss'])

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        dict_to_text(output_eval_file, results)

        if len(loss_list) > 3:
            sorted_loss_list = loss_list.sort()[:3]
            for i, k in enumerate(loss_list):
                if k in sorted_loss_list:
                    checkpoint_numbers.append(k)
        else:
            checkpoint_numbers = [i for i in range(len(loss_list))]
    # do predict
    if args.do_predict:
        test_dataset = processor.create_dataset(args.eval_max_seq_length, 'test.csv', 'test')
        if args.checkpoint_number != 0:
            checkpoints = get_checkpoints(args.output_dir, args.checkpoint_number, WEIGHTS_NAME)
        else:
            checkpoints = list()
            for i in checkpoint_numbers:
                checkpoints.extend(get_checkpoints(args.output_dir, i, WEIGHTS_NAME))

        for checkpoint in checkpoints:
            global_step = checkpoint.split("/")[-1].split("-")[-1]
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            trainer.predict(model, test_dataset=test_dataset, prefix=str(global_step))


if __name__ == "__main__":
    main()
