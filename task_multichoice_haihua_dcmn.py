import os
from torchblocks.metrics.classification import Accuracy
from torchblocks.callback import TrainLogger
from torchblocks.utils import seed_everything, dict_to_text, build_argparse
from torchblocks.utils import prepare_device, get_checkpoints
from model.modeling_nezha import NeZhaForMultipleChoice, NeZhaForMultipleChoiceWithMatch
from model.configuration_nezha import NeZhaConfig
from transformers import BertForMultipleChoice, BertConfig, BertTokenizer, WEIGHTS_NAME
from transformers import XLNetConfig, XLNetForMultipleChoice, XLNetTokenizer
from torchblocks.processor import InputExample
from torchblocks.trainer.classifier_trainer import TextClassifierTrainer
from processor.multiple_choice_processor import DCMNMultipleChoiceProcessor
from torch.utils.data import random_split
from model.modeling_bert import BertForMultipleChoiceWithMatch

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


MODEL_CLASSES = {
    'nezha': (NeZhaConfig, NeZhaForMultipleChoiceWithMatch, BertTokenizer),
    'bert': (BertConfig, BertForMultipleChoiceWithMatch, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForMultipleChoice, XLNetTokenizer)
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

    # 中文存在问题
    parser.add_argument("--do_debug", action="store_true", help="the do_debug only uses the first 10")

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
    # AutoModel.from_pretrained('bert-base-uncased', mirror='tuna')
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
    trainer = TextClassifierTrainer(logger=logger, args=args, collate_fn=processor.collate_fn,
                                    input_keys=processor.get_input_keys(),
                                    metrics=[Accuracy()])
    # trainer = AlumTrainer(logger=logger, args=args, collate_fn=processor.collate_fn,
    #                         input_keys=processor.get_input_keys(),
    #                         metrics=[Accuracy()])

    # do train
    if args.do_train:
        train_dataset = processor.create_dataset(args.train_max_seq_length, 'train.json', 'train')
        # eval_dataset = processor.create_dataset(args.eval_max_seq_length, 'dev.csv', 'dev')

        threshold = 0.9
        train_size = int(threshold * len(train_dataset))
        val_size = len(train_dataset) - train_size

        train_dataset, eval_dataset = random_split(train_dataset, [train_size, val_size])

        if args.do_debug:
            train_dataset, _ = random_split(train_dataset, [2, len(train_dataset) - 2])
            eval_dataset, _ = random_split(eval_dataset, [2, len(eval_dataset) - 2])

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
            print('eval_acc: ', checkpoint, trainer.records['result']['eval_acc'])
            loss_list.append(trainer.records['result']['eval_acc'])

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        dict_to_text(output_eval_file, results)

        if len(loss_list) > 1:
            import copy
            sorted_loss_list = copy.deepcopy(loss_list)
            sorted_loss_list.sort(reverse=True)
            sorted_loss_list = sorted_loss_list[:1]
            for i, k in enumerate(loss_list):
                if k in sorted_loss_list:
                    print('-' * 20)
                    print('best one:', checkpoints[i])
                    print('-' * 20)
                    checkpoint_numbers.append(checkpoints[i])
        else:
            checkpoint_numbers = [i for i in checkpoints]
    # do predict
    if args.do_predict:
        test_dataset = processor.create_dataset(args.eval_max_seq_length, 'validation.json', 'test')

        if args.do_debug:
            test_dataset, _ = random_split(test_dataset, [2, len(test_dataset) - 2])

        if args.checkpoint_number != 0:
            checkpoints = get_checkpoints(args.output_dir, args.checkpoint_number, WEIGHTS_NAME)
        else:
            checkpoints = checkpoint_numbers
            # for i in checkpoint_numbers:
            #     checkpoints.extend(get_checkpoints(args.output_dir, i, WEIGHTS_NAME))

        for checkpoint in checkpoints:
            global_step = checkpoint.split("/")[-1].split("-")[-1]
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            trainer.predict(model, test_dataset=test_dataset, prefix=str(global_step))


if __name__ == "__main__":
    main()
