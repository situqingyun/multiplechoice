import torch
from torchblocks.trainer.base import BaseTrainer
from torchblocks.callback import ProgressBar
from torchblocks.utils.tensor import tensor_to_cpu
from torchblocks.callback.adversarial import FreeLB


class MultipleChoiceTrainer(BaseTrainer):

    '''
    文本分类
    '''
    def predict_step(self, model, data_loader, do_eval, **kwargs):
        self.build_record_object()
        pbar = ProgressBar(n_total=len(data_loader), desc='Evaluating' if do_eval else 'Predicting')
        for step, batch in enumerate(data_loader):
            model.eval()
            inputs = self.build_inputs(batch)
            with torch.no_grad():
                outputs = model(**inputs)
            if do_eval:
                loss, logits = outputs[:2]
                loss = loss.mean()
                labels = inputs['labels']
                self.records['target'].append(tensor_to_cpu(labels))
                self.records['loss_meter'].update(loss.item(), n=1)
            else:
                if outputs[0].dim() == 1 and outputs[0].size(0) == 1:
                    logits = outputs[1]
                else:
                    logits = outputs[0]
            self.records['preds'].append(tensor_to_cpu(logits))
            pbar(step)
        self.records['preds'] = torch.cat(self.records['preds'], dim=0)
        if do_eval:
            self.records['target'] = torch.cat(self.records['target'], dim=0)


class FreelbTrainer(MultipleChoiceTrainer):
    def __init__(self, args, metrics, logger, input_keys, collate_fn=None):
        super().__init__(args=args, metrics=metrics, logger=logger,
                         input_keys=input_keys,collate_fn=collate_fn)
        self.adv_model = FreeLB(adv_K=args.adv_K, adv_lr=args.adv_lr,
                                adv_init_mag=args.adv_init_mag,
                                adv_norm_type=args.adv_norm_type,
                                base_model=args.base_model,
                                adv_max_norm=args.adv_max_norm)

    def train_step(self, model, batch, optimizer):
        model.train()
        inputs = self.build_inputs(batch)
        loss = self.adv_model.attack(model, inputs, gradient_accumulation_steps=self.args.gradient_accumulation_steps)
        return loss.item()