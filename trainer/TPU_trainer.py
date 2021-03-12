import os
import torch
from argparse import Namespace

from torch.utils.data.dataloader import DataLoader, default_collate
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler

from torchblocks.optims import AdamW
from torchblocks.optims.lr_scheduler import get_linear_schedule_with_warmup

from torchblocks.utils.paths import save_pickle, json_to_text
from torchblocks.utils.tools import seed_everything, AverageMeter, to_json_string
from torchblocks.callback import ModelCheckpoint, EarlyStopping, ProgressBar, TrainLogger, EMA
from torchblocks.utils.tensor import tensor_to_cpu

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl


def build_train_dataloader(self, train_dataset):
    '''
    Load train dataset
    '''
    if train_dataset is None:
        raise ValueError("Trainer: training requires an train_dataset.")
    batch_size = self.args.per_gpu_train_batch_size * max(1, self.args.n_gpu)
    # sampler = RandomSampler(train_dataset) if self.args.local_rank == -1 else DistributedSampler(train_dataset)
    sampler = DistributedSampler(
        train_dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=True
    )
    data_loader = DataLoader(train_dataset, sampler=sampler, batch_size=batch_size, collate_fn=self.collate_fn,
                             num_workers=self.args.num_workers)
    return data_loader


def build_eval_dataloader(self, eval_dataset):
    '''
    Load eval dataset
    '''
    if eval_dataset is None:
        raise ValueError("Trainer: evaluation requires an eval_dataset.")
    batch_size = self.args.per_gpu_eval_batch_size * max(1, self.args.n_gpu)
    # sampler = SequentialSampler(eval_dataset) if self.args.local_rank == -1 else DistributedSampler(eval_dataset)
    sampler = DistributedSampler(
        eval_dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=False
    )
    data_loader = DataLoader(eval_dataset, sampler=sampler, batch_size=batch_size, collate_fn=self.collate_fn,
                             num_workers=self.args.num_workers)
    return data_loader


def build_test_dataloader(self, test_dataset):
    '''
    Load test dataset
    '''
    if test_dataset is None:
        raise ValueError("Trainer: evaluation requires an test_dataset.")
    batch_size = self.args.per_gpu_eval_batch_size * max(1, self.args.n_gpu)
    # sampler = SequentialSampler(test_dataset) if self.args.local_rank == -1 else DistributedSampler(test_dataset)
    sampler = DistributedSampler(
        test_dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=False
    )
    data_loader = DataLoader(test_dataset, sampler=sampler, batch_size=batch_size, collate_fn=self.collate_fn,
                             num_workers=self.args.num_workers)
    return data_loader


def train(self, model, train_dataset, eval_dataset):
    """
    Main training entry point.
    """
    train_dataloader = self.build_train_dataloader(train_dataset)
    t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs
    optimizer = self.build_optimizer(model)
    scheduler = self.build_lr_scheduler(optimizer, t_total)
    optimizer, scheduler = self.restore_optimizer(optimizer, scheduler)
    model, optimizer = self.build_apex_and_distribute(model, optimizer)
    # Train!
    self.print_training_parameters(model, len(train_dataset), t_total)
    model.zero_grad()
    # ema
    if self.args.do_ema:
        ema = EMA(model, decay=self.args.ema_decay)
    seed_everything(self.args.seed)  # Added here for reproductibility (even between python 2 and 3)
    print('Start training.')
    if self.args.logging_steps < 0:
        self.args.logging_steps = len(train_dataloader)
    if self.args.save_steps < 0:
        self.args.save_steps = len(train_dataloader)
    for epoch in range(0, int(self.args.num_train_epochs)):
        self.build_record_object()
        train_dataloader = pl.ParallelLoader(train_dataloader, [self.args.device]).per_device_loader(self.args.device)
        pbar = ProgressBar(n_total=len(train_dataloader), desc='Training')
        for step, batch in enumerate(train_dataloader):
            loss = self.train_step(model, batch, optimizer)
            xm.optimizer_step(optimizer)
            if (step + 1) % self.args.gradient_accumulation_steps == 0:
                self.train_update(model, optimizer, loss, scheduler)
                if self.args.do_ema:
                    ema.update(model)
                pbar(step, {'loss': loss})
            if (self.args.local_rank in [-1, 0]
                    and self.args.logging_steps > 0
                    and self.global_step % self.args.logging_steps == 0
            ):
                if self.args.do_ema:
                    ema.apply_shadow(model)
                self.tb_writer.add_scalar('Loss/train_epoch_loss', self.records['loss_meter'].avg,
                                          int(self.global_step / self.args.logging_steps))
                self.evaluate(model, eval_dataset)
                if self.args.do_ema:
                    ema.restore(model)
                if hasattr(self.tb_writer, 'save'):
                    self.tb_writer.save()
            if (self.args.local_rank in [-1, 0]
                    and self.args.save_steps > 0
                    and self.global_step % self.args.save_steps == 0
            ):
                # model checkpoint
                if self.model_checkpoint:
                    state = self.build_state_object(model, optimizer, scheduler, self.global_step)
                    self.model_checkpoint.step(
                        state=state,
                        current=self.records['result'][self.model_checkpoint.monitor]
                    )
        if not self.scheduler_on_batch:  # epoch scheduler
            scheduler.step()
        # early_stopping
        if self.early_stopping:
            self.early_stopping.step(current=self.records['result'][self.early_stopping.monitor])
            if self.early_stopping.stop_training:
                break
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if self.tb_writer:
        self.tb_writer.close()



def predict_step(self, model, data_loader, do_eval, **kwargs):
    self.build_record_object()
    data_loader = pl.ParallelLoader(data_loader, [self.args.device]).per_device_loader(self.args.device)
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
