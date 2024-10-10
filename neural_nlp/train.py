#!/usr/bin/env python
#coding:utf-8
"""
Tencent is pleased to support the open source community by making NeuralClassifier available.
Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
Licensed under the MIT License (the "License"); you may not use this file except in compliance
with the License. You may obtain a copy of the License at
http://opensource.org/licenses/MIT
Unless required by applicable law or agreed to in writing, software distributed under the License
is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
or implied. See the License for thespecific language governing permissions and limitations under
the License.
"""

import os
import sys
import time

import torch
from torch.utils.data import DataLoader

import neural_nlp.util as util
from neural_nlp.util import ModeType
from .config import Config
from .dataset import ClassificationType, AVAILABLE_COLLATORS, AVAILABLE_DATASETS
from .evaluate.classification_evaluate import \
    ClassificationEvaluator as cEvaluator
from .model import AVAILABLE_MODELS
from .model.loss import ClassificationLoss
from .model.model_util import get_optimizer, get_hierar_relations


def get_data_loader(dataset_name, collate_name, conf):
    """Get data loader: Train, Validate, Test
    """
    train_dataset = AVAILABLE_DATASETS[dataset_name](
        conf, conf.data.train_json_files, generate_dict=True)
    collate_fn = AVAILABLE_COLLATORS[collate_name](conf, len(train_dataset.label_map))

    train_data_loader = DataLoader(
        train_dataset, batch_size=conf.train.batch_size, shuffle=True,
        num_workers=conf.data.num_worker, collate_fn=collate_fn,
        pin_memory=True)

    validate_dataset = AVAILABLE_DATASETS[dataset_name](
        conf, conf.data.validate_json_files)
    validate_data_loader = DataLoader(
        validate_dataset, batch_size=conf.eval.batch_size, shuffle=False,
        num_workers=conf.data.num_worker, collate_fn=collate_fn,
        pin_memory=True)

    test_dataset = AVAILABLE_DATASETS[dataset_name](conf, conf.data.test_json_files)
    test_data_loader = DataLoader(
        test_dataset, batch_size=conf.eval.batch_size, shuffle=False,
        num_workers=conf.data.num_worker, collate_fn=collate_fn,
        pin_memory=True)

    return train_data_loader, validate_data_loader, test_data_loader

def get_classification_model(model_name, dataset, conf):
    """Get classification model from configuration
    """
    model = AVAILABLE_MODELS[model_name](dataset, conf)
    model = model.cuda(conf.device) if conf.device.startswith("cuda") else model
    return model

class ClassificationTrainer(object):
    def __init__(self, label_map, logger, evaluator, conf, loss_fn):
        self.label_map = label_map
        self.logger = logger
        self.evaluator = evaluator
        self.conf = conf
        self.loss_fn = loss_fn
        if self.conf.task_info.hierarchical:
            self.hierar_relations = get_hierar_relations(
                    self.conf.task_info.hierar_taxonomy, label_map)

    def train(self, data_loader, model, optimizer, stage, epoch):
        model.update_lr(optimizer, epoch)
        model.train()
        return self.run(data_loader, model, optimizer, stage, epoch,
                        ModeType.TRAIN)

    def eval(self, data_loader, model, optimizer, stage, epoch):
        model.eval()
        return self.run(data_loader, model, optimizer, stage, epoch)

    def run(self, data_loader, model, optimizer, stage,
            epoch, mode=ModeType.EVAL):
        is_multi = False
        # multi-label classifcation
        if self.conf.task_info.label_type == ClassificationType.MULTI_LABEL:
            is_multi = True
        predict_probs = []
        standard_labels = []
        num_batch = data_loader.__len__()
        total_loss = 0.
        for batch in data_loader:
            # hierarchical classification with HMCN
            if self.conf.model_name == "HMCN":
                (global_logits, local_logits, logits) = model(batch)
                loss = self.loss_fn(
                    global_logits,
                    batch[data_loader.dataset.DOC_LABEL].to(self.conf.device),
                    False,
                    is_multi)
                loss += self.loss_fn(
                    local_logits,
                    batch[data_loader.dataset.DOC_LABEL].to(self.conf.device),
                    False,
                    is_multi)
            # hierarchical classification using hierarchy penalty loss
            elif self.conf.task_info.hierarchical:
                logits = model(batch)
                linear_paras = model.linear.weight
                is_hierar = True
                used_argvs = (self.conf.task_info.hierar_penalty, linear_paras, self.hierar_relations)
                loss = self.loss_fn(
                    logits,
                    batch[data_loader.dataset.DOC_LABEL].to(self.conf.device),
                    is_hierar,
                    is_multi,
                    *used_argvs)
            # flat classificaiton
            else:
                logits = model(batch) 
                loss = self.loss_fn(
                    logits,
                    batch[data_loader.dataset.DOC_LABEL].to(self.conf.device),
                    False,
                    is_multi)
            if mode == ModeType.TRAIN:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                continue
            total_loss += loss.item()
            if not is_multi:
                result = torch.nn.functional.softmax(logits, dim=1).cpu().tolist()
            else:
                result = torch.sigmoid(logits).cpu().tolist()
            predict_probs.extend(result)
            standard_labels.extend(batch[data_loader.dataset.DOC_LABEL_LIST])
        if mode == ModeType.EVAL:
            total_loss = total_loss / num_batch
            (_, precision_list, recall_list, fscore_list, right_list,
             predict_list, standard_list) = \
                self.evaluator.evaluate(
                    predict_probs, standard_label_ids=standard_labels, label_map=self.label_map,
                    threshold=self.conf.eval.threshold, top_k=self.conf.eval.top_k,
                    is_flat=self.conf.eval.is_flat, is_multi=is_multi)
            # precision_list[0] save metrics of flat classification
            # precision_list[1:] save metrices of hierarchical classification
            self.logger.warn(
                "%s performance at epoch %d is precision: %f, "
                "recall: %f, fscore: %f, macro-fscore: %f, right: %d, predict: %d, standard: %d.\n"
                "Loss is: %f." % (
                    stage, epoch, precision_list[0][cEvaluator.MICRO_AVERAGE],
                    recall_list[0][cEvaluator.MICRO_AVERAGE],
                    fscore_list[0][cEvaluator.MICRO_AVERAGE],
                    fscore_list[0][cEvaluator.MACRO_AVERAGE],
                    right_list[0][cEvaluator.MICRO_AVERAGE],
                    predict_list[0][cEvaluator.MICRO_AVERAGE],
                        standard_list[0][cEvaluator.MICRO_AVERAGE], total_loss))
            return fscore_list[0][cEvaluator.MICRO_AVERAGE]


def load_checkpoint(file_name, conf, model, optimizer):
    checkpoint = torch.load(file_name)
    conf.train.start_epoch = checkpoint["epoch"]
    best_performance = checkpoint["best_performance"]
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return best_performance


def save_checkpoint(state, file_prefix):
    file_name = file_prefix + "_" + str(state["epoch"])
    torch.save(state, file_name)


def train(conf):
    logger = util.Logger(conf)
    if not os.path.exists(conf.checkpoint_dir):
        os.makedirs(conf.checkpoint_dir)

    model_name = conf.model_name
    dataset_name = "ClassificationDataset"
    collate_name = "FastTextCollator" if model_name == "FastText" \
        else "ClassificationCollator"
    train_data_loader, validate_data_loader, _ = \
        get_data_loader(dataset_name, collate_name, conf)
    empty_dataset = AVAILABLE_DATASETS[dataset_name](conf, [], mode="train")
    model = get_classification_model(model_name, empty_dataset, conf)
    loss_fn = ClassificationLoss(
        label_size=len(empty_dataset.label_map), loss_type=conf.train.loss_type)
    optimizer = get_optimizer(conf, model)
    if conf.optimizer.lr_decay:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode = "max",
            factor=conf.optimizer.lr_decay_rate,
            patience=conf.optimizer.lr_patience,
            verbose=True
        )
    else:
        scheduler = None
    
    evaluator = cEvaluator(conf.eval.dir)
    trainer = ClassificationTrainer(
        empty_dataset.label_map, logger, evaluator, conf, loss_fn)

    best_performance = 0
    checkpoint_file = conf.checkpoint_dir + "/" + model_name + "_best"
    wait = 0

    for epoch in range(conf.train.start_epoch,
                       conf.train.start_epoch + conf.train.num_epochs):
        start_time = time.time()
        trainer.train(train_data_loader, model, optimizer, "Train", epoch)
        trainer.eval(train_data_loader, model, optimizer, "Train", epoch)
        performance = trainer.eval(
            validate_data_loader, model, optimizer, "Validate", epoch)

        if performance > best_performance:  # record the best model
            best_performance = performance
            torch.save({
                'epoch': epoch,
                'model_name': model_name,
                'state_dict': model.state_dict(),
                'best_performance': best_performance,
                'optimizer': optimizer.state_dict(),
            }, checkpoint_file)

            wait = 0
        else:
            wait += 1
        
        if scheduler is not None:
            old_rate = scheduler.get_last_lr()
            scheduler.step(performance)
            new_rate = scheduler.get_last_lr()
            if old_rate != new_rate:
                logger.info(f"Epoch {epoch}: adjusting LR {old_rate:.4e} -> {new_rate:.4e}")

        time_used = time.time() - start_time
        logger.info("Epoch %d cost time: %d second" % (epoch, time_used))

        if wait == conf.train.early_stopping:
            logger.warn(f"Early stopping triggered after {wait} epochs of no improvement")
            break

if __name__ == '__main__':
    config = Config(config_file=sys.argv[1])
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.train.visible_device_list)
    torch.manual_seed(2019)
    torch.cuda.manual_seed(2019)
    train(config)
