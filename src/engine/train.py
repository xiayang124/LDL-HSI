from typing import Optional
from engine import setting_model
from engine.config import Config
import engine.datasets as datasets
import logger.log as log
from engine.evals import evaluation

import torch
import torch.nn as nn
from torch.amp import autocast
import utils
import numpy as np

        
class Trainer:
    def __init__(self, cfg: Config, recorder: Optional[log.Logger] = None):
        self.cfg = cfg
        self.recorder = recorder

        # Placeholders
        self.model = None
        self.optimizer = None
        
        self.train_dataset = None
        self.test_dataset = None
        self.all_dataset = None
        
        self.loss = nn.CrossEntropyLoss()
        self.device = self.cfg.TrainingSetting.get('device', default='cpu')
        self.amp = self.cfg.TrainingSetting.get('amp', default=False)
        self.epochs = self.cfg.TrainingSetting.get('epochs', default=100)
        self.checkpoint = self.cfg.TrainingSetting.get('checkpoint', default=10)
        self.pca = self.cfg.TrainingSetting.get('pca', default=self.cfg.BaseSetting.get('bands', False))
        self.scaler = torch.amp.grad_scaler() if self.amp else None
        self.multi_args = self.cfg.TestSetting.get('multi_args', default=False) or self.cfg.TestSetting.get('ablation', default=False)
        self.class_num = self.cfg.BaseSetting.get('num_classes', default=16)
        
        self.best_oa = 0
        self.best_data = None
        self.best_per_classes = None
        
    def setup(self):
        self.train_dataset, self.test_dataset, self.all_dataset, classes_name, \
            self.data_shape, self.gt_mask, self.HSI_orig = datasets.data_preprocess(self.cfg)
        self.model, self.optimizer = setting_model.choose_model(self.cfg)
    
    def train_epoch(self):
        self.model.train()
        
        epoch_loss = 0.0
        total = 0
        
        all_target, all_pred = torch.tensor([]).to(self.device), torch.tensor([]).to(self.device)
        for idx, (batch_data, batch_label) in enumerate(self.train_dataset):
            if batch_label.shape[0] == 1:
                continue
            batch_data, batch_label = batch_data.to(self.device), (batch_label - 1).to(self.device)
            self.optimizer.zero_grad()
            if self.amp:
                with autocast():
                    outputs = self.model(batch_data)
                    loss = self.loss(outputs, batch_label.long())
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(batch_data)
                loss = self.loss(outputs, batch_label.long())
                loss.backward()
                self.optimizer.step()
            batch_size = batch_data.size(0)
            epoch_loss += loss.item() * batch_size
            total += batch_size
            all_target = torch.cat((all_target, batch_label), dim=0)
            all_pred = torch.cat((all_pred, outputs.argmax(dim=1)), dim=0)
        avg_loss = epoch_loss / total
        return all_target, all_pred, avg_loss
    
    def test_epoch(self):
        self.model.eval()
        
        all_target, all_pred = torch.tensor([]).to(self.device), torch.tensor([]).to(self.device)
        with torch.no_grad():
            for idx, (batch_data, batch_label) in enumerate(self.test_dataset):
                if batch_label.shape[0] == 1:
                    continue
                batch_data, batch_label = batch_data.to(self.device), (batch_label - 1).to(self.device)
                if self.amp:
                    with autocast():
                        outputs = self.model(batch_data)
                else:
                    outputs = self.model(batch_data)
                all_target = torch.cat((all_target, batch_label), dim=0)
                all_pred = torch.cat((all_pred, outputs.argmax(dim=1)), dim=0)
        return all_target, all_pred
    
    def all_epoch(self):
        self.model.eval()
        
        all_target, all_pred = torch.tensor([]).to(self.device), torch.tensor([]).to(self.device)
        with torch.no_grad():
            for idx, (batch_data, batch_label) in enumerate(self.all_dataset):
                if batch_label.shape[0] == 1:
                    continue
                batch_data, batch_label = batch_data.to(self.device), (batch_label - 1).to(self.device)
                if self.amp:
                    with autocast():
                        outputs = self.model(batch_data)
                else:
                    outputs = self.model(batch_data)
                all_target = torch.cat((all_target, batch_label), dim=0)
                all_pred = torch.cat((all_pred, outputs.argmax(dim=1)), dim=0)
        return all_target, all_pred

    def save_checkpoint(self, save_path):
        torch.save(self.model.state_dict(), save_path)
        return save_path

    def train(self):
        self.setup()
        for epoch in range(self.epochs):
            train_targ, train_pred, loss = self.train_epoch()
            train_eval = evaluation(train_targ, train_pred, label_value=list(range(self.class_num)))
            oa = train_eval.get_oa()
            print(f"Epoch: {epoch}, Overall Acc: {oa:.2f}, Loss: {loss:.4f}")

            if (epoch + 1) % self.checkpoint == 0:
                test_targ, test_pred = self.test_epoch()
                test_eval = evaluation(test_targ, test_pred, label_value=list(range(self.class_num)))
                oa, aa, kappa, per_classes = test_eval.get_oa(), test_eval.get_aa(), test_eval.get_kappa(), test_eval.get_per_class_accuracy()
                test_msg = f"Testing, Overall Acc: {oa:.2f}, Avg Acc: {aa: 2f}, Kappa: {kappa: 2f}\n"
                print(test_msg)

                if oa > self.best_oa:
                    self.best_oa = oa
                    utils.ensure_dir("./model_save/")
                    self.save_checkpoint(save_path=self.cfg.TrainingSetting.get('model_path', default=f"./model_save/{self.dataset_name}_best_model.pth"))
                    self.recorder.INFO_log(f"New best checkpoint saved: OA: {self.best_oa:.3f}, AA: {aa:.3f}, Kappa: {kappa:.4f} at epoch {epoch + 1}\n")
                    self.best_data = np.array([oa, aa, kappa])
                    self.best_per_classes = per_classes
        self.recorder.INFO_log("Training Finished.\n")
        return self.best_data, self.best_per_classes
    
class LDLHSITrainer(Trainer):
    def __init__(self, cfg: Config, recorder: Optional[log.Logger] = None):
        super().__init__(cfg, recorder)

    def setup(self):
        # Load datasets and preprocess
        self.train_dataset, self.test_dataset, self.all_dataset, _, \
            self.data_shape, self.gt_mask, self.HSI_orig, text_token = datasets.data_preprocess(self.cfg)
        # CLIP text encoder embedding dimension
        self.cfg.TrainingSetting.update({"embed_dim": 768})
        self.text_token = text_token.to(self.device)
        self.loss_fusion = torch.tensor(self.cfg.TrainingSetting.get('loss_fusion', 0.2)).to(self.device)
        self.dataset_name = self.cfg.BaseSetting.get('dataset', default='Houston')
        self.model, self.optimizer = setting_model.choose_model(self.cfg)
        self.cls_loss = nn.CrossEntropyLoss()
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        all_target, all_pred = torch.tensor([], dtype=torch.long, device=self.device), torch.tensor([], dtype=torch.long, device=self.device)

        for batch_data, batch_label in self.train_dataset:
            if batch_label.shape[0] == 1: continue

            batch_data, batch_label = batch_data.to(self.device), (batch_label - 1).to(self.device).long()
            cos_sim_label = torch.arange(self.class_num, device=self.device, dtype=torch.long)
            logits, cos_logit = self.model(batch_data, self.text_token, labels=batch_label)
            cls_loss, cos_loss = self.cls_loss(logits, batch_label), self.cls_loss(cos_logit, cos_sim_label)
            loss = (1 - self.loss_fusion) * cls_loss + self.loss_fusion * cos_loss
            loss.backward(); self.optimizer.step(); self.optimizer.zero_grad()
            
            total_loss += loss.item()
            all_target, all_pred = torch.cat((all_target, batch_label), dim=0), torch.cat((all_pred, logits.argmax(dim=1)), dim=0)
        return all_target, all_pred, total_loss

    @torch.no_grad()
    def test_epoch(self):
        self.model.eval()
        all_target, all_pred = torch.tensor([], dtype=torch.long, device=self.device), torch.tensor([], dtype=torch.long, device=self.device)

        for batch_data, batch_label in self.test_dataset:
            if batch_label.shape[0] == 1: continue
            batch_data, batch_label = batch_data.to(self.device), (batch_label - 1).to(self.device).long()

            logits, _ = self.model(batch_data, self.text_token, labels=None)
            all_target, all_pred = torch.cat((all_target, batch_label), dim=0), torch.cat((all_pred, logits.argmax(dim=1)), dim=0)
        return all_target, all_pred
    
    @torch.no_grad()
    def all_epoch(self):
        self.model.eval()
        all_target, all_pred = torch.tensor([], dtype=torch.long, device=self.device), torch.tensor([], dtype=torch.long, device=self.device)

        for batch_data, batch_label in self.all_dataset:
            if batch_label.shape[0] == 1: continue
            batch_data, batch_label = batch_data.to(self.device), (batch_label - 1).to(self.device).long()

            logits, _ = self.model(batch_data, self.text_token, labels=None)
            all_target, all_pred = torch.cat((all_target, batch_label), dim=0), torch.cat((all_pred, logits.argmax(dim=1)), dim=0)
        return all_target, all_pred
    

def choose_trainer(cfg: Config, recorder: Optional[log.Logger] = None):
    model_name = cfg.TrainingSetting.get('model')
    if model_name == "LDLHSI":
        return LDLHSITrainer(cfg, recorder)
    return Trainer(cfg, recorder)
