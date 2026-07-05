import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model.SVAFormer import SVAFormer
from extract_text.text_process import load_text_encoder


class LDLHSI(nn.Module):
    def __init__(self, args):
        super().__init__()
        dim = args.TrainingSetting.get("dim")              # feature dim from SVAFormer
        num_classes = int(args.BaseSetting.get("num_classes"))
        clip_dim = args.TrainingSetting.get("clip_dim")
        dropout = args.TrainingSetting.get("dropout", 0.1)
        device = args.TrainingSetting.get("device")
        pt_loc = args.TrainingSetting.get("pretrain_loc")
        slim_enhance_ratio = args.TrainingSetting.get("se_ratio", 0.05)
        prototype_momentum = args.TrainingSetting.get("proto_momentum", 0.5)

        self.args = args
        self.num_classes = num_classes
        self.feature_dim = dim
        self.device = device
        self.slim_enhance_ratio = slim_enhance_ratio
        self.prototype_momentum = prototype_momentum
        
        # --------------------- Regist Buffer --------------------- #
        self.register_buffer("cached_text_features", torch.empty(size=(0, clip_dim), device=device))
        self.register_buffer("prototype_image_feature", torch.zeros(size=(num_classes, dim), device=device))

        # --------------------- Image encoder --------------------- #
        self.image_encoder = SVAFormer(args.SVASetting)
        self.image_slim_projection = nn.Linear(dim, dim, bias=False)
        self.prototype_image_projection = nn.Linear(dim, dim, bias=False)

        # --------------------- Text projection ------------------- #
        # project CLIP text embedding (clip_dim) to image feature space (dim)
        self.text_encoder, _ = load_text_encoder(pt_loc)
        self.text_encoder.eval()
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        self.text_projection = nn.Linear(clip_dim, dim, bias=False)

        # --------------------- Classifier ------------------------ #
        hidden_dim = dim * 2
        self.classifier = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes),
        )
        
        self.proto_clip_temperature = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.diff_clip_temperature = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
    def EMA_update(self, image_feature, labels, momentum=0.9):
        classes = labels.unique().long()
        mask = labels.unsqueeze(1) == classes.unsqueeze(0)  # [B, K]
        count = mask.sum(dim=0).clamp(min=1).unsqueeze(1)
        
        proto_sum = mask.float().T @ image_feature
        proto_c = proto_sum / count

        proto_all = self.prototype_image_feature.clone()
        updated_proto = momentum * proto_all[classes] + (1 - momentum) * proto_c
        proto_all[classes] = updated_proto

        # Persist detached copy so next batch does not keep autograd history
        self.prototype_image_feature[classes] = updated_proto.detach()
        return proto_all
    
    # Sample-level diff feature enhancement
    def slim_feature_enhance(self, image_features, text_features, labels, prototype_features, label_mask=None):
        classes = labels.unique().long()        # [K]
        mask = labels.unsqueeze(1) == classes.unsqueeze(0)   # [B,K]
        if label_mask is not None:
            mask = mask & label_mask[:, None]
        proto = prototype_features[classes]     # [K,D]
        # [B,K,D]
        diff_all = image_features.unsqueeze(1) - proto.unsqueeze(0)
        diff = (diff_all * mask.unsqueeze(-1)).sum(dim=1)  # [B,D]

        projection_diff = self.image_slim_projection(diff)  # [B,D]
        projection_diff = F.normalize(projection_diff, dim=-1)
        align_text_features = F.normalize(text_features, dim=-1)
        
        diff_sim = torch.exp(self.diff_clip_temperature) * self.cos_sim(align_text_features, projection_diff)  # [C,B]
        diff_enhanced_features = torch.softmax(diff_sim.t(), dim=-1) @ text_features  # [B,D]
        return image_features + diff_enhanced_features * self.slim_enhance_ratio
    
    # Prototype-level alignment
    def cls_level_alignment(self, image_features, text_features, labels):
        updated_proto = self.EMA_update(image_features, labels, self.prototype_momentum)
        prototype_image_feature = self.prototype_image_projection(updated_proto)  # [C, D]
        proto_sim = torch.exp(self.proto_clip_temperature) * self.cos_sim(text_features, prototype_image_feature)  # [C, C]
        return proto_sim, updated_proto
    
    def feature_extractor(self, x, text_token):
        image_features, _ = self.image_encoder(x)  # [B, D]
        if self.cached_text_features.numel() == 0:
            _, text_features = self.text_encoder(text_token)
            text_features = self.text_projection(text_features)
            self.cached_text_features = text_features.detach()
        else:
            text_features = self.cached_text_features
        return image_features, text_features
            
    def forward(self, x, text_token, labels=None):
        image_features, text_features = self.feature_extractor(x, text_token)
        if labels is not None:
            proto_sim, prototype_features = self.cls_level_alignment(image_features, text_features, labels)
            mask = None
        else:
            prototype_features = self.prototype_image_feature
            proto_sim = F.normalize(self.cos_sim(self.prototype_image_projection(image_features), prototype_features), dim=-1)  # [B, C]
            labels = torch.argmax(proto_sim, dim=1)
            mask = (proto_sim.gather(1, labels[:, None]).squeeze(1) >= 0.6) # [B]
        image_enhancement_features = self.slim_feature_enhance(image_features, text_features, labels, prototype_features, label_mask=mask)
        logits = self.classifier(image_enhancement_features)  # [B, num_classes]
        return logits, proto_sim
        
    def cos_sim(self, x1, x2):
        x1 = F.normalize(x1, dim=-1)
        x2 = F.normalize(x2, dim=-1)
        return x1 @ x2.t()
    