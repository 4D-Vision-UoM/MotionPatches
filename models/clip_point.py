"""
Copyright 2024 LY Corporation
LY Corporation licenses this file to you under the CC BY-NC 4.0
(the "License"); you may not use this file except in compliance
with the License. You may obtain a copy of the License at:
    https://creativecommons.org/licenses/by-nc/4.0/
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
License for the specific language governing permissions and limitations
under the License.
"""

import numpy as np
import timm
import torch
import transformers
from torch import nn
from omegaconf import DictConfig
from models.pointnet import PointnetTransformer


class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim: int, projection_dim: int, dropout: float) -> None:
        super().__init__()

        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x += projected
        return self.layer_norm(x)


class TextEncoder(nn.Module):
    def __init__(self, model_name: str, trainable: bool = True) -> None:
        super().__init__()
        self.text_model = transformers.AutoModel.from_pretrained(model_name)

        for param in self.text_model.parameters():
            param.requires_grad = trainable

        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        output = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state

        return last_hidden_state[:, self.target_token_idx, :]


class MotionEncoder(nn.Module):
    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        trainable: bool = True,
        patch_size=16,
        num_frames=224,
        feature_dim=512,
    ) -> None:
        super().__init__()
        
        # Image size: [height=num_frames, width=feature_dim (multiple of patch_size)]
        # feature_dim should be patch_size * some_value to leverage pretrained weights
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
            img_size=(num_frames, feature_dim),
        )

        for param in self.model.parameters():
            param.requires_grad = trainable

        self.target_token_idx = 0

    def forward(self, x):
        return self.model(x)


class ClipModel(nn.Module):
    def __init__(
        self,
        motion_encoder_alias="vit_base_patch16_224_in21k",
        text_encoder_alias="distilbert-base-uncased",
        motion_encoder_pretrained: bool = True,
        motion_encoder_trainable: bool = True,
        text_encoder_trainable: bool = True,
        motion_embedding_dims: int = 768,
        text_embedding_dims: int = 768,
        projection_dims: int = 256,
        dropout: float = 0.5,
        logit: float = 0.07,
        patch_size: int = 16,
        num_frames: int = 224,
        num_groups: int = 64,
        point_encoder_config: dict = None,
    ) -> None:
        super().__init__()
        
        self.num_frames = num_frames
        self.patch_size = patch_size
        
        # Initialize PointNet + Transformer encoder for point cloud processing
        if point_encoder_config is None:
            point_encoder_config = {
                "dvae_config": {
                    "encoder_dim": 256,
                    "group_size": 32,
                    "num_group": num_groups,
                    "num_tokens": 512,
                },
                "transformer_config": {
                    "embed_dim": 768,
                    "depth": 4,
                    "num_heads": 12,
                    "mlp_ratio": 4.0,
                    "qkv_bias": False,
                    "qk_scale": None,
                    "drop_rate": 0.0,
                    "attn_drop_rate": 0.0,
                    "drop_path_rate": 0.1,
                },
            }
        
        dvae_config = DictConfig(point_encoder_config["dvae_config"])
        transformer_config = DictConfig(point_encoder_config["transformer_config"])
        
        self.point_encoder = PointnetTransformer(
            dvae_config=dvae_config,
            transformer_config=transformer_config,
        )
        
        # Output dimension from PointnetTransformer
        point_output_dim = dvae_config["num_tokens"] if "num_tokens" in dvae_config else 512
        # Ensure feature_dim is a multiple of patch_size for pretrained weights
        feature_dim = ((point_output_dim + patch_size - 1) // patch_size) * patch_size
        
        # Store feature dimension for later use
        self.feature_dim = feature_dim
        
        # If point encoder output doesn't match required dimension, add projection
        if point_output_dim != feature_dim:
            self.point_to_image = nn.Linear(point_output_dim, feature_dim)
        else:
            self.point_to_image = nn.Identity()

        motion_encoder = MotionEncoder(
            model_name=motion_encoder_alias,
            pretrained=motion_encoder_pretrained,
            trainable=motion_encoder_trainable,
            patch_size=patch_size,
            num_frames=num_frames,
            feature_dim=feature_dim,
        )
        text_encoder = TextEncoder(
            model_name=text_encoder_alias, trainable=text_encoder_trainable
        )

        self.motion_encoder = motion_encoder
        self.text_encoder = text_encoder

        self.motion_projection = ProjectionHead(
            embedding_dim=motion_embedding_dims,
            projection_dim=projection_dims,
            dropout=dropout,
        )
        self.text_projection = ProjectionHead(
            embedding_dim=text_embedding_dims,
            projection_dim=projection_dims,
            dropout=dropout,
        )

        self.logit_scale = nn.Parameter(torch.tensor(np.log(1 / logit)))

        self.log_softmax = nn.LogSoftmax(dim=-1)

    def encode_motion(self, motion):
        #dataset give 9 features per point, we only need xyz
        
        motion = motion[..., :3]
        # motion shape: [batch, frames, points, 3]
        batch_size, frames, points, _ = motion.shape
        
        # Flatten batch and frames for efficient processing: [batch*frames, points, 3]
        motion_flat = motion.view(batch_size * frames, points, 3)
        
        # Process all frames at once through PointNet + Transformer
        # Returns: [batch*frames, 3, feature_dim] - already stacked cls, mean, max
        point_feats = self.point_encoder(motion_flat)
        
        # Project to ensure feature_dim alignment if needed
        # point_feats shape: [batch*frames, 3, feature_dim]
        batch_frames, num_channels, feat_dim = point_feats.shape
        point_feats = point_feats.view(batch_frames * num_channels, feat_dim)
        point_feats = self.point_to_image(point_feats)
        point_feats = point_feats.view(batch_frames, num_channels, -1)  # [batch*frames, 3, feature_dim]
        
        # Reshape to separate batch and frames: [batch, frames, 3, feature_dim]
        point_feats = point_feats.view(batch_size, frames, num_channels, -1)
        
        # Permute to match image format: [batch, 3, frames, feature_dim]
        motion_image = point_feats.permute(0, 2, 1, 3)  # [batch, 3, frames, feature_dim]
        
        # Process through motion encoder
        motion_features = self.motion_encoder(motion_image)
        motion_embeddings = self.motion_projection(motion_features)
        return motion_embeddings

    def encode_text(self, text):
        text_features = self.text_encoder(
            input_ids=text["input_ids"], attention_mask=text["attention_mask"]
        )

        text_embeddings = self.text_projection(text_features)

        return text_embeddings

    def contrastive_loss(self, logits: torch.Tensor) -> torch.Tensor:
        return nn.functional.cross_entropy(
            logits, torch.arange(len(logits), device=logits.device)
        )

    def clip_loss(self, similarity: torch.Tensor) -> torch.Tensor:
        caption_loss = self.contrastive_loss(similarity)
        motion_loss = self.contrastive_loss(similarity.t())
        return (caption_loss + motion_loss) / 2.0

    def forward(self, motion, text, return_loss=False):
        motion_embeds = self.encode_motion(motion)
        text_embeds = self.encode_text(text)

        # normalized features
        motion_embeds = motion_embeds / motion_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, motion_embeds.t()) * logit_scale
        logits_per_motion = logits_per_text.T

        if return_loss:
            return self.clip_loss(logits_per_text)
        else:
            return motion_embeds, text_embeds
