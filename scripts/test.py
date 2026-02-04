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

import logging
import os
import sys
from os.path import join as pjoin

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, os.getcwd())
from datasets import TextMotionPatchDataset
from datasets.humanML import HumanMLDataset, collate_fn
from models.clip import ClipModel
from models.clip_point import ClipModel as ClipModelPoint

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_name="test_config", config_path="../conf")
def main(cfg: DictConfig) -> None:
    saved_cfg = OmegaConf.load(pjoin(cfg.checkpoints_dir, ".hydra/config.yaml"))
    print(OmegaConf.to_yaml(saved_cfg))
    test_dataloader = prepare_test_dataset(saved_cfg)
    model, tokenizer = prepare_test_model(saved_cfg)
    eval(saved_cfg, test_dataloader, model, tokenizer)


def prepare_test_dataset(cfg):
    # Check if using point cloud dataset
    if hasattr(cfg.dataset, 'num_points'):
        # Point cloud dataset
        test_dataset = HumanMLDataset(
            data_root=cfg.dataset.data_root,
            seed=cfg.train.seed,
            split="test" if not cfg.eval.eval_train else "train",
            num_frames=cfg.dataset.num_frames,
            num_points=cfg.dataset.num_points
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=cfg.train.batch_size,
            shuffle=False,
            num_workers=cfg.dataset.num_workers if hasattr(cfg.dataset, 'num_workers') else 4,
            collate_fn=collate_fn
        )
    else:
        # Original patch dataset
        mean = np.load(pjoin(cfg.dataset.data_root, "Mean_raw.npy"))
        std = np.load(pjoin(cfg.dataset.data_root, "Std_raw.npy"))

        if cfg.eval.eval_train:
            test_split_file = pjoin(cfg.dataset.data_root, "train.txt")
        else:
            test_split_file = pjoin(cfg.dataset.data_root, "test.txt")
        test_dataset = TextMotionPatchDataset(
            cfg,
            mean,
            std,
            test_split_file,
            eval_mode=True,
            patch_size=cfg.train.patch_size,
            fps=True,
        )
        test_dataloader = DataLoader(
            test_dataset, batch_size=cfg.train.batch_size, shuffle=False, num_workers=16
        )
    return test_dataloader


def prepare_test_model(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    motion_encoder_alias = cfg.model.motion_encoder
    text_encoder_alias = cfg.model.text_encoder
    motion_embedding_dims: int = 768
    text_embedding_dims: int = 768
    projection_dims: int = 256

    tokenizer = AutoTokenizer.from_pretrained(text_encoder_alias)

    # Check if using point cloud model
    if hasattr(cfg.dataset, 'num_points'):
        # Point cloud model
        point_encoder_config = None
        if hasattr(cfg.model, 'point_encoder'):
            point_encoder_config = {
                "dvae_config": dict(cfg.model.point_encoder.dvae_config),
                "transformer_config": dict(cfg.model.point_encoder.transformer_config),
            }
        
        model = ClipModelPoint(
            motion_encoder_alias=motion_encoder_alias,
            text_encoder_alias=text_encoder_alias,
            motion_embedding_dims=motion_embedding_dims,
            text_embedding_dims=text_embedding_dims,
            projection_dims=projection_dims,
            patch_size=cfg.train.patch_size,
            dropout=0.5 if cfg.dataset.dataset_name == "HumanML3D" else 0.0,
            num_frames=cfg.dataset.num_frames,
            num_groups=point_encoder_config["dvae_config"]["num_group"] if point_encoder_config else 64,
            point_encoder_config=point_encoder_config,
        )
    else:
        # Original patch model
        model = ClipModel(
            motion_encoder_alias=motion_encoder_alias,
            text_encoder_alias=text_encoder_alias,
            motion_embedding_dims=motion_embedding_dims,
            text_embedding_dims=text_embedding_dims,
            projection_dims=projection_dims,
            patch_size=cfg.train.patch_size,
        )

    if cfg.eval.use_best_model:
        model_path = pjoin(cfg.checkpoints_dir, "best_model.pt")
    else:
        model_path = pjoin(cfg.checkpoints_dir, "last_model.pt")

    print(model_path)
    state_dict = torch.load(model_path)

    model.load_state_dict(state_dict)

    model.to(device)

    return model, tokenizer


def eval(cfg, test_dataloader, model, tokenizer=None, verbose=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_pair = dict()

    all_imgs_feat = []
    all_captions_feat = []

    all_img_idxs = []
    all_captions = []

    step = 0
    with torch.no_grad():
        model.eval()
        test_pbar = tqdm(test_dataloader, leave=False)
        for batch in test_pbar:
            step += 1
            # Handle both old format (4 values) and new format (2 values)
            if len(batch) == 4:
                texts, motions, m_length, img_indexs = batch
            else:
                motions, texts = batch
                # Generate dummy img_indexs based on batch position
                img_indexs = torch.arange(len(texts)) + step * len(texts)
            
            motions = motions.to(device)

            texts_token = tokenizer(
                texts, padding=True, truncation=True, return_tensors="pt"
            ).to(device)

            motion_features = model.encode_motion(motions)
            text_features = model.encode_text(texts_token)

            # normalized features
            motion_features = motion_features / motion_features.norm(
                dim=1, keepdim=True
            )
            text_features = text_features / text_features.norm(dim=1, keepdim=True)

            for i in range(motion_features.size(0)):
                all_imgs_feat.append(motion_features[i].cpu().numpy())
                all_captions_feat.append(text_features[i].cpu().numpy())

                all_captions.append(texts[i])
                if isinstance(img_indexs, torch.Tensor):
                    all_img_idxs.append(img_indexs[i].item())
                else:
                    all_img_idxs.append(img_indexs[i])

    all_captions = np.array(all_captions)
    for img_idx, caption in zip(all_img_idxs, all_captions):
        dataset_pair[img_idx] = np.where(all_captions == caption)[0]

    all_imgs_feat = np.vstack(all_imgs_feat)
    all_captions_feat = np.vstack(all_captions_feat)

    # match test queries to target motions, get nearest neighbors
    sims_t2m = 100 * all_captions_feat.dot(all_imgs_feat.T)

    t2m_r1 = 0
    # Text->Motion
    ranks = np.zeros(sims_t2m.shape[0])
    for index, score in enumerate(tqdm(sims_t2m)):
        inds = np.argsort(score)[::-1]
        # Score
        rank = 1e20
        for i in dataset_pair[index]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    for k in [1, 2, 3, 5, 10]:
        # Compute metrics
        r = 100.0 * len(np.where(ranks < k)[0]) / len(ranks)
        if k == 1:
            t2m_r1 = r
        if verbose:
            log.info(f"t2m_recall_top{k}_correct_composition: {r:.2f}")
    if verbose:
        log.info(f"t2m_recall_median_correct_composition: {np.median(ranks)+1:.2f}")

    # match motions queries to target texts, get nearest neighbors
    sims_m2t = sims_t2m.T

    m2t_r1 = 0
    # Motion->Text
    ranks = np.zeros(sims_m2t.shape[0])
    for index, score in enumerate(tqdm(sims_m2t)):
        inds = np.argsort(score)[::-1]
        # Score
        rank = 1e20
        for i in dataset_pair[index]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    for k in [1, 2, 3, 5, 10]:
        # Compute metrics
        r = 100.0 * len(np.where(ranks < k)[0]) / len(ranks)
        if k == 1:
            m2t_r1 = r
        if verbose:
            log.info(f"m2t_recall_top{k}_correct_composition: {r:.2f}")
    if verbose:
        log.info(f"m2t_recall_median_correct_composition: {np.median(ranks)+1:.2f}")

    return t2m_r1, m2t_r1


if __name__ == "__main__":
    main()
