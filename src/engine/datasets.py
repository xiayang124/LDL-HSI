import os
import numpy as np
import utils
import torch
import CONSTANT
try:
    import src.extract_text.text_process as text_process
except:
    import extract_text.text_process as text_process
CLASS_DES = CONSTANT.CLASS_DESCRIPTION


def data_preprocess(cfg):
    """
    Main data pre-process function.
    -----------------------------------------
    Arguments:
        setting: Configuration settings.
    -----------------------------------------
    """
    # Load dataset
    dataset_name, dataset_loc = cfg.BaseSetting.get('dataset'), cfg.BaseSetting.get('root')
    HSI_orig, gt_orig = utils.read_mat(f"{dataset_loc}/{dataset_name}.mat"), utils.read_mat(f"{dataset_loc}/{dataset_name}_gt.mat")
    classes_name = CLASS_DES[dataset_name]
    
    HSI_orig = utils.normalize(HSI_orig, if_255=False)
    # Check NaN
    nan_mask = np.isnan(HSI_orig.sum(axis=-1))
    if np.count_nonzero(nan_mask) > 0:
        print(f"There are {np.count_nonzero(nan_mask)} pixels are NaN")
        HSI_orig[nan_mask], gt_orig[nan_mask] = 0, 0

    data_shape = gt_orig.shape
    # Get per-class train/test num
    train_num_list, test_num_list = utils.get_train_test_num(gt_orig, cfg.TrainingSetting.get('train_num'))
    # Split train/test gt
    train_gt, test_gt = utils.split_gt(gt_orig, train_num_list)
    # CLIP text embedding
    text_token = text_process.get_text_token(classes_name, cfg)
    # Create datasets
    train_dataset = HSIProcessing(HSI_orig, train_gt, cfg, return_coord=False, training=True)
    test_dataset = HSIProcessing(HSI_orig, test_gt, cfg, return_coord=False, training=False)
    all_dataset = HSIProcessing(HSI_orig, gt_orig, cfg, return_coord=True, training=False)
    
    extra_test_ratio = 4  # Use a larger batch size for testing to speed up

    g = torch.Generator()
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size= cfg.TrainingSetting.get('batchsize'),
        shuffle=True,
        generator=g,
        pin_memory=cfg.TrainingSetting.get('pin_memory')
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.TrainingSetting.get('batchsize') * extra_test_ratio,
        shuffle=False,
        generator=g,
        pin_memory=cfg.TrainingSetting.get('pin_memory')
    )
    all_loader = torch.utils.data.DataLoader(
        all_dataset,
        batch_size=cfg.TrainingSetting.get('batchsize') * extra_test_ratio,
        shuffle=False,
        generator=g,
        pin_memory=cfg.TrainingSetting.get('pin_memory')
    )
    return train_loader, test_loader, all_loader, classes_name, data_shape, gt_orig, HSI_orig, text_token


class HSIProcessing(torch.utils.data.Dataset):
    def __init__(self, HSI: np.ndarray, gt: np.ndarray, cfg, return_coord=False, training=False):
        if cfg.TrainingSetting.get("pca") > 0:
            HSI = utils.apply_pca(HSI, cfg.TrainingSetting.get("pca"))
        self.HSI = HSI
        self.gt = gt
        self.cfg = cfg
        self.patch_size = cfg.TrainingSetting.get('patchsize')
        self.padding = int(self.patch_size / 2)
        self.return_coord = return_coord
        self.training = training
        
        self.hsi_pad = np.pad(self.HSI, ((self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode='reflect')
        self.H, self.W, self.C = HSI.shape
        
        coords = np.argwhere(gt > 0)
        labels = gt[tuple(coords.T)]
        
        self.coords = coords
        self.labels = np.array(labels, dtype=np.int64)

    def __len__(self):
        if self.return_coord:
            return self.H * self.W
        return self.coords.shape[0]

    def __getitem__(self, idx):
        if self.return_coord:
            x_tensor, y_tensor = self._extract_all_patch(idx)
            return x_tensor, y_tensor
        y, x = self.coords[idx]
        patch = self._extract_patch(y, x)

        # Change to CxHxW
        x_tensor = torch.from_numpy(patch).permute(2, 0, 1).float()
        if self.training:
            x_tensor = self._augment_patch(x_tensor)
        y_tensor = torch.tensor(self.labels[idx]).float()
        return x_tensor, y_tensor
    
    def _extract_all_patch(self, idx):
        row = idx // self.W
        col = idx % self.W
        if idx >= self.H * self.W:
            raise IndexError("Index out of range")
        patch = self._extract_patch(row, col)
        return torch.from_numpy(patch).permute(2, 0, 1).float(), torch.tensor(self.gt[row, col]).float()

    def _extract_patch(self, y, x):
        patch = self.hsi_pad[y:y + self.patch_size, x:x + self.patch_size, :]
        return patch.astype(np.float32)
        
    def _augment_patch(self, patch):
        # Randomly rotate the patch
        # rotation
        if np.random.rand() < 0.5 and (self.cfg.Argument.get('rotation')):
            candidates = torch.tensor([0, 90, 180, 270])
            prob = torch.ones(len(candidates)) / len(candidates)
            angle = int(candidates[torch.multinomial(prob, 1)].item())
            patch = torch.rot90(patch, k=angle // 90, dims=(1, 2))

        # spectral noise
        if np.random.rand() < 0.5 and (self.cfg.Argument.get('spectral_noise')):
            scale_factor = torch.empty(patch.shape[0], dtype=patch.dtype, device=patch.device).uniform_(0.8, 1.2)
            patch = patch * (scale_factor.view(-1, 1, 1))

        # overall noise
        if np.random.rand() < 0.5 and (self.cfg.Argument.get('overall_noise')):
            noise = torch.normal(0, 0.01, patch.shape)
            patch = torch.clamp(patch + noise, -2, 2)
        return patch
