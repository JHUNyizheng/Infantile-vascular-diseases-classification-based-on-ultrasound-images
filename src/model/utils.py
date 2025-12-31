import numpy as np

import sys
import time
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
from sklearn.metrics import mean_squared_error, r2_score
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio
import torch
from torch.nn import functional as F
from torchvision.transforms.functional import resize, to_pil_image  # type: ignore

from copy import deepcopy
from typing import Tuple
from einops.layers.torch import Rearrange
from einops import repeat, rearrange


def PCA_svd(X, k, center=True):
    """
    param X: BxCxHxW
    param k: scalar
    return:
    """
    B, C, H, W = X.shape
    X = X.permute(0, 2, 3, 1)  # BxHxWxC
    X = X.reshape(B, H * W, C)
    U, S, V = torch.pca_lowrank(X, center=center)
    Y = torch.bmm(X, V[:, :, :k])
    Y = Y.reshape(B, H, W, k)
    Y = Y.permute(0, 3, 1, 2)  # BxHxWxk
    return Y


def calculate_metrics(y_true, y_pred):
    """ Ground truth """
    y_true = y_true.detach().cpu().numpy()
    # y_true = y_true > 0.5
    y_true = y_true.astype(np.uint8)
    y_true = y_true.reshape(-1)

    """ Prediction """
    y_pred = y_pred.detach().cpu().numpy()
    # y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.uint8)
    y_pred = y_pred.reshape(-1)

    score_f1 = f1_score(y_true, y_pred,  average='macro')
    score_recall = recall_score(y_true, y_pred, average='macro')
    score_precision = precision_score(y_true, y_pred, average='macro')

    return score_f1, score_recall, score_precision

def RMSE(result,test_y):
    return np.sqrt(mean_squared_error(result, test_y))

def calculate_rmse(y_true, y_pred):
    y_true = y_true.detach().cpu().numpy()
    y_true = y_true.reshape(-1)
    y_pred = y_pred.detach().cpu().numpy()
    y_pred = y_pred.reshape(-1)
    """ Prediction """
    score_RMSE = RMSE(y_pred, y_true)
    return score_RMSE

def calculate_nmse(y_true, y_pred):
    y_true = y_true.detach().cpu().numpy()
    # y_true = y_true > 0.5
    #
    y_true = y_true.reshape(-1)

    y_pred = y_pred.detach().cpu().numpy()
    y_pred = y_pred.reshape(-1)

    """ Prediction """
    MSE = mean_squared_error(y_true, y_pred)
    NMSE = MSE / np.var(y_true)

    return NMSE

def calculate_ssim(y_true, y_pred):
    y_true = y_true.detach().cpu().numpy()
    y_true = y_true.reshape(-1)
    y_pred = y_pred.detach().cpu().numpy()
    y_pred = y_pred.reshape(-1)
    """ Prediction """
    SSIM = structural_similarity(y_true, y_pred)
    return SSIM

def calculate_pnsr(y_true, y_pred):
    y_true = y_true.detach().cpu().numpy()
    y_true = y_true.reshape(-1)
    y_pred = y_pred.detach().cpu().numpy()
    y_pred = y_pred.reshape(-1)
    """ Prediction """
    PNSR = peak_signal_noise_ratio(y_true, y_pred)
    return PNSR

def calculate_mse(y_true, y_pred):
    y_true = y_true.detach().cpu().numpy()
    y_true = y_true.reshape(-1)
    y_pred = y_pred.detach().cpu().numpy()
    y_pred = y_pred.reshape(-1)
    """ Prediction """
    MSE = mean_squared_error(y_true, y_pred)
    return MSE

class ResizeLongestSide:
    """
    Resizes images to the longest side 'target_length', as well as provides
    methods for resizing coordinates and boxes. Provides methods for
    transforming both numpy array and batched torch tensors.
    """

    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        target_size = self.get_preprocess_shape(image.shape[2], image.shape[3], self.target_length)
        return np.array(resize(to_pil_image(image), target_size))

    def apply_coords(self, coords: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        """
        Expects a numpy array of length 2 in the final dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], self.target_length
        )
        # coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes(self, boxes: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        """
        Expects a numpy array shape Bx4. Requires the original image size
        in (H, W) format.
        """
        boxes = self.apply_coords(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    def apply_image_torch(self, image: torch.Tensor) -> torch.Tensor:
        """
        Expects batched images with shape BxCxHxW and float format. This
        transformation may not exactly match apply_image. apply_image is
        the transformation expected by the model.
        """
        # Expects an image in BCHW format. May not exactly match apply_image.
        target_size = self.get_preprocess_shape(image.shape[2], image.shape[3], self.target_length)
        return F.interpolate(
            image, target_size, mode="bilinear", align_corners=False, antialias=True
        )

    def apply_coords_torch(
        self, coords: torch.Tensor, original_size: Tuple[int, ...]
    ) -> torch.Tensor:
        """
        Expects a torch tensor with length 2 in the last dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], self.target_length
        )
        coords = deepcopy(coords).to(torch.float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes_torch(
        self, boxes: torch.Tensor, original_size: Tuple[int, ...]
    ) -> torch.Tensor:
        """
        Expects a torch tensor with shape Bx4. Requires the original image
        size in (H, W) format.
        """
        boxes = self.apply_coords_torch(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)