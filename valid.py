"""SOTS validation during training (flat data layout, sharded across ranks)."""
import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from data.data_load import valid_dataloader
from utils import Adder
from skimage.metrics import peak_signal_noise_ratio






def _valid(
        model: nn.Module,
        args: argparse.Namespace,
        ep: int,
        *,
        rank: int = 0,
        world_size: int = 1
) -> float:
    """Run SOTS validation, sharding the 500 images across all ranks.

    Every rank validates its own ``~500 / world_size`` subset concurrently (no
    rank waits at a barrier for another to finish the whole set), then the local
    ``(psnr * count, count)`` totals are all-reduced so each rank returns the
    same global mean PSNR.  This keeps validation fast even on slow storage.

    Args:
        model: The (optionally DDP-wrapped) model.
        args: Parsed CLI arguments (``valid_dir``/``hazy_dir``/``gt_dir``/``num_worker``/``result_dir``).
        ep: Current epoch, used for the per-epoch output directory.
        rank: Process rank.
        world_size: Total process count.

    Returns:
        The global mean PSNR over the SOTS test set (identical on every rank).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = valid_dataloader(
        args.valid_dir,
        data=args.data,
        hazy_dir=args.hazy_dir,
        gt_dir=args.gt_dir,
        batch_size=1,
        num_workers=args.num_worker,
        distributed=world_size > 1,
        world_size=world_size,
        rank=rank
    )
    model.eval()
    psnr_adder = Adder()

    with torch.no_grad():
        # WMCB 需要输入为 64 的倍数：stage3 特征 H/4 经 pool=8 得 H/32，HaarDWT 要求其为偶数。
        factor = 64
        for idx, data in enumerate(dataset):
            input_img, label_img = data
            input_img = input_img.to(device)

            h, w = input_img.shape[2], input_img.shape[3]
            H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
            padh = H - h if h % factor != 0 else 0
            padw = W - w if w % factor != 0 else 0
            input_img = F.pad(input_img, (0, padw, 0, padh), 'reflect')

            if not os.path.exists(os.path.join(args.result_dir, '%d' % ep)):
                os.makedirs(os.path.join(args.result_dir, '%d' % ep))

            pred = model(input_img)[2]
            pred = pred[:, :, :h, :w]

            pred_clip = torch.clamp(pred, 0, 1)
            p_numpy = pred_clip.squeeze(0).cpu().numpy()
            label_numpy = label_img.squeeze(0).cpu().numpy()

            psnr = peak_signal_noise_ratio(p_numpy, label_numpy, data_range=1)
            psnr_adder(psnr)
            print('\r%03d' % idx, end=' ', flush=True)

    model.train()
    mean_psnr = psnr_adder.average() or 0.0
    count = psnr_adder.count
    if world_size > 1:
        total = torch.tensor([mean_psnr * count, count], device=device, dtype=torch.float64)
        torch.distributed.all_reduce(total)
        mean_psnr = (total[0] / max(total[1].item(), 1.0)).item()
    return mean_psnr
