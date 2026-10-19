"""SOTS evaluation for ConvIR (adapted for flat data layout + auto-test hook)."""
import argparse
import os
import time

from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import device
from torch.utils.data.dataloader import DataLoader

from utils import Adder, load_model_state
from data.data_load import test_dataloader
from pytorch_msssim import ssim






def _run_eval(
        model: nn.Module,
        dataloader: DataLoader,
        device: device,
        *,
        result_dir: str | Path = None,
        save_image: bool = False
) -> tuple[float, float, float]:
    """Run PSNR+SSIM evaluation over a dataloader with the current model weights.

    Args:
        model: Model already holding the weights to evaluate (eval mode is set/restored here).
        dataloader: Test dataloader yielding ``(hazy, clear[, name])``.
        device: torch device.
        result_dir: Optional directory for saved predictions (used only when ``save_image``).
        save_image: Save dehazed PNGs to ``result_dir`` when ``True``.

    Returns:
        A tuple ``(psnr, ssim, avg_time_s)`` of the dataset averages.
    """
    model.eval()
    # WMCB 需要输入为 64 的倍数：stage3 特征 H/4 经 pool=8 得 H/32，HaarDWT 要求其为偶数。
    factor = 64
    psnr_adder = Adder()
    ssim_adder = Adder()
    time_adder = Adder()

    with torch.no_grad():
        for iter_idx, data in enumerate(dataloader):
            has_name = len(data) == 3
            input_img, label_img = data[0], data[1]
            name = data[2] if has_name else None
            input_img = input_img.to(device)
            label_img = label_img.to(device)

            h, w = input_img.shape[2], input_img.shape[3]
            H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
            padh = H - h if h % factor != 0 else 0
            padw = W - w if w % factor != 0 else 0
            input_img = F.pad(input_img, (0, padw, 0, padh), 'reflect')

            tm = time.time()
            pred = model(input_img)[2]
            pred = pred[:, :, :h, :w]
            time_adder(time.time() - tm)

            pred_clip = torch.clamp(pred, 0, 1)
            psnr_val = 10 * torch.log10(1 / F.mse_loss(pred_clip, label_img))
            down_ratio = max(1, round(min(H, W) / 256))
            ssim_val = ssim(
                F.adaptive_avg_pool2d(pred_clip, (int(H / down_ratio), int(W / down_ratio))),
                F.adaptive_avg_pool2d(label_img, (int(H / down_ratio), int(W / down_ratio))),
                data_range=1, size_average=False)

            if save_image and result_dir is not None and name is not None:
                os.makedirs(result_dir, exist_ok=True)
                save_name = os.path.join(result_dir, name[0])
                to_save = pred_clip + 0.5 / 255
                F.to_pil_image(to_save.squeeze(0).cpu(), 'RGB').save(save_name)

            psnr_adder(psnr_val.item())
            ssim_adder(ssim_val.item())
            print('%d iter PSNR: %.2f SSIM: %.5f time: %f' % (
                iter_idx + 1, psnr_val.item(), ssim_val.item(), time_adder.average()))

    model.train()
    return psnr_adder.average(), ssim_adder.average(), time_adder.average()






def _eval(
        model: nn.Module,
        args: argparse.Namespace
):
    """Load ``--test_model`` into ``model`` and evaluate on SOTS."""
    state_dict = torch.load(args.test_model, map_location='cpu')
    # load_model_state tolerates ``module.`` (DDP) and ``_orig_mod.`` (torch.compile) prefixes
    # in both the checkpoint and the live model.
    load_model_state(model, state_dict['model'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    dataloader = test_dataloader(args.valid_dir, args.data, args.hazy_dir, args.gt_dir, batch_size=1, num_workers=0)
    torch.cuda.empty_cache()
    psnr, ssim, avg_time = _run_eval(model, dataloader, device, result_dir=args.result_dir, save_image=args.save_image)
    print('==========================================================')
    print('The average PSNR is %.2f dB' % psnr)
    print('The average SSIM is %.5f dB' % ssim)
    print("Average time: %f" % avg_time)






def auto_test(
        model: nn.Module,
        args: argparse.Namespace,
        result_path: str | Path
):
    """Evaluate the current model weights on SOTS and write a YAML summary.

    Called from the training loop on rank 0 after training finishes; the caller is
    expected to load the best checkpoint first.

    Args:
        model: The model (optionally DDP-wrapped) to evaluate.
        args: Parsed CLI arguments (``valid_dir``/``hazy_dir``/``gt_dir``).
        result_path: Destination file for the YAML summary.

    Returns:
        None.
    """
    import yaml

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    dataloader = test_dataloader(args.valid_dir, args.data, args.hazy_dir, args.gt_dir,
                                 batch_size=1, num_workers=0)
    torch.cuda.empty_cache()
    print('\n[auto_test] evaluating on SOTS ...')
    psnr, ssim, avg_time = _run_eval(model, dataloader, device)
    summary = {'PSNR': round(psnr, 3), 'SSIM': round(ssim, 5), 'avg_time_s': round(avg_time, 3)}
    with open(result_path, 'w', encoding='utf-8') as fp:
        yaml.safe_dump(summary, fp, allow_unicode=True)
    print('[auto_test] PSNR=%.3f SSIM=%.5f -> %s' % (psnr, ssim, result_path))
