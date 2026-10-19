"""ConvIR training loop, adapted for our deployment.

Changes vs the official ConvIR ``train.py``:

- Removed TensorBoard (progress goes to stdout).
- Optional FP16 AMP (``torch.autocast`` + ``GradScaler``); the FFT loss is
  still computed in FP32 outside ``autocast`` so its numerical behavior is
  unchanged.
- Optional multi-GPU training via a pre-wrapped ``DistributedDataParallel``
  model (started with ``torchrun`` and ``--distributed``).
- Optional early stopping based on the SOTS validation PSNR.
- Data layout is the flat ``<root>/<hazy_dir>`` + ``<root>/<gt_dir>``.
"""
import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import train_dataloader
from utils import Adder, Timer, load_model_state, normalize_state_dict
from valid import _valid
from warmup_scheduler import GradualWarmupScheduler






def _train(
        model: nn.Module,
        args: argparse.Namespace,
        *,
        rank: int = 0,
        world_size: int = 1
):
    """Run the training loop (optionally distributed).

    Args:
        model: The model, optionally wrapped in ``DistributedDataParallel``.
        args: Parsed CLI arguments.
        rank: Process rank (0 = main).
        world_size: Total number of processes.

    Returns:
        None.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = torch.nn.L1Loss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-8)
    dataloader = train_dataloader(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_worker,
        data=args.data,
        hazy_dir=args.hazy_dir,
        gt_dir=args.gt_dir,
        crop_size=args.crop_size,
        distributed=args.distributed,
        world_size=world_size,
        rank=rank
    )
    max_iter = len(dataloader)
    warmup_epochs = 3
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epoch - warmup_epochs, eta_min=1e-6)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
    scheduler.step()
    epoch = 1
    if args.resume:
        state = torch.load(args.resume, map_location=device)
        epoch = state['epoch']
        optimizer.load_state_dict(state['optimizer'])
        load_model_state(model, state['model'])
        print('[rank%d] Resume from epoch %d' % (rank, epoch))
        epoch += 1

    use_amp = args.amp and device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    epoch_pixel_adder = Adder()
    epoch_fft_adder = Adder()
    iter_pixel_adder = Adder()
    iter_fft_adder = Adder()
    epoch_timer = Timer('m')
    iter_timer = Timer('m')
    best_psnr = -1.0
    no_improve_rounds = 0

    for epoch_idx in range(epoch, args.num_epoch + 1):
        if args.distributed:
            dataloader.sampler.set_epoch(epoch_idx)
        epoch_timer.tic()
        iter_timer.tic()
        model.train()
        for iter_idx, batch_data in enumerate(dataloader):
            input_img, label_img = batch_data
            input_img = input_img.to(device)
            label_img = label_img.to(device)

            optimizer.zero_grad()
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                pred_img = model(input_img)
                label_img2 = F.interpolate(label_img, scale_factor=0.5, mode='bilinear')
                label_img4 = F.interpolate(label_img, scale_factor=0.25, mode='bilinear')
                l1 = criterion(pred_img[0], label_img4)
                l2 = criterion(pred_img[1], label_img2)
                l3 = criterion(pred_img[2], label_img)
                loss_content = l1 + l2 + l3

            # FFT losses are computed in FP32 (outside autocast) to preserve the official behavior.
            loss_fft = None
            for pred, label in ((pred_img[0], label_img4), (pred_img[1], label_img2), (pred_img[2], label_img)):
                pred_fft = torch.fft.fft2(pred.float(), dim=(-2, -1))
                pred_fft = torch.stack((pred_fft.real, pred_fft.imag), -1)
                label_fft = torch.fft.fft2(label.float(), dim=(-2, -1))
                label_fft = torch.stack((label_fft.real, label_fft.imag), -1)
                fft_loss = criterion(pred_fft, label_fft)
                loss_fft = fft_loss if loss_fft is None else loss_fft + fft_loss

            loss = loss_content + 0.1 * loss_fft
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.001)
            scaler.step(optimizer)
            scaler.update()

            iter_pixel_adder(loss_content.item())
            iter_fft_adder(loss_fft.item())
            epoch_pixel_adder(loss_content.item())
            epoch_fft_adder(loss_fft.item())

            if (iter_idx + 1) % args.print_freq == 0 and rank == 0:
                print("Time: %7.4f Epoch: %03d Iter: %4d/%4d LR: %.10f Loss content: %7.4f Loss fft: %7.4f" % (
                    iter_timer.toc(), epoch_idx, iter_idx + 1, max_iter, scheduler.get_lr()[0], iter_pixel_adder.average(),
                    iter_fft_adder.average()))
                iter_timer.tic()
                iter_pixel_adder.reset()
                iter_fft_adder.reset()

        if rank == 0:
            overwrite_name = os.path.join(args.model_save_dir, 'model.pkl')
            torch.save({'model': normalize_state_dict(model.state_dict()), 'optimizer': optimizer.state_dict(), 'epoch': epoch_idx}, overwrite_name)
            if epoch_idx % args.save_freq == 0:
                save_name = os.path.join(args.model_save_dir, 'model_%d.pkl' % epoch_idx)
                torch.save({'model': normalize_state_dict(model.state_dict())}, save_name)
            print("EPOCH: %02d\nElapsed time: %4.2f Epoch Pixel Loss: %7.4f Epoch FFT Loss: %7.4f" % (
                epoch_idx, epoch_timer.toc(), epoch_pixel_adder.average(), epoch_fft_adder.average()))
        epoch_fft_adder.reset()
        epoch_pixel_adder.reset()
        scheduler.step()

        if epoch_idx % args.valid_freq == 0:
            val = _valid(model, args, epoch_idx, rank=rank, world_size=world_size)
            # 早停判定在所有 rank 上基于同一全局 PSNR 同步演化，因此 break 必须对所有 rank 生效，
            # 否则 rank0 先退出训练循环，rank1/2 的 DDP backward ALLREDUCE 将找不到对端而超时。
            if val >= best_psnr:
                best_psnr = val
                no_improve_rounds = 0
                if rank == 0:
                    torch.save({'model': normalize_state_dict(model.state_dict())}, os.path.join(args.model_save_dir, 'Best.pkl'))
            else:
                no_improve_rounds += 1
            if rank == 0:
                print('%03d epoch \n Average PSNR %.2f dB' % (epoch_idx, val))
            if 0 < args.early_stopping <= no_improve_rounds:
                if rank == 0:
                    print('[early stop] no SOTS PSNR improvement for %d validations at epoch %d' % (no_improve_rounds, epoch_idx))
                break

    if rank == 0:
        if args.auto_test:
            from eval import auto_test
            best_pkl = os.path.join(args.model_save_dir, 'Best.pkl')
            if os.path.exists(best_pkl):
                load_model_state(model, torch.load(best_pkl, map_location='cpu')['model'])
                print('[auto_test] loaded %s' % best_pkl)
            auto_test(model, args, os.path.join(args.model_save_dir, 'auto_test_results.yaml'))
        save_name = os.path.join(args.model_save_dir, 'Final.pkl')
        torch.save({'model': normalize_state_dict(model.state_dict())}, save_name)
