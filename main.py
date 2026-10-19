"""ConvIR dehazing entry point (adapted for our deployment).

Run modes:
- Train (single GPU):
    python main.py --mode train --version small --data ITS --data_dir <ITS根> \
        --valid_dir <SOTS根> --batch_size 20 --num_epoch 300 --amp --early_stopping 5
- Train (3x A6000, torchrun):
    torchrun --nproc_per_node=3 main.py --mode train ... --distributed
- Train with torch.compile speedup:
    python main.py --mode train ... --amp --compile [--compile_mode max-autotune-no-cudagraphs]
- Test with a checkpoint:
    python main.py --mode test --data ITS --valid_dir <SOTS根> --test_model <Best.pkl>

The dataset is expected in the flat layout ``<root>/hazy + <root>/clear``
(respectively ``RESIDE-ITS`` and ``RESIDE-SOTS/indoor``); point ``--data_dir``
at the ITS root and ``--valid_dir`` at the SOTS root.
"""

import argparse
import os

import torch
import torch.distributed as dist
from torch.backends import cudnn

from models.ConvIR import build_net
from train import _train
from eval import _eval






def main(args: argparse.Namespace):
    """Build the model (optionally DDP) and dispatch to train/test."""
    cudnn.benchmark = True

    # exist_ok=True: DDP 多 rank 同时检查+创建会竞态，创建处必须容错。
    os.makedirs(args.model_save_dir, exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)

    rank = 0
    world_size = 1
    if args.distributed:
        dist.init_process_group(backend='nccl')
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(rank % torch.cuda.device_count())

    model = build_net(args.version, args.data)
    if torch.cuda.is_available():
        model.cuda()
    # 先 compile 再包 DDP：Dynamo 直接 trace 原始模块（无 DDP forward 开销），
    # 编译图写入的仍是同一批 Parameter，DDP 的梯度 hook 照常工作。
    if args.compile:
        if hasattr(torch, 'compile'):
            mode = None if args.compile_mode == 'default' else args.compile_mode
            model = torch.compile(model, mode=mode)
            print('[compile] torch.compile enabled (mode=%s)' % args.compile_mode)
        else:
            print('[compile] PyTorch >= 2.0 required for --compile, running eagerly')
    if args.compile_dynamo_cache:
        os.makedirs(args.compile_dynamo_cache, exist_ok=True)
        torch._dynamo.config.cache_dir = args.compile_dynamo_cache
        print('[compile] dynamo cache dir: %s' % args.compile_dynamo_cache)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[torch.cuda.current_device()], find_unused_parameters=False)
        if args.compile and args.compile_mode in ('reduce-overhead', 'max-autotune'):
            print('[compile] WARNING: CUDA-graph modes (%s) + DDP may deadlock; '
                  'prefer --compile_mode default or max-autotune-no-cudagraphs for --distributed runs'
                  % args.compile_mode)

    if args.mode == 'train':
        _train(model, args, rank=rank, world_size=world_size)
    elif args.mode == 'test':
        _eval(model, args)

    if args.distributed:
        dist.destroy_process_group()






if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model / mode
    parser.add_argument('--model_name', default='ConvIR', type=str)
    parser.add_argument('--data', type=str, default='ITS', choices=['ITS', 'Haze4K', 'NHR', 'GTA5', 'real_haze'])
    parser.add_argument('--version', default='small', choices=['small', 'base', 'large'], type=str)
    parser.add_argument('--mode', default='test', choices=['train', 'test'], type=str)
    parser.add_argument('--fusion', type=str, default=None,
                        choices=['sg', 'v1', 'v1_hh', 'v1_lift'],
                        help='WMCB low/high fusion variant (default: $WMCB_FUSION or "v1"); '
                             'test mode must pass the variant the checkpoint was trained with')

    # Data layout (flat <root>/<hazy_dir> + <root>/<gt_dir>)
    parser.add_argument('--data_dir', type=str, default='')  # training root, e.g. RESIDE-ITS
    parser.add_argument('--valid_dir', type=str, default='')  # validation/test root, e.g. RESIDE-SOTS/indoor
    parser.add_argument('--hazy_dir', type=str, default='hazy')
    parser.add_argument('--gt_dir', type=str, default='clear')

    # Training hyper-parameters
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--num_epoch', type=int, default=300)
    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--num_worker', type=int, default=8)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--valid_freq', type=int, default=20)
    parser.add_argument('--resume', type=str, default='')

    # Engineering options
    parser.add_argument('--amp', action='store_true', help='enable FP16 AMP')
    parser.add_argument('--compile', action='store_true',
                        help='wrap the model with torch.compile (kernel fusion; first iterations pay compile cost)')
    parser.add_argument('--compile_mode', type=str, default='default',
                        choices=['default', 'reduce-overhead', 'max-autotune-no-cudagraphs', 'max-autotune'],
                        help='torch.compile mode: default is the safest; the CUDA-graph modes '
                             '(reduce-overhead / max-autotune) add warmup cost and are risky with --distributed')
    parser.add_argument('--compile_dynamo_cache', type=str, default='',
                        help='optional torch._dynamo cache dir so re-runs reuse compiled artifacts')
    parser.add_argument('--early_stopping', type=int, default=0,
                        help='stop after N SOTS validations without PSNR improvement (0=disabled)')
    parser.add_argument('--distributed', action='store_true', help='run under torchrun with DDP')
    parser.add_argument('--auto_test', action='store_true',
                        help='after training, load Best.pkl and evaluate PSNR+SSIM on SOTS into auto_test_results.yaml')

    # Test
    parser.add_argument('--test_model', type=str, default='')
    parser.add_argument('--save_image', type=bool, default=False, choices=[True, False])

    args = parser.parse_args()
    # Resolve the fusion variant before build_net and export it for models.wmcb:
    # explicit --fusion > WMCB_FUSION env var > "v1". Output dirs are tagged with
    # the variant so ablation runs never overwrite each other's Best.pkl.
    args.fusion = args.fusion or os.environ.get('WMCB_FUSION', 'v1')
    os.environ['WMCB_FUSION'] = args.fusion
    print(f'[fusion] WMCB low/high fusion variant: {args.fusion}')
    args.model_save_dir = os.path.join('results/', f'{args.model_name}-{args.fusion}', 'Training-Results/')
    args.result_dir = os.path.join('results/', f'{args.model_name}-{args.fusion}', 'images', args.data)
    os.makedirs(args.model_save_dir, exist_ok=True)
    print(args)
    main(args)
