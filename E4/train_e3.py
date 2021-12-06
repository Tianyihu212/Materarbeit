import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os
from pathlib import Path
import torch.nn as nn
from engine import train_one_epoch, evaluate, train_one_epoch_contra, test_retrieval
from losses import CosContrastiveLoss
import utils
from data.dataset_contra import GLDDataset 
from efficientnet_pytorch.model_e3 import EfficientNet
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler,WeightedRandomSampler
def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=8, type=int)
    parser.add_argument('--num-workers', default=4, type=int)
    parser.add_argument('--trial', default='0', type=str,
                        help='trial name')

    parser.add_argument('--model', default='efficientnet_b0', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--lr', type=float, default=1.5e-4, metavar='LR',
                        help='learning rate (default: 1.5e-4)')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--output_dir', default='./output_e3-3',
                        help='path where to save, empty for no saving')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--resume', default=False)
    parser.add_argument('--model_path', default='output_e3-3/checkpoint_e10_75.96104708068093.pth')
    return parser


def main(args):

    print(args)
    # batch_size = 48 #72
    # num_workers = 8
    # num_classes = None
    cudnn.benchmark = True
    

    index_dataset = GLDDataset(root='../../data/train', input_size=224, subset='index')
    test_dataset = GLDDataset(root='../../test_1k', input_size=224, subset='test')
    train_dataset = GLDDataset(root='../../data/train', input_size=224, subset='train')
    val_dataset = GLDDataset(root='../../data/train', input_size=224, subset='val')
    train_sample_list = train_dataset.gen_train_sample_list()
    sampler = WeightedRandomSampler(weights=train_sample_list, num_samples=1400000, replacement=False)
    index_dataloader = DataLoader(
                                    index_dataset,
                                    batch_size=384,
                                    shuffle=False,
                                    num_workers=args.num_workers,
                                    drop_last=False,
                                    pin_memory=True
                                    # worker_init_fn=_worker_init_fn_()
                                    )

    
    test_dataloader = DataLoader(
                                    test_dataset,
                                    batch_size=384,
                                    shuffle=False,
                                    num_workers= args.num_workers,
                                    drop_last=False,
                                    pin_memory=True
                                    # worker_init_fn=_worker_init_fn_()
                                    )
    train_dataloader = DataLoader(
                                    train_dataset,
                                    sampler=sampler,
                                    batch_size=args.batch_size,
#                                     shuffle=True,
                                    num_workers= args.num_workers,
                                    drop_last=False,
                                    pin_memory=True
                                    # worker_init_fn=_worker_init_fn_()
                                    )
    val_dataloader = DataLoader(
                                    val_dataset,
                                    # sampler=test_sampler,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers= args.num_workers,
                                    drop_last=False,
                                    pin_memory=True
                                    # worker_init_fn=_worker_init_fn_()
                                    )
    # model = create_model(
    #     args.model,
    #     pretrained=True,
    #     num_classes=num_classes,
    #     drop_rate=args.drop,
    #     drop_path_rate=args.drop_path,
    #     drop_block_rate=None,
    # )
    model = EfficientNet.from_pretrained('efficientnet-b0',num_classes=81313)
    

    if args.resume:
        checkpoint = torch.load(args.model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        args.start_epoch = checkpoint['epoch'] + 1
#     model = nn.DataParallel(model)
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), args.lr, eps=args.opt_eps)

    criterion_contra = CosContrastiveLoss(margin = 0.4)
    criterion = torch.nn.CrossEntropyLoss()

    output_dir = Path(args.output_dir)#.mkdir(parents=True, exist_ok=True)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0

    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch_contra(
            model, criterion, criterion_contra, train_dataloader,
            optimizer, epoch, args.epochs
        )
        val_stats = evaluate(val_dataloader, model)
        test_stats = test_retrieval(index_dataloader, test_dataloader, model)
        print(f"Accuracy of the network on the {len(val_dataset)} test images: {val_stats['acc1']:.2f}%")
#         if test_stats["acc1"] > max_accuracy:
        if True:
            if args.output_dir:
                checkpoint_paths = [output_dir / 'checkpoint_e{}_{}.pth'.format(epoch,val_stats["acc1"] )]
                for checkpoint_path in checkpoint_paths:
#                     torch.save(model.state_dict(), checkpoint_path)
                    torch.save({
                        'model': model.state_dict(),
#                         'model': model.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        # 'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'args': args,
                    }, checkpoint_path)
        max_accuracy = max(max_accuracy, val_stats["acc1"])
        print(f'Max accuracy: {max_accuracy:.2f}%')

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'val_{k}': v for k, v in val_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch
                     }

        if args.output_dir:
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    main(args)
