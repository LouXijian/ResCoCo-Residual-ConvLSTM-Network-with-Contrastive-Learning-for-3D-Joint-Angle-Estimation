import argparse
import os

import numpy as np
import torch
# distributed training
import torch.distributed as dist
# TensorBoard
from torch.utils.tensorboard import SummaryWriter

from model.IS import Encoder
from model.model import load_optimizer, save_model
# SimCLR
from simclr import SimCLR
from simclr.modules import NT_Xent
from simclr.modules.sync_batchnorm import convert_model
from utils.input_loader import input_loader


def train(args, train_loader, model, criterion, optimizer, writer):
    loss_epoch = 0
    for step, (x_i, x_j) in enumerate(train_loader):
        optimizer.zero_grad()
        x_i = x_i.float().cuda(non_blocking=True)
        x_j = x_j.float().cuda(non_blocking=True)

        # positive pair, with encoding
        h_i, h_j, z_i, z_j = model(x_i, x_j)

        loss = criterion(z_i, z_j)
        loss.backward()

        optimizer.step()

        if dist.is_available() and dist.is_initialized():
            loss = loss.data.clone()
            dist.all_reduce(loss.div_(dist.get_world_size()))

        if args.nr == 0 and step % 50 == 0:
            print("Step [" + str(step) + "/" + str(len(train_loader)) + "]\t Loss: " + str(loss.item()))

        if args.nr == 0:
            writer.add_scalar("Loss/train_epoch", loss.item(), args.global_step)
            args.global_step += 1

        loss_epoch += loss.item()
    return loss_epoch


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_dataset = input_loader(args.datapath, args.seq_len, 5, "training", args.rotate_aug_size,
                                 args.placement_aug_size, args.label_len, args.align_range)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers)

    encoder = Encoder(seq_length=args.seq_len, device=args.device, label_len=args.label_len)
    n_features = 128  # get dimensions of fc layer

    # initialize model
    model = SimCLR(encoder, args.projection_dim, n_features*args.label_len)
    model = model.to(args.device)

    # optimizer / loss
    optimizer, scheduler = load_optimizer(args, model)
    criterion = NT_Xent(args.batch_size,args.temperature,1)

    model = convert_model(model)

    model = model.to(args.device)

    writer = None
    if args.nr == 0:
        writer = SummaryWriter()

    args.global_step = 0
    args.current_epoch = 0
    for epoch in range(args.start_epoch, args.epochs):
        lr = optimizer.param_groups[0]["lr"]
        loss_epoch = train(args, train_loader, model, criterion, optimizer, writer)

        if args.nr == 0 and scheduler:
            scheduler.step()

        if args.nr == 0 and epoch % 50 == 0:
            save_model(args, model, "training")

        if args.nr == 0:
            writer.add_scalar("Loss/train", loss_epoch / len(train_loader), epoch)
            writer.add_scalar("Misc/learning_rate", lr, epoch)
            print("Epoch [" + str(epoch) + "/" + str(args.epochs) + "]\t Loss: " + str(
                loss_epoch / len(train_loader)) + "\t lr: " + str(round(lr, 5)))
            args.current_epoch += 1

    save_model(args, model, "encoder")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SimCLR")
    parser.add_argument('--nr', type=int, default=0)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--model_path', type=str, default="./checkpoint/")
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('-label_len',type=int,default=40)
    parser.add_argument('--align_range', type=int, default=60)

    parser.add_argument('--projection_dim', type=int, default=64)
    parser.add_argument('--optimizer', type=str, default="Adam")
    parser.add_argument('--weight_decay', type=float, default=1.0e-6)
    parser.add_argument('--seq_len', type=int, default=60)

    parser.add_argument('--rotate_aug_size', type=int, default=1)
    parser.add_argument('--placement_aug_size', type=int, default=5)
    parser.add_argument('', type=int, default=256)
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--exp_num', type=int,default=1)
    parser.add_argument('--datapath', type=str, default="/project/6005622/xijian/Data/AIMU/WEVAL/")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.num_gpus = torch.cuda.device_count()

    print("rotate_aug_size", args.rotate_aug_size, '\nplacement_aug_size', args.placement_aug_size,'\ntemperature',
          args.temperature, '\nbatch size', args.batch_size)

    main(args)
