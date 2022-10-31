import argparse

import numpy as np
import torch

from model.IS import Encoder, Decoder
from model.model import save_model
from simclr import SimCLR
from utils.input_label_loader import input_label_loader


def inference(loader, simclr_model, device):
    feature_vector = []
    labels_vector = []
    for step, (x, y) in enumerate(loader):
        x = x.float().to(device)
        y = y.float()
        batch_size = x.shape[0]

        # get encoding
        with torch.no_grad():
            h = simclr_model(x)

        h = h.detach()
        h = h.reshape(batch_size, 60, 16)

        feature_vector.extend(h.cpu().detach().numpy())
        labels_vector.extend(y.numpy())

        if step % 20 == 0:
            print("Step [" + str(step) + "/" + str(len(loader)) + "]\t Computing features...")

    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    print("Features shape" + str(feature_vector.shape))
    return feature_vector, labels_vector


def create_data_loaders_from_arrays(X_train, y_train, X_val, y_val, X_test, y_test, batch_size):
    train = torch.utils.data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)

    val = torch.utils.data.TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=False)

    test = torch.utils.data.TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def train(args, loader, model, criterion, optimizer):
    model.train()
    loss_epoch = 0
    accuracy_epoch = 0
    batch_num = 0
    for step, (x, y) in enumerate(loader):
        optimizer.zero_grad()

        x = x.float().to(args.device)
        y = y.float().to(args.device)

        output = model(x)
        loss = criterion(output, y)

        diff = output - y
        acc = torch.mean(torch.mean(torch.abs(diff), dim=0), dim=0)
        accuracy_epoch += acc

        loss.backward()
        optimizer.step()

        loss_epoch += loss.item()
        if step % 100 == 0:
            print(
                "Step [" + str(step) + "/" + str(len(loader)) + "]\t Loss: " + str(loss.item()) + "\t Accuracy: " + str(
                    acc) + ""
            )
        batch_num += 1

    return loss_epoch, accuracy_epoch / batch_num


def test(args, loader, model, criterion):
    loss_epoch = 0
    accuracy_epoch = 0
    batch_num = 0
    model.eval()
    for step, (x, y) in enumerate(loader):
        model.zero_grad()

        x = x.float().to(args.device)
        y = y.float().to(args.device)

        output = model(x)
        loss = criterion(output, y)

        diff = output - y
        acc = torch.mean(torch.mean(torch.abs(diff), dim=0), dim=0)
        accuracy_epoch += acc

        loss_epoch += loss.item()
        batch_num += 1

    return loss_epoch, accuracy_epoch / batch_num


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SimCLR")
    parser.add_argument('--nodes', type=int, default=1)
    parser.add_argument('--nr', type=int, default=0)
    parser.add_argument('--dataparallel', type=int, default=0)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--pretrain', type=bool, default=True)
    parser.add_argument('--projection_dim', type=int, default=64)
    parser.add_argument('--optimizer', type=str, default="Adam")
    parser.add_argument('--weight_decay', type=float, default=1.0e-6)
    parser.add_argument('--model_path', type=str, default="save")

    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--epoch_num', type=int, default=10)
    parser.add_argument('--datapath', type=str, default="E:/Research/IMU/data/WEVAL/")
    parser.add_argument('--alignment_range', type=int, default=180)
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = input_label_loader(args.datapath, args.seq_len, 5, None, "training", args.rotate_aug_size,
                                 args.placement_aug_size,
                                 args.label_len, args.align_range)
    val_dataset = input_label_loader(args.datapath, args.seq_len, 66.67, None, "validation", args.rotate_aug_size,
                               args.placement_aug_size,
                               args.label_len, args.align_range)
    test_dataset = input_label_loader(args.datapath, args.seq_len, 66.67, None, "testing", args.rotate_aug_size,
                                args.placement_aug_size,
                                args.label_len, args.align_range)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.workers)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.workers)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.workers)

    encoder = Encoder(seq_length=args.seq_len, device=args.device, label_len=args.label_len)
    n_features = 16 * 60  # get dimensions of fc layer

    # load pre-trained model from checkpoint
    simclr_model = SimCLR(encoder, args.projection_dim, n_features)
    # simclr_model.load_state_dict(
    #     torch.load("./save/checkpoint_100_" + str(args.batch_size) + ".pt", map_location=args.device.type))
    # simclr_model = simclr_model.encoder.to(args.device)
    # simclr_model.eval()

    decoder = Decoder()
    decoder = decoder.to(args.device)

    optimizer = torch.optim.Adam(decoder.parameters(), lr=3e-4)
    criterion = torch.nn.MSELoss()

    print("### Creating features from pre-trained context model ###")
    train_X, train_y = inference(train_loader, simclr_model, args.device)
    test_X, test_y = inference(test_loader, simclr_model, args.device)
    val_X, val_y = inference(val_loader, simclr_model, args.device)

    arr_train_loader, arr_val_loader, arr_test_loader = create_data_loaders_from_arrays(
        train_X, train_y, val_X, val_y, test_X, test_y, args.batch_size
    )
    val_acc = 100
    for epoch in range(args.epochs):
        loss_epoch, accuracy_epoch = train(args, arr_train_loader, decoder, criterion, optimizer)
        _, temp_val_acc = test(args, arr_val_loader, decoder, criterion)
        if temp_val_acc.sum() / 3 < val_acc:
            val_acc = temp_val_acc.sum() / 3
            save_model(args, decoder, "decoder")
            print("Saving...")
        print(
            "Epoch [" + str(epoch) + "/" + str(args.epochs) + "]\t Loss: " + str(
                loss_epoch / len(arr_train_loader)) + "\t Accuracy: " + str(accuracy_epoch))

    # final testing
    loss_epoch, accuracy_epoch = test(args, arr_test_loader, decoder, criterion)
    print("[FINAL]\t Loss: " + str(loss_epoch / len(arr_test_loader)) + "\t Accuracy: " + str(accuracy_epoch) + "")
