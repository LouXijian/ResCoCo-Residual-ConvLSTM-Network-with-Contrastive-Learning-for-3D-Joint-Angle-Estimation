import argparse
import logging

import numpy as np
import torch
import torch.onnx
from models.IS import DeepConvLSTM
from scipy.spatial.transform import Rotation
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from utils.WEVAL_reader_2sensors import WEVAL_processor
from utils.utils import sixD2mat
from utils.utils import write_board, EarlyStopping

INFO = 20
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=INFO,
    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger()
logger.setLevel(INFO)


def test_evaluate(model, loader):
    model = model.to(device)
    model.eval()
    MSE = MAE = num = 0
    with torch.no_grad():
        for i, batch_data in enumerate(loader):
            input, label = batch_data
            input = input.float().to(device)
            label = label.float().numpy()
            seq_len = label.shape[1]
            num += input.shape[0]
            true_angle = label
            true_angle = true_angle.reshape(-1, 6)
            true_angle = sixD2mat(true_angle)
            true_angle = Rotation.from_matrix(true_angle)
            true_angle = true_angle.as_euler('xyz', degrees=True)

            est_angle, _ = model(input)
            est_angle = est_angle.data.cpu().numpy().reshape(-1, 6)
            est_angle = sixD2mat(est_angle)
            est_angle = Rotation.from_matrix(est_angle)
            est_angle = est_angle.as_euler('xyz', degrees=True)

            diff = np.array(true_angle) - np.array(est_angle)
            temp_MSE = np.sum((diff) ** 2, axis=0)
            MSE += temp_MSE
            MAE += np.sum(np.abs(diff), axis=0)
        RMSE = np.sqrt(MSE / num / seq_len)
        MAE = MAE / num / seq_len
    return {'RMSE': RMSE, 'MAE': MAE}


def train(model, train_loader, val_loader, num_epochs):
    model = model.to(device)
    loss = 0

    # Train the model
    for epoch in range(num_epochs):
        for i, batch_data in enumerate(train_loader):
            model.train()
            input, label = batch_data
            input = input.float().to(device)
            label = label.float().to(device)
            # Forward pass
            est_angle, _ = model(input)
            loss = criterion(est_angle, label)
            # Backward and optimize
            full_optimizer.zero_grad()
            loss.backward()
            full_optimizer.step()
        errors = test_evaluate(model, val_loader)
        messages = list(info + temp for temp in ['/RMSE', '/MAE', '/loss'])
        content = [errors['RMSE'][2], errors['MAE'][2], loss.item()]
        write_board(writer, messages, content, epoch)
        full_early_stopping(sum(errors['MAE']) / 3, model)
        full_scheduler.step()
        logger.info(
            "This is the " + str(epoch + 1) + "epoch, the loss is " + str(loss.item()) + ", the val MAE is " + str(
                errors['MAE']))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DeepConvLSTM model for 3D knee joint kinematics estimation')
    parser.add_argument('--location', type=str, default='LOCAL')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--exp_num', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=100)

    parser.add_argument('--seq_length', type=int, default=60)
    parser.add_argument('--num_conv_layers', type=int, default=7)
    parser.add_argument('--num_LSTM_layers', type=int, default=1)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--filter_size', type=int, default=3)
    parser.add_argument('--label_len', type=int, default=40)

    parser.add_argument('--rotate_aug_size', type=int, default=1)
    parser.add_argument('--placement_aug_size', type=int, default=1)
    parser.add_argument('--align_range', type=int, default=60)
    parser.add_argument('--pretrained_batch_size', type=int, default=256)
    parser.add_argument('--sensor_pair_thigh',type=int,default=0)
    parser.add_argument('--sensor_pair_shank',type=int,default=0)

    args = parser.parse_args()
    # WEVAL_path = "E:/Research/IMU/data/WEVAL/"
    WEVAL_path = "/project/6005622/xijian/Data/AIMU/WEVAL/"

    sensor_pair = [args.sensor_pair_thigh,args.sensor_pair_shank]
    writer = SummaryWriter("./logs")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    info = str(args.num_conv_layers) + "Conv" + str(args.num_LSTM_layers) + "LSTM_" + str(
        args.lr) + "_" + str(args.sensor_pair_thigh) + str(args.sensor_pair_shank) + "_" + str(args.exp_num)
    print(info)
    print("Experiment description: ResConvLSTM model with acc and ang as input, shortened label.")
    print("The training dataset is enlarged with two augmentation methods.")
    print("Test the CL framework.")

    # The sensor order is RUAcc, RLAcc, LUAcc, LLAcc, RUAng, RLAng, LUAng, LLAng
    # The joint order is RightKnee, LeftKnee
    train_loader, val_loader, test_loader = WEVAL_processor(WEVAL_path, args.seq_length, 5, args.batch_size,
                                                            args.label_len, args.rotate_aug_size,
                                                            args.placement_aug_size, args.align_range, sensor_pair)
    criterion = nn.MSELoss()

    full_model = DeepConvLSTM(int(args.seq_length), args.hidden_size, args.filter_size, 2, device, args.label_len,
                              conv_layer_num=args.num_conv_layers, lstm_layer_num=args.num_LSTM_layers).to(device)
    full_optimizer = torch.optim.Adam(full_model.parameters(), args.lr, betas=(0.9, 0.98), eps=1e-9)
    full_scheduler = StepLR(full_optimizer, step_size=50, gamma=0.1)
    full_early_stopping = EarlyStopping(20, True, './checkpoint/' + info + '.pt')

    full_model.load_state_dict(torch.load(
        '/home/xijian/Code/HPE_WEVAL_4/Exp1/checkpoint/encoder_256_300_0.5_' + str(args.exp_num) + '.pt'),strict=False)

    train(full_model, train_loader, val_loader, args.num_epochs)
    full_model.load_state_dict(torch.load('./checkpoint/' + info + '.pt'))
    errors = test_evaluate(full_model, test_loader)
    print("Total Testing RMSE = (" + str(errors['RMSE']) + "),MAE = (" + str(errors['MAE']) + ").")

