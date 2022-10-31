import torch
from torch.utils.data import TensorDataset

from utils.input_label_loader import extract_data, rotate_augmentation


def input_augmentation(datapath, seq_len, interval, set_type, rotate_aug_size, placement_aug_size,
                 label_len, align_range):
    imu, mocap = [], []
    for j in range(placement_aug_size):
        new_imu, new_mocap = extract_data(datapath, seq_len, interval, set_type, label_len)
        imu = new_imu if placement_aug_size == 0 else torch.cat((torch.tensor(imu), torch.tensor(new_imu)), dim=0)
        mocap = new_mocap if placement_aug_size == 0 else torch.cat((torch.tensor(mocap), torch.tensor(new_mocap)),
                                                                    dim=0)

    imu = torch.tensor(imu).type(torch.float64)
    imu = imu.reshape((-1, 12))
    mean, std_dev = imu.mean(dim=0), imu.std(dim=0)
    imu = imu.sub(mean).div(std_dev + 1e-8)
    imu = imu.reshape((-1, seq_len, 12))

    print("After augmentation the input shape is", imu.shape)

    imu, _ = rotate_augmentation(imu, mocap, rotate_aug_size, align_range)

    return imu

def input_loader(datapath, seq_len, interval, set_type, rotate_aug_size, placement_aug_size,
                           label_len, align_range):
    x_i = input_augmentation(datapath, seq_len, interval, set_type, rotate_aug_size, placement_aug_size,
                 label_len, align_range)
    x_j = input_augmentation(datapath, seq_len, interval, set_type, rotate_aug_size, placement_aug_size,
                 label_len, align_range)
    dataset = TensorDataset(x_i, x_j)
    return dataset
