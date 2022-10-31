import random

import numpy as np
import scipy.signal as sps
import torch
from scipy.io import loadmat
from scipy.spatial.transform import Rotation
from torch.utils.data import TensorDataset


def mat2sixD(mat):
    num = mat.shape[0]
    result = mat[:, :, :2].reshape(num, 6)
    return result


def rotate_augmentation(imu, mocap, augmentation_size, align_range):
    original_mocap = mocap
    for j in range(augmentation_size):
        new_imu = np.zeros(imu.shape)
        for i in range(imu.shape[0]):
            random_rotation1 = Rotation.from_euler('xyz', np.random.random((3)) * 120 - 60, degrees=True)
            random_rotation2 = Rotation.from_euler('xyz', np.random.random((3)) * 120 - 60, degrees=True)
            new_imu[i, :, 0:3] = random_rotation1.apply(imu[i, :, 0:3])
            new_imu[i, :, 3:6] = random_rotation2.apply(imu[i, :, 3:6])
            new_imu[i, :, 6:9] = random_rotation1.apply(imu[i, :, 6:9])
            new_imu[i, :, 9:12] = random_rotation2.apply(imu[i, :, 9:12])
            # new_imu[i, :, 12:16] = (Rotation.from_quat(imu[i, :, 12:16]) * random_rotation1).as_quat()
            # new_imu[i, :, 16:20] = (Rotation.from_quat(imu[i, :, 16:20]) * random_rotation2).as_quat()
        imu = torch.cat((torch.tensor(imu), torch.tensor(new_imu)), dim=0)
        mocap = torch.cat((mocap, original_mocap), dim=0)
    return imu, mocap


def find_mocap_idx(raw_mocap, subject, session):
    idx_list = []
    if session == 1:
        for i in range(10):
            idx_list.append(raw_mocap['Optic_Session1'][subject, 6][0, 0]['trials'][0, i]['TrialNumber'][0][-2:])
    else:
        for i in range(10):
            idx_list.append(raw_mocap['Optic_Session2'][subject, 6][0, 0]['trials'][0, i]['TrialNumber'][0][-2:])
    return sorted(range(len(idx_list)), key=lambda k: idx_list[k])


def find_imu_idx(raw_imu, subject, session):
    idx_list = []
    if session == 1:
        for i in range(10):
            idx_list.append(raw_imu['session_1'][subject, 9][i, 0][-2:])
    else:
        for i in range(10):
            idx_list.append(raw_imu['session_2'][subject, 9][i, 0][-2:])
    return sorted(range(len(idx_list)), key=lambda k: idx_list[k])


def extract_data(datapath, seq_len, interval, set_type, label_len):
    # directory of data files
    raw_imu = loadmat(datapath + 'WEVAL_IMU_April_2021_array.mat')
    raw_mocap = loadmat(datapath + 'WEVAL_MOCAP_April_2021.mat')
    subject_division = {'training': [1, 12], 'validation': [13, 13], 'testing': [14, 15]}
    start_subject = subject_division[set_type][0]
    end_subject = subject_division[set_type][1]
    unit_len = 100
    interval = interval if set_type == "training" else int(label_len / 60 * 100)

    shank_sensor_pos = ['RTibShim', 'DisTibShim', 'LatTibShim']
    thigh_sensor_pos = ['DisThShim', 'AntThShim', 'ProxThShim']
    imu_session1 = []
    imu_session2 = []
    temp_imu = []
    for i in range(start_subject, end_subject + 1):
        for j in range(10):
            imu_pos = [random.choice(thigh_sensor_pos), random.choice(shank_sensor_pos)]
            for imu in imu_pos:
                temp_imu.append(raw_imu['session_1'][i, 9][j, 1][imu][0, 0][:, 0:24])  # first acc then gyro
            temp_imu = np.array(temp_imu)
            temp_imu = list(np.concatenate((temp_imu[:, :, 4:7], temp_imu[:, :, 8:11], temp_imu[:, :, 20:24]), axis=2))
            imu_session1.append(temp_imu)
            temp_imu = []
    for i in range(start_subject, end_subject + 1):
        if i == 13:
            continue
        for j in range(10):
            if i ==14:
                imu_pos = [random.choice(['DisThShim', 'ProxThShim']), random.choice(shank_sensor_pos)]
            else:
                imu_pos = [random.choice(thigh_sensor_pos), random.choice(shank_sensor_pos)]
            for imu in imu_pos:
                temp_imu.append(raw_imu['session_2'][i, 9][j, 1][imu][0, 0][:, 0:24])
            temp_imu = np.array(temp_imu)
            temp_imu = list(np.concatenate((temp_imu[:, :, 4:7], temp_imu[:, :, 8:11], temp_imu[:, :, 20:24]), axis=2))
            imu_session2.append(temp_imu)
            temp_imu = []
    imu_session1 = np.array(imu_session1)
    imu_session2 = np.array(imu_session2)

    mocap = []
    imu = []
    for i in range(start_subject, end_subject + 1):
        mocap_idx_list = find_mocap_idx(raw_mocap, i, 1)
        imu_idx_list = find_imu_idx(raw_imu, i, 1)
        for j in range(10):
            temp_mocap = raw_mocap['Optic_Session1'][i, 6][0, 0]['trials'][0, mocap_idx_list[j]]['RightKneeAngle'][:,
                         1:]
            a = np.array(~np.isnan(temp_mocap)).reshape(temp_mocap.shape)
            nonzero_rows = np.nonzero(a[:, :])
            nonzero_row_start = nonzero_rows[0][0]
            nonzero_row_end = nonzero_rows[0][-1]
            for k in range(nonzero_row_start, nonzero_row_end - unit_len + 2, interval):
                current_mocap = temp_mocap[k:k + unit_len, :]
                mocap.append(current_mocap)
                current_imu = np.concatenate(
                    (imu_session1[(i - start_subject) * 10 + imu_idx_list[j], 0][k + 200:k + unit_len + 200, :],
                     imu_session1[(i - start_subject) * 10 + imu_idx_list[j], 1][k + 200:k + unit_len + 200, :]),
                    axis=1)
                imu.append(current_imu)
    for i in range(start_subject, end_subject + 1):
        if i == 13:
            continue
        mocap_idx_list = find_mocap_idx(raw_mocap, i, 2)
        imu_idx_list = find_imu_idx(raw_imu, i, 2)
        for j in range(10):
            temp_mocap = raw_mocap['Optic_Session2'][i, 6][0, 0]['trials'][0, mocap_idx_list[j]]['RightKneeAngle'][:,
                         1:]
            a = np.array(~np.isnan(temp_mocap)).reshape(temp_mocap.shape)
            nonzero_rows = np.nonzero(a[:, :])
            nonzero_row_start = nonzero_rows[0][0]
            nonzero_row_end = nonzero_rows[0][-1]
            for k in range(nonzero_row_start, nonzero_row_end - unit_len + 2, interval):
                current_mocap = temp_mocap[k:k + unit_len, :]
                mocap.append(current_mocap)
                current_imu = np.concatenate(
                    (imu_session2[(i - start_subject) * 10 + imu_idx_list[j], 0][k + 200:k + unit_len + 200, :],
                     imu_session2[(i - start_subject) * 10 + imu_idx_list[j], 1][k + 200:k + unit_len + 200, :]),
                    axis=1)
                imu.append(current_imu)
    imu = np.array(imu)
    imu = np.concatenate(
        (imu[:, :, :3], imu[:, :, 10:13], imu[:, :, 3:6], imu[:, :, 13:16]), axis=2)
    print("Before resampling", np.array(imu.shape), np.array(mocap).shape)
    imu = sps.resample(imu, seq_len, axis=1)
    mocap = sps.resample(mocap, seq_len, axis=1)[:, (seq_len - label_len) // 2:(seq_len - label_len) // 2 + label_len,
            :]
    print("After resampling", imu.shape, mocap.shape)
    imu = np.array(imu)

    mocap = mocap.reshape((-1, 3))
    mocap = Rotation.from_euler("xyz", mocap, degrees=True)
    mocap = mocap.as_matrix()
    mocap = mat2sixD(mocap)
    mocap = mocap.reshape((imu.shape[0], label_len, 6))
    mocap = torch.tensor(mocap)
    return imu, mocap


def input_label_loader(datapath, seq_len, interval, norm_data, set_type, rotate_aug_size, placement_aug_size,
                 label_len, align_range):
    imu, mocap = [], []
    for j in range(placement_aug_size):
        new_imu, new_mocap = extract_data(datapath, seq_len, interval, set_type, label_len)
        imu = new_imu if placement_aug_size == 0 else torch.cat((torch.tensor(imu), torch.tensor(new_imu)), dim=0)
        mocap = new_mocap if placement_aug_size == 0 else torch.cat((torch.tensor(mocap), torch.tensor(new_mocap)),
                                                                   dim=0)
        print("After the ", j, "th augmentation", imu.shape, mocap.shape)

    imu = torch.tensor(imu)
    imu = imu.reshape((-1, 12))
    if norm_data is None:
        mean, std_dev = imu.mean(dim=0), imu.std(dim=0)
        norm_data = (mean, std_dev)
    else:
        mean, std_dev = norm_data
    imu = imu.sub(mean).div(std_dev + 1e-8)
    imu = imu.reshape((-1, seq_len, 12))

    imu,mocap = rotate_augmentation(imu,mocap,rotate_aug_size,align_range)

    dataset = TensorDataset(imu, mocap)
    return dataset, norm_data
