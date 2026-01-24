from torch.utils.data import Dataset
import h5py
import cv2
import os
import numpy as np
import torch

class Solar_Sat(Dataset):
    def __init__(self, dataset_dir):
        data = h5py.File(dataset_dir, "r")
        self.input_seq = data['arr_0']
        self.output_seq = data['arr_1']
        self.length = len(self.input_seq)

    def __getitem__(self, idx):
        return self.input_seq[idx], self.output_seq[idx]

    def __len__(self):
        return self.length

class Solar_Insitu(Dataset):
    def __init__(self, dataset_dir):
        data = np.load(dataset_dir)
        self.input_seq = data['arr_0']
        self.output_seq = data['arr_1']
        self.length = len(self.input_seq)

    def __getitem__(self, idx):
        return self.input_seq[idx], self.output_seq[idx]

    def __len__(self):
        return self.length

class Solar_Sat_Insitu(Dataset):
    def __init__(self, dataset_dir_sat):
        sat_data = h5py.File(dataset_dir_sat, 'r')
        self.sat_input_seq = sat_data['arr_0']
        self.sat_output_seq = sat_data['arr_1']
        self.sat_length = len(self.sat_input_seq)

    def __getitem__(self, idx):
        return self.sat_input_seq[idx], self.sat_output_seq[idx]

    def __len__(self):
        return self.sat_length

class Solar_Sat_Insitu_Multi_Input(Dataset):
    def __init__(self, dataset_dir_sat, points_dir):
        sat_data = h5py.File(dataset_dir_sat, 'r')
        points_data = h5py.File(points_dir, 'r')
        self.points = points_data['points']
        self.sat_input_seq = sat_data['arr_0']
        self.sat_output_seq = sat_data['arr_1']
        self.sat_length = len(self.sat_input_seq)

    def __getitem__(self, idx):
        input_seq = self.sat_input_seq[idx]
        output_seq = self.sat_output_seq[idx]
        points = self.points[idx]
        multi_input_seq = input_seq[:,points[1], points[0],:4]

        return input_seq, output_seq, multi_input_seq, points

    def __len__(self):
        return self.sat_length

