# -*- coding: utf-8 -*-
import h5py
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import librosa
import random


class ToTensor(object):
    r"""Convert ndarrays in sample to Tensors."""

    def __call__(self, x):
        return torch.from_numpy(x).float()


class TrainingDataset(Dataset):
    r"""Training dataset."""

    def __init__(self, file_list, conf, nsamples=64000):
        # 6535 * 50 = 326750
        self.num_repeats = 50
        self.file_list = []
        for _ in range(self.num_repeats):
            random.shuffle(file_list)
            self.file_list.extend(file_list)
        self.to_tensor = ToTensor()
        self.speech_dir = conf["speech_dir"]
        self.noise_dir = conf["noise_dir"]
        self.noise_split = 100000
        self.folder_cap = 5000
        self.epsilon = 1e-10
        self.snr_list = [-5.0, -4.0, -3.0, -2.0, -1.0, -0.0]
        self.nsamples = nsamples

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        filename = self.file_list[index]
        reader = h5py.File(os.path.join(self.speech_dir, filename), 'r')
        label = reader['clean_raw'][:]
        label = label / (np.max(np.abs(label)) + 1e-6)
        reader.close()

        noise_ind = random.randint(0, self.noise_split - 1)
        filefolder = '%d-%d/' % ((noise_ind // self.folder_cap) * self.folder_cap,
                                 (noise_ind // self.folder_cap + 1) * self.folder_cap - 1)
        filename = 'noise_%d.samp' % (noise_ind)
        filepath = os.path.join(self.noise_dir, filefolder, filename)
        reader = h5py.File(filepath, 'r')
        noise = reader['noise'][:]
        reader.close()

        label, _ = librosa.effects.trim(
            label, top_db=40, frame_length=512, hop_length=128)

        size = label.shape[0]
        start = random.randint(0, max(0, size + 1 - self.nsamples))
        label = label[start:start + self.nsamples]

        label = label / (np.max(label) + 1e-6)

        snr_c = random.randint(0, len(self.snr_list) - 1)
        snr = self.snr_list[snr_c]

        start_cut_point = random.randint(0, noise.size - label.size)
        while np.sum(noise[start_cut_point:start_cut_point + label.size] ** 2.0) == 0.0:
            start_cut_point = random.randint(0, noise.size - label.size)

        n_t = noise[start_cut_point:start_cut_point + label.size]

        alpha = np.sqrt(np.sum(label ** 2.0) /
                        (np.sum(n_t ** 2.0) * (10.0 ** (snr / 10.0))))
        snr_check = 10.0 * np.log10(np.sum(label ** 2.0) /
                                    (np.sum((n_t * alpha) ** 2.0)))
        noise = alpha * n_t
        feature = label + noise

        scale = np.sqrt(feature.size / (np.sum((feature) ** 2.0) + 1e-6))
        feature = feature * scale
        label = label * scale

        feature = np.reshape(feature, [1, -1])
        label = np.reshape(label, [1, -1])

        feature = self.to_tensor(feature)
        label = self.to_tensor(label)

        return feature, label


class EvalDataset(Dataset):
    r"""Evaluation dataset."""

    def __init__(self, filename, length):
        self.filename = filename
        self.length = length
        self.to_tensor = ToTensor()

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        reader = h5py.File(self.filename, 'r')
        reader_grp = reader[str(index)]
        feature = reader_grp['noisy_raw'][:]
        label = reader_grp['clean_raw'][:]
        reader.close()

        scale = np.sqrt(feature.size / (np.sum(feature ** 2.0) + 1e-6))
        feature = feature * scale
        label = label * scale

        feature = np.reshape(feature, [1, -1])
        label = np.reshape(label, [1, -1])

        feature = self.to_tensor(feature)
        label = self.to_tensor(label)

        return feature, label


class TrainCollate(object):

    def __init__(self):
        self.name = 'collate'

    def __call__(self, batch):
        if isinstance(batch, list):

            feat_dim = batch[0][0].shape[-1]
            label_dim = batch[0][1].shape[-1]

            feat_nchannels = batch[0][0].shape[0]
            label_nchannels = batch[0][1].shape[0]
            sorted_batch = sorted(
                batch, key=lambda x: x[0].shape[1], reverse=True)
            lengths = list(
                map(lambda x: (x[0].shape[1], x[1].shape[1]), sorted_batch))

            padded_feature_batch = torch.zeros(
                (len(lengths), feat_nchannels, lengths[0][0]))
            padded_label_batch = torch.zeros(
                (len(lengths), label_nchannels, lengths[0][1]))
            lengths1 = torch.zeros((len(lengths),), dtype=torch.int32)
            for i in range(len(lengths)):
                padded_feature_batch[i, :, 0:lengths[i][0]] = sorted_batch[i][0]
                padded_label_batch[i, :, 0:lengths[i][1]] = sorted_batch[i][1]
                lengths1[i] = lengths[i][1]

            return padded_feature_batch, padded_label_batch, lengths1
        else:
            raise TypeError('`batch` should be a list.')


class EvalCollate(object):

    def __init__(self):
        self.name = 'collate'

    def __call__(self, batch):
        if isinstance(batch, list):
            feat_nchannels = batch[0][0].shape[0]
            label_nchannels = batch[0][1].shape[0]
            sorted_batch = sorted(
                batch, key=lambda x: x[0].shape[1], reverse=True)
            lengths = list(
                map(lambda x: (x[0].shape[1], x[1].shape[1]), sorted_batch))

            padded_feature_batch = torch.zeros(
                (len(lengths), feat_nchannels, lengths[0][0]))
            padded_label_batch = torch.zeros(
                (len(lengths), label_nchannels, lengths[0][1]))
            lengths1 = torch.zeros((len(lengths),), dtype=torch.int32)
            for i in range(len(lengths)):
                padded_feature_batch[i, :, 0:lengths[i][0]] = sorted_batch[i][0]
                padded_label_batch[i, :, 0:lengths[i][1]] = sorted_batch[i][1]
                lengths1[i] = lengths[i][1]

            return padded_feature_batch, padded_label_batch, lengths1
        else:
            raise TypeError('`batch` should be a list.')
