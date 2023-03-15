import os
import re
import shutil
import sys
import time
import scipy
import torch
import logging
import datetime
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def gen_list(wav_dir, append):
    l = []
    lst = os.listdir(wav_dir)
    lst.sort()
    for f in lst:
        if re.search(append, f):
            l.append(f)
    return l


class ProgressBar:
    def __init__(self, minV=0, maxV=100, barLength=50):
        self.minV = minV
        self.maxV = maxV
        self.barLength = barLength
        self.persent = 0

    @staticmethod
    def format_time(seconds):
        'Formats time as the string "HH:MM:SS".'
        return str(datetime.timedelta(seconds=int(seconds)))

    def start(self):
        self.start_time = time.time()

    def finish(self):
        sys.stdout.write('\n')
        sys.stdout.flush()
        self.finish = True

    def __get_persent__(self, progress):
        self.status = ""
        if progress < self.minV:
            self.status = "Halt..."
            return 0
        if progress > self.maxV:
            self.status = "Done..."
            return 1
        return (progress - self.minV) / (self.maxV - self.minV)

    def update_progress(self, progress, prefix_message='', suffix_message=''):
        self.persent = self.__get_persent__(progress)
        block = int(round(self.persent * self.barLength))
        sys.stdout.write(str(block))
        sys.stdout.flush()

        cur_time = time.time()
        elapsed_time = cur_time - self.start_time
        if self.persent == 0:
            ela_fomat = '_:__:__'
            eta_fomat = '_:__:__'
        else:
            eta_time = elapsed_time * (1 - self.persent) / self.persent
            ela_fomat = self.format_time(elapsed_time)
            eta_fomat = self.format_time(eta_time)

        text = "\r{6}: [{0}] {1}% {2} elapsed:{3} ETA:{4} {5}".format(
            "#" * block + "-" * (self.barLength - block),
            round(self.persent * 100, 2),
            self.status, ela_fomat,
            eta_fomat,
            suffix_message,
            prefix_message
        )
        sys.stdout.write(text)
        sys.stdout.flush()


class Checkpoint(object):
    def __init__(
            self,
            start_epoch=None,
            train_loss=None,
            eval_loss=None,
            best_loss=np.inf,
            state_dict=None,
            optimizer=None,
            scaler=None,
    ):
        self.start_epoch = start_epoch
        self.train_loss = train_loss
        self.eval_loss = eval_loss
        self.best_loss = best_loss
        self.state_dict = state_dict
        self.optimizer = optimizer
        self.scaler = scaler

    def save(self, is_best, filename1, filename2):
        logging.info('Saving checkpoint at "%s"' % filename1)
        torch.save(self, filename1)
        if is_best:
            shutil.copyfile(filename1, filename2)

    def load(self, filename):
        if os.path.isfile(filename):
            logging.info('Loading checkpoint from "%s"' % filename)
            checkpoint = torch.load(filename, map_location='cpu')

            self.start_epoch = checkpoint.start_epoch
            self.train_loss = checkpoint.train_loss
            self.eval_loss = checkpoint.eval_loss
            self.best_loss = checkpoint.best_loss
            self.state_dict = checkpoint.state_dict
            self.optimizer = checkpoint.optimizer
            self.scaler = checkpoint.scaler
        else:
            raise ValueError('No checkpoint found at "%s"' % filename)


class STFT(nn.Module):
    def __init__(self, win_size=320, hop_size=160, requires_grad=False):
        super(STFT, self).__init__()

        self.win_size = win_size
        self.hop_size = hop_size
        self.n_overlap = self.win_size // self.hop_size
        self.requires_grad = requires_grad

        win = torch.from_numpy(scipy.hamming(self.win_size).astype(np.float32))
        win = F.relu(win)
        win = nn.Parameter(data=win, requires_grad=self.requires_grad)
        self.register_parameter('win', win)

        fourier_basis = np.fft.fft(np.eye(self.win_size))
        fourier_basis_r = np.real(fourier_basis).astype(np.float32)
        fourier_basis_i = np.imag(fourier_basis).astype(np.float32)

        self.register_buffer('fourier_basis_r', torch.from_numpy(fourier_basis_r))
        self.register_buffer('fourier_basis_i', torch.from_numpy(fourier_basis_i))

        idx = torch.tensor(range(self.win_size // 2 - 1, 0, -1), dtype=torch.long)
        self.register_buffer('idx', idx)

        self.eps = torch.finfo(torch.float32).eps

    def kernel_fw(self):
        fourier_basis_r = torch.matmul(self.fourier_basis_r, torch.diag(self.win))
        fourier_basis_i = torch.matmul(self.fourier_basis_i, torch.diag(self.win))

        fourier_basis = torch.stack([fourier_basis_r, fourier_basis_i], dim=-1)
        forward_basis = fourier_basis.unsqueeze(dim=1)

        return forward_basis

    def kernel_bw(self):
        inv_fourier_basis_r = self.fourier_basis_r / self.win_size
        inv_fourier_basis_i = -self.fourier_basis_i / self.win_size

        inv_fourier_basis = torch.stack([inv_fourier_basis_r, inv_fourier_basis_i], dim=-1)
        backward_basis = inv_fourier_basis.unsqueeze(dim=1)
        return backward_basis

    def window(self, n_frames):
        assert n_frames >= 2
        seg = sum([self.win[i * self.hop_size:(i + 1) * self.hop_size] for i in range(self.n_overlap)])
        seg = seg.unsqueeze(dim=-1).expand((self.hop_size, n_frames - self.n_overlap + 1))
        window = seg.contiguous().view(-1).contiguous()

        return window

    def stft(self, sig, return_complex=False):
        batch_size = sig.shape[0]
        n_samples = sig.shape[1]
        pad_right = (n_samples // self.hop_size - 1) * self.hop_size + self.win_size - n_samples
        sig = F.pad(sig, (0, pad_right))
        n_samples = sig.shape[1]
        cutoff = self.win_size // 2 + 1

        sig = sig.view(batch_size, 1, n_samples)
        kernel = self.kernel_fw()
        kernel_r = kernel[..., 0]
        kernel_i = kernel[..., 1]
        spec_r = F.conv1d(sig,
                          kernel_r[:cutoff],
                          stride=self.hop_size,
                          padding=self.win_size - self.hop_size)
        spec_i = F.conv1d(sig,
                          kernel_i[:cutoff],
                          stride=self.hop_size,
                          padding=self.win_size - self.hop_size)
        spec_r = spec_r.transpose(-1, -2).contiguous()
        spec_i = spec_i.transpose(-1, -2).contiguous()

        mag = torch.sqrt(spec_r ** 2 + spec_i ** 2)
        pha = torch.atan2(spec_i.data, spec_r.data)

        return (spec_r, spec_i) if return_complex else (mag, pha)

    def istft(self, x, siglen):
        spec_r = x[:, 0, :, :]
        spec_i = x[:, 1, :, :]

        n_frames = spec_r.shape[1]

        spec_r = torch.cat([spec_r, spec_r.index_select(dim=-1, index=self.idx)], dim=-1)
        spec_i = torch.cat([spec_i, -spec_i.index_select(dim=-1, index=self.idx)], dim=-1)
        spec_r = spec_r.transpose(-1, -2).contiguous()
        spec_i = spec_i.transpose(-1, -2).contiguous()

        kernel = self.kernel_bw()
        kernel_r = kernel[..., 0].transpose(0, -1)
        kernel_i = kernel[..., 1].transpose(0, -1)

        sig = F.conv_transpose1d(spec_r,
                                 kernel_r,
                                 stride=self.hop_size,
                                 padding=self.win_size - self.hop_size) \
              - F.conv_transpose1d(spec_i,
                                   kernel_i,
                                   stride=self.hop_size,
                                   padding=self.win_size - self.hop_size)
        sig = sig.squeeze(dim=1)

        window = self.window(n_frames)
        sig = sig / (window + self.eps)

        return sig[..., :siglen]

    def forward(self, sig):
        siglen = sig.shape[-1]
        sig = self.stft(sig)
        sig = self.istft(sig, siglen)
        return sig
