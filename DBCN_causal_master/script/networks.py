import math
import librosa
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.autograd.variable
from thop import profile, clever_format
from torch.autograd import Variable


def frames_data(noisy_speech, hop_size=160, use_window=True):
    n_samples = noisy_speech.shape[1]
    pad_right = (n_samples // hop_size - 1) * hop_size + hop_size * 2 - n_samples
    noisy_speech = F.pad(noisy_speech, (0, pad_right))
    hamming_win = torch.from_numpy(np.hamming(hop_size * 2).astype(np.float32)).cuda().reshape(1, 1, hop_size * 2)

    # noisy_speech = [batch_size, nsample]
    noisy_speech = noisy_speech[:, :noisy_speech.size(1) // hop_size * hop_size]
    # left data
    frame_speech = noisy_speech.reshape(noisy_speech.size(0), -1, hop_size)
    # right data
    frame_speech2 = frame_speech.clone()[:, 1:]

    frames = torch.cat([frame_speech[:, :-1], frame_speech2[:, :]], dim=-1)
    if use_window is True:
        frames = frames * hamming_win
    return frames


def overlap_data(data, siglen, hop_size=160, deweight=True):
    left_data = data[:, :, :hop_size]
    left_data = torch.cat([left_data, torch.zeros_like(left_data[:, -1:, :]).cuda()], dim=1)
    right_data = data[:, :, hop_size:]
    right_data = torch.cat([torch.zeros_like(right_data[:, 0:1, :]).cuda(), right_data], dim=1)

    # [:,:left_data.size(1), :]
    overlap_res = (left_data + right_data).reshape(data.size(0), -1)
    if deweight is True:
        overlap_res[..., hop_size:-hop_size] /= 2.0
    return overlap_res[..., :siglen]


class Loss1D(nn.Module):
    def __init__(self, loss_type="mse"):
        super(Loss1D, self).__init__()
        # utterance level loss
        self.loss_type = loss_type
        assert self.loss_type in ["mse", "mae"], "Loss not implemented"

    def forward(self, outputs, labels, nsamples):
        loss_mask = self.lossMask(labels, nsamples)
        if self.loss_type == "mse":
            loss = ((outputs - labels) * loss_mask) ** 2
        elif self.loss_type == "mae":
            loss = torch.abs((outputs - labels) * loss_mask)
        loss = torch.sum(loss) / torch.sum(loss_mask)
        return loss

    def lossMask(self, labels, nsamples):
        loss_mask = torch.zeros_like(labels).requires_grad_(False)
        for j, seq_len in enumerate(nsamples):
            loss_mask.data[j, 0:seq_len] += 1.0
        return loss_mask


class LossPCM(nn.Module):
    def __init__(self, frame_size=512, frame_shift=128, magnitude_type="l1", loss_type="l1"):
        super(LossPCM, self).__init__()
        self.frame_size = frame_size
        self.frame_shift = frame_shift
        self.magnitude_type = magnitude_type
        self.loss_type = loss_type
        assert self.magnitude_type in ["l1", "l2"]
        assert self.loss_type in ["l1", "l2"]

    def forward(self, est, tgt, inp, nsamples_list):
        # est : [batch_size, num_samples]
        # tgt : [batch_size, num_samples]
        # inp : [batch_size, num_samples]
        # nsamples_list: [batch_size]
        # print("est", est.shape, "tgt", tgt.shape, "inp", inp.shape)
        assert est.shape == tgt.shape
        assert inp.shape == tgt.shape

        window = torch.hamming_window(
            window_length=self.frame_size,
            dtype=tgt.dtype,
            device=tgt.device,
            requires_grad=False,
        )
        est_n = inp - est
        tgt_n = inp - tgt

        all_batch = torch.cat([est, tgt, est_n, tgt_n], dim=0)
        a, c = all_batch.shape

        all_s = torch.stft(
            all_batch,
            n_fft=self.frame_size,
            win_length=self.frame_size,
            hop_length=self.frame_shift,
            pad_mode="constant",
            window=window,
        )
        all_s = all_s.reshape([a, all_s.shape[1], all_s.shape[2], 2])

        all_s_r, all_s_i = all_s[..., 0], all_s[..., 1]
        if self.magnitude_type == "l1":
            all_s_m = torch.abs(all_s_r) + torch.abs(all_s_i)
        elif self.magnitude_type == "l2":
            all_s_m = all_s_r ** 2 + all_s_i ** 2
        # all_s_m: [batch_size*4, num_out_channels, num_freq_bins, num_frames)

        est_s_m, tgt_s_m, est_n_m, tgt_n_m = torch.chunk(all_s_m, 4, dim=0)

        if self.magnitude_type == "l2":
            est_s_m = torch.sqrt(est_s_m + 1e-5)
            est_n_m = torch.sqrt(est_n_m + 1e-5)
            tgt_s_m = torch.sqrt(tgt_s_m)
            tgt_n_m = torch.sqrt(tgt_n_m)

        est = torch.stack([est_s_m, est_n_m], dim=1)
        tgt = torch.stack([tgt_s_m, tgt_n_m], dim=1)
        # est: [batch_size, 2, num_out_channels, num_freq_bins, num_frames]
        # tgt: [batch_size, 2, num_out_channels, num_freq_bins, num_frames]

        loss_mask = self.loss_mask(est, nsamples_list)

        if self.loss_type == "l1":
            loss = torch.abs((tgt - est) * loss_mask)
        elif self.loss_type == "l2":
            loss = ((tgt - est) * loss_mask) ** 2
        loss = torch.sum(loss, dim=(1, 2, 3))
        loss_mask = torch.sum(loss_mask, dim=(1, 2, 3))
        loss = torch.sum(loss) / torch.sum(loss_mask)

        return loss

    def loss_mask(self, tgt, nsamples_list):
        # tgt: [batch_size, 2, num_out_channels, num_freq_bins, num_frames]
        # nsamples_list: [batch_size]

        mask = torch.zeros_like(tgt, requires_grad=False)
        for i, nsamples in enumerate(nsamples_list):
            mask[i, :, :, : nsamples // self.frame_shift + 1] = 1.0

        # mask: [batch_size, 2, num_out_channels, num_freq_bins, num_frames]
        return mask


class TorchSRS(nn.Module):
    def __init__(self, frame_len=320, frame_shift=160):
        super(TorchSRS, self).__init__()
        self.frame_len = frame_len
        self.frame_shift = frame_shift
        self.n_pad_zeros = 2

    def forward(self, s):
        batch, sig_len = s.shape
        pad_right = (sig_len // self.frame_shift - 1) * self.frame_shift + self.frame_len - sig_len
        s = F.pad(s, (0, pad_right))
        batch, sig_len = s.shape

        n_frames = sig_len // self.frame_shift - 1
        frames = torch.as_strided(s, size=[batch, n_frames, self.frame_len], stride=[sig_len, self.frame_shift, 1])

        # causual pad
        fft_window = librosa.filters.get_window('hann', self.frame_len, fftbins=True)
        fft_window = librosa.util.pad_center(fft_window, size=self.frame_len)
        fft_window = torch.Tensor(fft_window).cuda()
        frames = frames * fft_window

        frames = F.pad(frames, [self.frame_len + self.n_pad_zeros, 0], 'constant', value=0)

        srs = torch.fft.rfft(frames, dim=-1)
        srs = torch.real(srs)
        srs = srs[..., :-2]

        return srs


class TorchISRS(nn.Module):
    def __init__(self, frame_len=320, frame_shift=160):
        super(TorchISRS, self).__init__()
        self.frame_len = frame_len
        self.frame_shift = frame_shift

    def forward(self, srs, siglen):
        srs = F.pad(srs, [0, 2], 'constant', value=0.)
        # srs = srs.astype(np.complex)
        imag = torch.zeros(srs.shape, dtype=torch.float32).cuda()
        complex_srs = torch.complex(srs, imag)
        frames = torch.fft.irfft(complex_srs, dim=-1)
        frames = frames[..., -self.frame_len:]
        batch, n_frames, _ = frames.shape

        left_data = frames[:, :, :self.frame_shift]
        left_data = torch.cat([left_data, torch.zeros_like(left_data[:, -1:, :]).cuda()], dim=1)
        right_data = frames[:, :, self.frame_shift:]
        right_data = torch.cat([torch.zeros_like(right_data[:, 0:1, :]).cuda(), right_data], dim=1)

        s = (left_data + right_data).reshape(batch, -1)[..., :siglen]

        return s * 2


class FFTBridge(nn.Module):
    def __init__(self, len):
        super(FFTBridge, self).__init__()
        self.len = len
        basis = np.fft.fft(np.eye(self.len))
        basis = np.real(basis[0:len, 0:len])
        win = np.ones([1, 1, len])
        self.fbasis = nn.Parameter(torch.FloatTensor(basis), requires_grad=True)
        self.bbasis = nn.Parameter(torch.FloatTensor(basis), requires_grad=True)
        self.reset_parameters()

    def forward(self, input, fft=True):
        if fft is True:
            output = torch.matmul(input, self.fbasis.to(input.device))
        else:
            output = torch.matmul(input, self.bbasis.to(input.device)) / self.len
        return output

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.fbasis.size(1))
        self.fbasis.data.uniform_(-stdv, stdv)
        self.bbasis.data.uniform_(-stdv, stdv)


class G_CLstm(nn.Module):
    def __init__(self, channel, feature, groups, bidirectional=False):
        super(G_CLstm, self).__init__()
        self.channel = channel
        self.feature = feature
        self.groups = groups
        self.bidir = bidirectional
        self.step = feature * channel // groups
        self.sublstmlist = nn.ModuleList()
        for l in range(2):
            for g in range(groups):
                self.sublstmlist.append(
                    nn.LSTM(input_size=self.step, hidden_size=self.step, num_layers=1, batch_first=True,
                            bidirectional=bidirectional))
        snd_layer_index = []
        for i in range(channel // groups):
            for g in range(groups):
                for f in range(feature):
                    snd_layer_index.append(i * feature + g * self.step + f)
        self.register_buffer('snd_layer_index', torch.from_numpy(np.array(snd_layer_index, dtype=np.int64)))
        if self.bidir:
            self.linear = nn.Linear(in_features=640, out_features=320)

    def forward(self, input):
        [batch, channel, time, feature] = input.shape
        output = input.transpose(1, 2).contiguous().view(batch, time, -1)
        # first layer
        out = []
        for g in range(self.groups):
            self.sublstmlist[g].flatten_parameters()
            lstm_out, _ = self.sublstmlist[g](output[:, :, g * self.step:(g + 1) * self.step])
            out.append(lstm_out)
        output = torch.cat(out, dim=2)
        if self.bidir:
            output = self.linear(output)
        # second layer
        out = []
        for g in range(self.groups):
            self.sublstmlist[g + self.groups].flatten_parameters()
            output1 = torch.index_select(output, dim=2, index=self.snd_layer_index[g * self.step:(g + 1) * self.step])
            lstm_out, _ = self.sublstmlist[g + self.groups](output1)
            out.append(lstm_out)
        output = torch.cat(out, dim=2)
        if self.bidir:
            output = self.linear(output)

        lstm_out = output.contiguous().view(batch, time, channel, feature)
        lstm_out = lstm_out.transpose(1, 2)

        return lstm_out


class RmsNorm(nn.Module):
    def __init__(self, feature_size):
        super(RmsNorm, self).__init__()
        self.weights = nn.Parameter(torch.FloatTensor(np.ones([1, 1, 1, feature_size])))
        self.bias = nn.Parameter(torch.FloatTensor(np.zeros([1, 1, 1, feature_size])))

    def forward(self, input):
        rms = torch.sqrt(torch.mean(input ** 2, dim=[1, 2], keepdim=True))
        return input / (rms + 1.0e-8) * self.weights + self.bias


class GCN_Norm2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1), padding=(0, 0), bias=True,
                 out_feature_size=0):
        super(GCN_Norm2d, self).__init__()
        self.conv_l = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                stride=stride, padding=padding, bias=bias)
        self.conv_s = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                stride=stride, padding=padding, bias=bias)
        self.norm = RmsNorm(out_feature_size) if out_feature_size != 0 else nn.Identity()

    def forward(self, input):
        return self.conv_l(input) * torch.sigmoid(self.norm(self.conv_s(input)))


class TransGCN_Norm2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1), output_padding=(0, 0), padding=(0, 0),
                 bias=True, out_feature_size=0):
        super(TransGCN_Norm2d, self).__init__()
        self.conv_l = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, output_padding=output_padding, padding=padding, bias=bias)
        self.conv_s = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, output_padding=output_padding, padding=padding, bias=bias)
        self.norm = RmsNorm(out_feature_size) if out_feature_size != 0 else nn.Identity()

    def forward(self, input):
        return self.conv_l(input) * torch.sigmoid(self.norm(self.conv_s(input)))


class Net(nn.Module):
    def __init__(self, frame_size=320, frame_shift=160):
        super(Net, self).__init__()
        self.frame_size = frame_size
        self.frame_shift = frame_shift
        C = [64, 64, 64, 64, 64]
        LSTM = 64
        self.get_srs = TorchSRS(frame_size, frame_shift)
        self.get_isrs = TorchISRS(frame_size, frame_shift)

        # time tower
        # encoder
        self.tconv1 = GCN_Norm2d(in_channels=1, out_channels=C[0], kernel_size=(1, 3), stride=(1, 2), padding=(0, 1),
                                 bias=False, out_feature_size=160)
        self.tconv1_ac = nn.PReLU(C[0])
        self.tconv2 = GCN_Norm2d(in_channels=C[0] * 2, out_channels=C[1], kernel_size=(1, 3), stride=(1, 2),
                                 padding=(0, 1), bias=False, out_feature_size=80)
        self.tconv2_ac = nn.PReLU(C[1])
        self.tconv3 = GCN_Norm2d(in_channels=C[1] * 2, out_channels=C[2], kernel_size=(1, 3), stride=(1, 2),
                                 padding=(0, 1), bias=False, out_feature_size=40)
        self.tconv3_ac = nn.PReLU(C[2])
        self.tconv4 = GCN_Norm2d(in_channels=C[2] * 2, out_channels=C[3], kernel_size=(1, 3), stride=(1, 2),
                                 padding=(0, 1), bias=False, out_feature_size=20)
        self.tconv4_ac = nn.PReLU(C[3])
        self.tconv5 = GCN_Norm2d(in_channels=C[3] * 2, out_channels=C[4], kernel_size=(1, 3), stride=(1, 2),
                                 padding=(0, 1), bias=False, out_feature_size=10)
        self.tconv5_ac = nn.PReLU(C[4])
        self.tconv6 = GCN_Norm2d(in_channels=C[4] * 2, out_channels=LSTM, kernel_size=(1, 3), stride=(1, 2),
                                 padding=(0, 1), bias=False, out_feature_size=5)
        self.tconv6_ac = nn.PReLU(LSTM)
        # lstm
        self.t_lstm_norm = RmsNorm(5)
        self.t_lstm = G_CLstm(channel=LSTM, feature=5, groups=2)
        # decoder
        self.tconv6_t = TransGCN_Norm2d(in_channels=LSTM, out_channels=C[4], kernel_size=(1, 3), stride=(1, 2),
                                        output_padding=(0, 1), padding=(0, 1), bias=False, out_feature_size=10)
        self.tconv6_t_ac = nn.PReLU(C[4])
        self.tconv5_t = TransGCN_Norm2d(in_channels=C[4] * 3, out_channels=C[3], kernel_size=(1, 3), stride=(1, 2),
                                        output_padding=(0, 1), padding=(0, 1), bias=False, out_feature_size=20)
        self.tconv5_t_ac = nn.PReLU(C[3])
        self.tconv4_t = TransGCN_Norm2d(in_channels=C[3] * 3, out_channels=C[2], kernel_size=(1, 3), stride=(1, 2),
                                        output_padding=(0, 1), padding=(0, 1), bias=False, out_feature_size=40)
        self.tconv4_t_ac = nn.PReLU(C[2])
        self.tconv3_t = TransGCN_Norm2d(in_channels=C[2] * 3, out_channels=C[1], kernel_size=(1, 3), stride=(1, 2),
                                        output_padding=(0, 1), padding=(0, 1), bias=False, out_feature_size=80)
        self.tconv3_t_ac = nn.PReLU(C[1])
        self.tconv2_t = TransGCN_Norm2d(in_channels=C[1] * 3, out_channels=C[0], kernel_size=(1, 3), stride=(1, 2),
                                        output_padding=(0, 1), padding=(0, 1), bias=False, out_feature_size=160)
        self.tconv2_t_ac = nn.PReLU(C[0])
        self.tconv1_t = TransGCN_Norm2d(in_channels=C[0] * 3, out_channels=1, kernel_size=(1, 3), stride=(1, 2),
                                        output_padding=(0, 1), padding=(0, 1), bias=False, out_feature_size=320)

        # fre tower
        # encoder
        self.fconv1 = GCN_Norm2d(in_channels=1, out_channels=C[0], kernel_size=(1, 3), stride=(1, 2), padding=(0, 1),
                                 bias=False, out_feature_size=160)
        self.fconv1_ac = nn.PReLU(C[0])
        self.fconv2 = GCN_Norm2d(in_channels=C[0] * 2, out_channels=C[1], kernel_size=(1, 3), stride=(1, 2),
                                 padding=(0, 1), bias=False, out_feature_size=80)
        self.fconv2_ac = nn.PReLU(C[1])
        self.fconv3 = GCN_Norm2d(in_channels=C[1] * 2, out_channels=C[2], kernel_size=(1, 3), stride=(1, 2),
                                 padding=(0, 1), bias=False, out_feature_size=40)
        self.fconv3_ac = nn.PReLU(C[2])
        self.fconv4 = GCN_Norm2d(in_channels=C[2] * 2, out_channels=C[3], kernel_size=(1, 3), stride=(1, 2),
                                 padding=(0, 1), bias=False, out_feature_size=20)
        self.fconv4_ac = nn.PReLU(C[3])
        self.fconv5 = GCN_Norm2d(in_channels=C[3] * 2, out_channels=C[4], kernel_size=(1, 3), stride=(1, 2),
                                 padding=(0, 1), bias=False, out_feature_size=10)
        self.fconv5_ac = nn.PReLU(C[4])
        self.fconv6 = GCN_Norm2d(in_channels=C[4] * 2, out_channels=LSTM, kernel_size=(1, 3), stride=(1, 2),
                                 padding=(0, 1), bias=False, out_feature_size=5)
        self.fconv6_ac = nn.PReLU(LSTM)
        # lstm
        self.f_lstm_norm = RmsNorm(5)
        self.f_lstm = G_CLstm(channel=LSTM, feature=5, groups=2)
        # decoder
        self.fconv6_t = TransGCN_Norm2d(in_channels=LSTM, out_channels=C[4], kernel_size=(1, 3), stride=(1, 2),
                                        output_padding=(0, 1), padding=(0, 1), bias=False, out_feature_size=10)
        self.fconv6_t_ac = nn.PReLU(C[4])
        self.fconv5_t = TransGCN_Norm2d(in_channels=C[4] * 3, out_channels=C[3], kernel_size=(1, 3), stride=(1, 2),
                                        output_padding=(0, 1), padding=(0, 1), bias=False, out_feature_size=20)
        self.fconv5_t_ac = nn.PReLU(C[3])
        self.fconv4_t = TransGCN_Norm2d(in_channels=C[3] * 3, out_channels=C[2], kernel_size=(1, 3), stride=(1, 2),
                                        output_padding=(0, 1), padding=(0, 1), bias=False, out_feature_size=40)
        self.fconv4_t_ac = nn.PReLU(C[2])
        self.fconv3_t = TransGCN_Norm2d(in_channels=C[2] * 3, out_channels=C[1], kernel_size=(1, 3), stride=(1, 2),
                                        output_padding=(0, 1), padding=(0, 1), bias=False, out_feature_size=80)
        self.fconv3_t_ac = nn.PReLU(C[1])
        self.fconv2_t = TransGCN_Norm2d(in_channels=C[1] * 3, out_channels=C[0], kernel_size=(1, 3), stride=(1, 2),
                                        output_padding=(0, 1), padding=(0, 1), bias=False, out_feature_size=160)
        self.fconv2_t_ac = nn.PReLU(C[0])
        self.fconv1_t = TransGCN_Norm2d(in_channels=C[0] * 3, out_channels=1, kernel_size=(1, 3), stride=(1, 2),
                                        output_padding=(0, 1), padding=(0, 1), bias=False, out_feature_size=320)

        # bridge layer
        self.bridge1 = FFTBridge(len=160)
        self.bridge2 = FFTBridge(len=80)
        self.bridge3 = FFTBridge(len=40)
        self.bridge4 = FFTBridge(len=20)
        self.bridge5 = FFTBridge(len=10)
        self.MSELoss = Loss1D()
        self.PCMLoss = LossPCM(frame_size, frame_shift)

    def forward(self, mixture, nsamples, label):
        with torch.no_grad():
            mixture = mixture.squeeze(1)
            label = label.squeeze(1)
            batch, siglen = mixture.shape
            slice_wav = frames_data(mixture, self.frame_shift, False)
            slice_fft = self.get_srs(mixture)

        e1_t = self.tconv1_ac(self.tconv1(slice_wav.unsqueeze(dim=1)))
        e1_f = self.fconv1_ac(self.fconv1(slice_fft.unsqueeze(dim=1)))
        e2_t = self.tconv2_ac(self.tconv2(torch.cat([e1_t, self.bridge1(e1_f, False)], dim=1)))
        e2_f = self.fconv2_ac(self.fconv2(torch.cat([e1_f, self.bridge1(e1_t, True)], dim=1)))
        e3_t = self.tconv3_ac(self.tconv3(torch.cat([e2_t, self.bridge2(e2_f, False)], dim=1)))
        e3_f = self.fconv3_ac(self.fconv3(torch.cat([e2_f, self.bridge2(e2_t, True)], dim=1)))
        e4_t = self.tconv4_ac(self.tconv4(torch.cat([e3_t, self.bridge3(e3_f, False)], dim=1)))
        e4_f = self.fconv4_ac(self.fconv4(torch.cat([e3_f, self.bridge3(e3_t, True)], dim=1)))
        e5_t = self.tconv5_ac(self.tconv5(torch.cat([e4_t, self.bridge4(e4_f, False)], dim=1)))
        e5_f = self.fconv5_ac(self.fconv5(torch.cat([e4_f, self.bridge4(e4_t, True)], dim=1)))
        e6_t = self.tconv6_ac(self.tconv6(torch.cat([e5_t, self.bridge5(e5_f, False)], dim=1)))
        e6_f = self.fconv6_ac(self.fconv6(torch.cat([e5_f, self.bridge5(e5_t, True)], dim=1)))

        inp_t = self.t_lstm_norm(e6_t)
        inp_f = self.f_lstm_norm(e6_f)
        lstm_t = self.t_lstm(inp_t)
        lstm_f = self.f_lstm(inp_f)

        # de tower
        d6_t = self.tconv6_t_ac(self.tconv6_t(e6_t * lstm_t))
        d6_f = self.fconv6_t_ac(self.fconv6_t(e6_f * lstm_f))
        d5_t = self.tconv5_t_ac(self.tconv5_t(torch.cat([d6_t, e5_t, self.bridge5(d6_f, False)], dim=1)))
        d5_f = self.fconv5_t_ac(self.fconv5_t(torch.cat([d6_f, e5_f, self.bridge5(d6_t, True)], dim=1)))
        d4_t = self.tconv4_t_ac(self.tconv4_t(torch.cat([d5_t, e4_t, self.bridge4(d5_f, False)], dim=1)))
        d4_f = self.fconv4_t_ac(self.fconv4_t(torch.cat([d5_f, e4_f, self.bridge4(d5_t, True)], dim=1)))
        d3_t = self.tconv3_t_ac(self.tconv3_t(torch.cat([d4_t, e3_t, self.bridge3(d4_f, False)], dim=1)))
        d3_f = self.fconv3_t_ac(self.fconv3_t(torch.cat([d4_f, e3_f, self.bridge3(d4_t, True)], dim=1)))
        d2_t = self.tconv2_t_ac(self.tconv2_t(torch.cat([d3_t, e2_t, self.bridge2(d3_f, False)], dim=1)))
        d2_f = self.fconv2_t_ac(self.fconv2_t(torch.cat([d3_f, e2_f, self.bridge2(d3_t, True)], dim=1)))
        d1_t = self.tconv1_t(torch.cat([d2_t, e1_t, self.bridge1(d2_f, False)], dim=1))
        d1_f = self.fconv1_t(torch.cat([d2_f, e1_f, self.bridge1(d2_t, True)], dim=1))

        wav = d1_t[:, 0, :, :]
        fft = d1_f[:, 0, :, :]

        est_wav_time = overlap_data(wav, siglen, self.frame_shift, deweight=True)
        est_wav_freq = self.get_isrs(fft, siglen)

        # loss
        loss_time = self.MSELoss(est_wav_time, label, nsamples)
        loss_freq = self.PCMLoss(est_wav_freq, label, mixture, nsamples)
        loss = loss_time + loss_freq
        return est_wav_freq, loss


if __name__ == '__main__':
    input = Variable(torch.FloatTensor(torch.rand(1, 16000))).cuda(0)
    nsamples = torch.IntTensor([16000]).cuda(0)
    net = Net(320, 160).cuda()
    macs, params = profile(net, inputs=(input, nsamples, input))
    macs, params = clever_format([macs, params], "%.3f")
    print("%s | %s | %s" % ('DBCN', params, macs))
