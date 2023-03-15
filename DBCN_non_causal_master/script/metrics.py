import os
import re
import time
import h5py
import timeit
import numpy as np
import pandas as pd
from pystoi import stoi
from pypesq import pesq
from multiprocessing import Pool


# from matlab_eval import eval_composite


def normalize_wav(sig):
    scale = np.max(np.abs(sig)) + 1e-7
    sig = sig / scale

    return sig, scale


def si_snr(tgt, est):
    s_tgt = (
            np.sum(est * tgt, axis=-1, keepdims=True)
            * tgt
            / np.sum(tgt * tgt, axis=-1, keepdims=True)
    )
    e_ns = est - s_tgt
    ans = 10.0 * np.log10(np.sum(s_tgt * s_tgt, axis=-1) / np.sum(e_ns * e_ns, axis=-1))
    return ans


def snr(s, s_p):
    r""" calculate signal-to-noise ratio (SNR)

        Parameters
        ----------
        s: clean speech
        s_p: processed speech
    """
    return 10.0 * np.log10(np.sum(s ** 2) / np.sum((s_p - s) ** 2))


class Metric(object):
    def __init__(self, args):
        with open(args.conf['dataset']['assess_list'], "r") as assess_list_file:
            self.assess_list = []
            self.assess_len_list = []
            for line in assess_list_file:
                name, length = line.strip().split(" ")
                self.assess_list.append(name)
                self.assess_len_list.append(int(length))
        print("assess list:", self.assess_list)
        print("assess len list:", self.assess_len_list)

        self.test_path = args.conf['dataset']['test_path']
        self.predict_path = args.conf['output']['predict_path']
        self.srate = args.conf['models']['srate']
        self.frame_size = args.conf['models']['frame_size']
        self.frame_shift = args.conf['models']['frame_shift']
        self.is_causal = args.conf['models']['causal']

        self.output_csv_file = os.path.join(args.conf['output']['log_path'], "results.csv")
        self.train_corpus = 'WSJ'
        self.model_type = "CRN_MSE_STOI"
        self.feat_type = "WAVE"

        self.store_wave = True
        self.nthreads = 1
        self.df_dict = {}
        self.completed_count = 0

    def computeMetrics(self):
        df_list = []

        def update(*a):
            df = a[0]
            df_list.append(df)

        start = timeit.default_timer()
        pool = Pool(processes=self.nthreads)
        for i in range(len(self.assess_list)):
            pool.apply_async(self.process, args=(i,), callback=update)
        pool.close()
        pool.join()

        final_df = pd.concat(df_list)
        final_df["TrainCorpus"] = self.train_corpus
        final_df["ModelType"] = self.model_type
        final_df["FeatType"] = self.feat_type
        final_df["FrameSize"] = self.frame_size
        final_df["FrameShift"] = self.frame_shift
        final_df["IsCausal"] = self.is_causal
        final_df.to_csv(self.output_csv_file)
        # print(final_df)
        print("Unprocessed stats ...")
        print(
            final_df[
                [
                    "TestCorpus",
                    "TestNoise",
                    "TestSNR",
                    "MIX_STOI",
                    "MIX_PESQ_NB",
                    "MIX_SISNR",
                ]
            ]
                .groupby(["TestNoise", "TestCorpus", "TestSNR"])
                .mean()
        )
        print("")
        print("Processed stats ...")
        print(
            final_df[["TestCorpus", "TestNoise", "TestSNR", "EST_STOI", "EST_PESQ_NB"]]
                .groupby(["TestNoise", "TestCorpus", "TestSNR"])
                .mean()
        )
        print("")
        end = timeit.default_timer()
        print("Completed. Time taken: {:.2f} seconds.".format(end - start))

    def process(self, i):
        print(self.assess_list[i])
        # m = re.search("(.+)/test_(.+)_snr(-*[0-9]+)_(.+)$", self.assess_list[i])
        # test_corpus = m.group(1)
        # test_noise = m.group(2)
        # test_snr = m.group(3)
        m = re.search("test_(.+)_snr(-*[0-9]+)_(.+)$", self.assess_list[i])
        test_noise = m.group(1)
        test_snr = m.group(2)
        test_corpus = m.group(3)

        print(test_corpus, test_noise, test_snr)
        start_time = time.time()
        print("")
        print(
            "{}/{}, Started assess on {} ...".format(
                i + 1, len(self.assess_list), self.assess_list[i]
            )
        )
        print("")
        f_mix = h5py.File(os.path.join(self.test_path, self.assess_list[i] + "_mix.dat"), "r")
        f_s = h5py.File(os.path.join(self.test_path, self.assess_list[i] + "_s.dat"), "r")

        filepath = os.path.join(
            self.predict_path, self.assess_list[i] + "_s_est.dat"
        )
        filename = os.path.basename(filepath)
        dirname = os.path.dirname(filepath)
        filepath = os.path.join(dirname, filename)
        f_est_s = h5py.File(filepath, "r")

        filepath = os.path.join(
            self.predict_path, self.assess_list[i] + "_s_ideal.dat"
        )
        filename = os.path.basename(filepath)
        dirname = os.path.dirname(filepath)
        filepath = os.path.join(dirname, filename)
        f_ideal_s = h5py.File(filepath, "r")

        # create arrays for stoi and snr
        est_stoi_s_accu = 0.0
        ideal_stoi_s_accu = 0.0
        mix_stoi_s_accu = 0.0

        est_estoi_s_accu = 0.0
        ideal_estoi_s_accu = 0.0
        mix_estoi_s_accu = 0.0

        est_pesq_nb_s_accu = 0.0
        ideal_pesq_nb_s_accu = 0.0
        mix_pesq_nb_s_accu = 0.0

        est_pesq_wb_s_accu = 0.0
        ideal_pesq_wb_s_accu = 0.0
        mix_pesq_wb_s_accu = 0.0

        est_snr_s_accu = 0.0
        ideal_snr_s_accu = 0.0
        mix_snr_s_accu = 0.0

        est_csig_s_accu = 0.0
        ideal_csig_s_accu = 0.0
        mix_csig_s_accu = 0.0

        est_cbak_s_accu = 0.0
        ideal_cbak_s_accu = 0.0
        mix_cbak_s_accu = 0.0

        est_covl_s_accu = 0.0
        ideal_covl_s_accu = 0.0
        mix_covl_s_accu = 0.0

        mdict_list = []
        for j in range(self.assess_len_list[i]):
            mix = f_mix[str(j)][:]
            s = f_s[str(j)][:]
            est_s = f_est_s[str(j)][:]
            ideal_s = f_ideal_s[str(j)][:]

            mix = mix  # / np.max(np.abs(mix))
            s = s  # / np.max(np.abs(s))
            est_s = est_s  # / np.max(np.abs(est_s))
            ideal_s = ideal_s  # / np.max(np.abs(ideal_s))
            # print(j, 'mix', np.max(np.abs(mix)), 's', np.max(np.abs(s)), 'est_s', np.max(np.abs(est_s)), 'ideal_s', np.max(np.abs(ideal_s)))

            # compute stoi
            est_stoi_s = stoi(s, est_s, int(self.srate), extended=False)
            ideal_stoi_s = stoi(s, ideal_s, int(self.srate), extended=False)
            mix_stoi_s = stoi(s, mix, int(self.srate), extended=False)

            # compute estoi
            est_estoi_s = stoi(s, est_s, int(self.srate), extended=True)
            ideal_estoi_s = stoi(s, ideal_s, int(self.srate), extended=True)
            mix_estoi_s = stoi(s, mix, int(self.srate), extended=True)

            # # compute pesq
            est_pesq_nb_s = pesq(s, est_s, int(self.srate))
            ideal_pesq_nb_s = pesq(s, ideal_s, int(self.srate))
            mix_pesq_nb_s = pesq(s, mix, int(self.srate))

            # compute si_snr
            est_snr_s = si_snr(s, est_s)
            ideal_snr_s = si_snr(s, ideal_s)
            mix_snr_s = si_snr(s, mix)

            # mix_d = eval_composite(s, mix)
            # est_d = eval_composite(s, est_s)
            # ideal_d = eval_composite(s, ideal_s)

            mix_pesq_wb_s = 0  # mix_d["pesq"]
            est_pesq_wb_s = 0  # est_d["pesq"]
            ideal_pesq_wb_s = 0  # ideal_d["pesq"]

            mix_csig_s = 0  # mix_d["csig"]
            est_csig_s = 0  # est_d["csig"]
            ideal_csig_s = 0  # ideal_d["csig"]

            mix_cbak_s = 0  # mix_d["cbak"]
            est_cbak_s = 0  # est_d["cbak"]
            ideal_cbak_s = 0  # ideal_d["cbak"]

            mix_covl_s = 0  # mix_d["covl"]
            est_covl_s = 0  # est_d["covl"]
            ideal_covl_s = 0  # ideal_d["covl"]

            # region: print info per sentence
            # print('#' * 5)
            # print('[{}, {}, {}dB], {}/{}, mix_stoi={:.4f}, ideal_stoi={:.4f}, est_stoi={:.4f}'.format(test_corpus,
            #                                                                                           test_noise,
            #                                                                                           test_snr,
            #                                                                                           j + 1,
            #                                                                                           self.assess_len_list[
            #                                                                                               i],
            #                                                                                           mix_stoi_s,
            #                                                                                           ideal_stoi_s,
            #                                                                                           est_stoi_s))

            # print('[{}, {}, {}dB], {}/{}, mix_estoi={:.4f}, ideal_estoi={:.4f}, est_estoi={:.4f}'.format(test_corpus,
            #                                                                                              test_noise,
            #                                                                                              test_snr,
            #                                                                                              j + 1,
            #                                                                                              self.assess_len_list[
            #                                                                                                  i],
            #                                                                                              mix_estoi_s,
            #                                                                                              ideal_estoi_s,
            #                                                                                              est_estoi_s))

            # print('[{}, {}, {}dB], {}/{}, mix_pesq_nb={:.2f}, ideal_pesq_nb={:.2f}, est_pesq_nb={:.2f}'.format(test_corpus,
            #                                                                                                    test_noise,
            #                                                                                                    test_snr,
            #                                                                                                    j + 1,
            #                                                                                                    self.assess_len_list[
            #                                                                                                        i],
            #                                                                                                    mix_pesq_nb_s,
            #                                                                                                    ideal_pesq_nb_s,
            #                                                                                                    est_pesq_nb_s))

            # print('[{}, {}, {}dB], {}/{}, mix_pesq_wb = {:.2f}, ideal_pesq_wb={:.2f}, est_pesq_wb={:.2f}'.format(test_corpus,
            #                                                                                                      test_noise,
            #                                                                                                      test_snr,
            #                                                                                                      j + 1,
            #                                                                                                      self.assess_len_list[
            #                                                                                                          i],
            #                                                                                                      mix_pesq_wb_s,
            #                                                                                                      ideal_pesq_wb_s,
            #                                                                                                      est_pesq_wb_s))

            # print('[{}, {}, {}dB], {}/{}, mix_snr={:.1f}, ideal_snr={:.1f}, est_snr={:.1f}'.format(test_corpus,
            #                                                                                        test_noise,
            #                                                                                        test_snr,
            #                                                                                        j+1, self.assess_len_list[i],
            #                                                                                        mix_snr_s,
            #                                                                                        ideal_snr_s,
            #                                                                                        est_snr_s))

            # print('[{}, {}, {}dB], {}/{}, mix_csig={:.2f}, ideal_csig={:.2f}, est_csig={:.2f}'.format(test_corpus,
            #                                                                                           test_noise,
            #                                                                                           test_snr,
            #                                                                                           j+1, self.assess_len_list[i],
            #                                                                                           mix_csig_s,
            #                                                                                           ideal_csig_s,
            #                                                                                           est_csig_s))

            # print('[{}, {}, {}dB], {}/{}, mix_cbak={:.2f}, ideal_cbak={:.2f}, est_cbak={:.2f}'.format(test_corpus,
            #                                                                                           test_noise,
            #                                                                                           test_snr,
            #                                                                                           j+1, self.assess_len_list[i],
            #                                                                                           mix_cbak_s,
            #                                                                                           ideal_cbak_s,
            #                                                                                           est_cbak_s))

            # print('[{}, {}, {}dB], {}/{}, mix_covl={:.2f}, ideal_covl={:.2f}, est_covl={:.2f}'.format(test_corpus,
            #                                                                                           test_noise,
            #                                                                                           test_snr,
            #                                                                                           j+1, self.assess_len_list[i],
            #                                                                                           mix_covl_s,
            #                                                                                           ideal_covl_s,
            #                                                                                           est_covl_s))
            # endregion

            mdict = {}
            mdict["MIX_STOI"] = mix_stoi_s * 100
            mdict["MIX_ESTOI"] = mix_estoi_s * 100
            mdict["MIX_PESQ_NB"] = mix_pesq_nb_s
            mdict["MIX_PESQ_WB"] = mix_pesq_wb_s
            mdict["MIX_SISNR"] = mix_snr_s
            mdict["MIX_CSIG"] = mix_csig_s
            mdict["MIX_CBAK"] = mix_cbak_s
            mdict["MIX_COVL"] = mix_covl_s

            mdict["EST_STOI"] = est_stoi_s * 100
            mdict["EST_ESTOI"] = est_estoi_s * 100
            mdict["EST_PESQ_NB"] = est_pesq_nb_s
            mdict["EST_PESQ_WB"] = est_pesq_wb_s
            mdict["EST_SISNR"] = est_snr_s
            mdict["EST_CSIG"] = est_csig_s
            mdict["EST_CBAK"] = est_cbak_s
            mdict["EST_COVL"] = est_covl_s
            mdict_list.append(mdict)

            # compute accu of them
            est_stoi_s_accu += est_stoi_s
            ideal_stoi_s_accu += ideal_stoi_s
            mix_stoi_s_accu += mix_stoi_s

            est_estoi_s_accu += est_estoi_s
            ideal_estoi_s_accu += ideal_estoi_s
            mix_estoi_s_accu += mix_estoi_s

            est_pesq_nb_s_accu += est_pesq_nb_s
            ideal_pesq_nb_s_accu += ideal_pesq_nb_s
            mix_pesq_nb_s_accu += mix_pesq_nb_s

            est_pesq_wb_s_accu += est_pesq_wb_s
            ideal_pesq_wb_s_accu += ideal_pesq_wb_s
            mix_pesq_wb_s_accu += mix_pesq_wb_s

            est_snr_s_accu += est_snr_s
            ideal_snr_s_accu += ideal_snr_s
            mix_snr_s_accu += mix_snr_s

            est_csig_s_accu += est_csig_s
            ideal_csig_s_accu += ideal_csig_s
            mix_csig_s_accu += mix_csig_s

            est_cbak_s_accu += est_cbak_s
            ideal_cbak_s_accu += ideal_cbak_s
            mix_cbak_s_accu += mix_cbak_s

            est_covl_s_accu += est_covl_s
            ideal_covl_s_accu += ideal_covl_s
            mix_covl_s_accu += mix_covl_s

        # bar.finish()
        f_mix.close()
        f_s.close()
        f_est_s.close()
        f_ideal_s.close()
        df = pd.DataFrame(mdict_list)
        df["TestCorpus"] = test_corpus
        df["TestNoise"] = test_noise
        df["TestSNR"] = test_snr

        mix_stoi_s_ave = mix_stoi_s_accu / float(self.assess_len_list[i])
        ideal_stoi_s_ave = ideal_stoi_s_accu / float(self.assess_len_list[i])
        est_stoi_s_ave = est_stoi_s_accu / float(self.assess_len_list[i])

        mix_estoi_s_ave = mix_estoi_s_accu / float(self.assess_len_list[i])
        ideal_estoi_s_ave = ideal_estoi_s_accu / float(self.assess_len_list[i])
        est_estoi_s_ave = est_estoi_s_accu / float(self.assess_len_list[i])

        mix_pesq_nb_s_ave = mix_pesq_nb_s_accu / float(self.assess_len_list[i])
        ideal_pesq_nb_s_ave = ideal_pesq_nb_s_accu / float(self.assess_len_list[i])
        est_pesq_nb_s_ave = est_pesq_nb_s_accu / float(self.assess_len_list[i])

        mix_pesq_wb_s_ave = mix_pesq_wb_s_accu / float(self.assess_len_list[i])
        ideal_pesq_wb_s_ave = ideal_pesq_wb_s_accu / float(self.assess_len_list[i])
        est_pesq_wb_s_ave = est_pesq_wb_s_accu / float(self.assess_len_list[i])

        mix_snr_s_ave = mix_snr_s_accu / float(self.assess_len_list[i])
        ideal_snr_s_ave = ideal_snr_s_accu / float(self.assess_len_list[i])
        est_snr_s_ave = est_snr_s_accu / float(self.assess_len_list[i])

        mix_csig_s_ave = mix_csig_s_accu / float(self.assess_len_list[i])
        ideal_csig_s_ave = ideal_csig_s_accu / float(self.assess_len_list[i])
        est_csig_s_ave = est_csig_s_accu / float(self.assess_len_list[i])

        mix_cbak_s_ave = mix_cbak_s_accu / float(self.assess_len_list[i])
        ideal_cbak_s_ave = ideal_cbak_s_accu / float(self.assess_len_list[i])
        est_cbak_s_ave = est_cbak_s_accu / float(self.assess_len_list[i])

        mix_covl_s_ave = mix_covl_s_accu / float(self.assess_len_list[i])
        ideal_covl_s_ave = ideal_covl_s_accu / float(self.assess_len_list[i])
        est_covl_s_ave = est_covl_s_accu / float(self.assess_len_list[i])

        end_time = time.time()
        print("")
        print(
            "{}/{}, Finished assess on {}. time taken = {:.4f}".format(
                i + 1, len(self.assess_list), self.assess_list[i], end_time - start_time
            )
        )
        print("")

        print("#" * 100)
        print("#" * 100)
        print(
            "[{}, {}, {}dB], mix_stoi: {:.4f} | ideal_stoi: {:.4f} | est_stoi: {:.4f}".format(
                test_corpus,
                test_noise,
                test_snr,
                mix_stoi_s_ave,
                ideal_stoi_s_ave,
                est_stoi_s_ave,
            )
        )
        print(
            "[{}, {}, {}dB], mix_estoi: {:.4f} | ideal_estoi: {:.4f} | est_estoi: {:.4f}".format(
                test_corpus,
                test_noise,
                test_snr,
                mix_estoi_s_ave,
                ideal_estoi_s_ave,
                est_estoi_s_ave,
            )
        )
        print(
            "[{}, {}, {}dB], mix_pesq_nb: {:.4f} | ideal_pesq_nb: {:.4f} | est_pesq_nb: {:.4f}".format(
                test_corpus,
                test_noise,
                test_snr,
                mix_pesq_nb_s_ave,
                ideal_pesq_nb_s_ave,
                est_pesq_nb_s_ave,
            )
        )
        print(
            "[{}, {}, {}dB], mix_pesq_wb: {:.4f} | ideal_pesq_wb: {:.4f} | est_pesq_wb: {:.4f}".format(
                test_corpus,
                test_noise,
                test_snr,
                mix_pesq_wb_s_ave,
                ideal_pesq_wb_s_ave,
                est_pesq_wb_s_ave,
            )
        )
        print(
            "[{}, {}, {}dB], mix_snr: {:.2f} | ideal_snr: {:.2f} | est_snr: {:.2f}".format(
                test_corpus,
                test_noise,
                test_snr,
                mix_snr_s_ave,
                ideal_snr_s_ave,
                est_snr_s_ave,
            )
        )
        print(
            "[{}, {}, {}dB], mix_csig: {:.2f} | ideal_csig: {:.2f} | est_csig: {:.2f}".format(
                test_corpus,
                test_noise,
                test_snr,
                mix_csig_s_ave,
                ideal_csig_s_ave,
                est_csig_s_ave,
            )
        )
        print(
            "[{}, {}, {}dB], mix_cbak: {:.2f} | ideal_cbak: {:.2f} | est_cbak: {:.2f}".format(
                test_corpus,
                test_noise,
                test_snr,
                mix_cbak_s_ave,
                ideal_cbak_s_ave,
                est_cbak_s_ave,
            )
        )
        print(
            "[{}, {}, {}dB], mix_covl: {:.2f} | ideal_covl: {:.2f} | est_covl: {:.2f}".format(
                test_corpus,
                test_noise,
                test_snr,
                mix_covl_s_ave,
                ideal_covl_s_ave,
                est_covl_s_ave,
            )
        )
        print("#" * 100)
        print("#" * 100)
        return df
