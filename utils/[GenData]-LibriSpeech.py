import random
import os
import time
from datetime import datetime
import timeit
import argparse
import h5py
import soundfile as sf
import numpy as np
from progressbar import progressbar as pb

def get_start_point(n_speech, n_noise, is_mc=False, is_train=True):
    if is_mc:
        return random.randint(0, n_noise - n_speech)
    else:
        if is_train:
            return random.randint(0, n_noise / 2 - n_speech)
        else:
            return random.randint(n_noise / 2, n_noise - n_speech)


########################### 1. Configurations
######### parse commands
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'])
FLAGS = parser.parse_args()

######### file paths
filelists_path = '/home/imu_zhangkanghao/ddn/project_data/SpeechEnhancement/filelists/'
train_speech_path = '/home/imu_zhangkanghao/ddn/common_data/LIBRI_datasets/train-other-500/'
test_speech_path = '/mnt/raid2/user_space/shenpengjie/data/LibriSpeech/test-clean/'
train_noise_path = '/mnt/raid2/user_space/shenpengjie/enhancement_data/enhancement_data_libri_nosilence/'
test_noise_path = '/mnt/raid/user_space/zhangkanghao/data/NOISE/'
test_mixture_path = '/mnt/raid/user_space/zhangkanghao/data/ns_data/LIBRI_TEST_SAMP/'
train_mixture_path = '/home/imu_zhangkanghao/ddn/project_data/SpeechEnhancement/LIBRI_TRAIN_SAMP/OTHER_500/'

######### settings
num_snr_per_utterance = 5
folder_cap = 5000  # each folder includes 5000 mixture files at most
num_jobs = 10
num_train_sentences = 28539

test_snr_list = [-5.0, -2.0, 0.0, 2.0, 5.0]
test_noise_list = ['ADTbabble.wav', 'ADTcafeteria.wav']
srate = 16000
random.seed(datetime.now())
###############################################
########################### 2. Generating mixtures
if FLAGS.mode == 'train':
    # sys.exit()
    speech_list = open(filelists_path + 'trainFileList_LIBRI_OTHER_500.txt', 'r')
    filelines_speech = speech_list.readlines()
    speech_list.close()
    print('%d sentences in total' % num_train_sentences)
    cnt = 0
    total_time = 0.
    for count in range(len(filelines_speech)):
        start = timeit.default_timer()
        # write all examples into an h5py file
        filefolder = '%d-%d/' % ((count // folder_cap) * folder_cap, (count // folder_cap + 1) * folder_cap - 1)
        filename = '%s_%d.samp' % (FLAGS.mode, count)
        if not os.path.isdir(train_mixture_path + filefolder):
            os.makedirs(train_mixture_path + filefolder)
        writer = h5py.File(train_mixture_path + filefolder + filename, 'w')

        random.seed(datetime.now())

        speech_name = filelines_speech[count].strip()
        s, srate_s = sf.read(train_speech_path + speech_name)
        if srate_s != 16000:
            raise ValueError('Invalid sample rate!')
        s = s / np.max(np.abs(s))

        writer.create_dataset('clean_raw', data=s.astype(np.float32), chunks=True)
        writer.close()
        cnt += 1
        end = timeit.default_timer()
        curr_time = end - start
        total_time += curr_time
        mtime = total_time / cnt
        print('{}/{}, time={:.4f}, mtime={:.4f}, etime={:.4f}'.format(count, len(filelines_speech), curr_time,
                                                                      mtime, mtime * len(filelines_speech)))
        print('Written into %s' % (filename))
    print('sleep for 3 secs...')
    time.sleep(3)
    f_train_list = open(filelists_path + 'trainList_LIBRI_OTHER_500.txt', 'w')
    for count in range(len(filelines_speech)):
        filefolder = '%d-%d/' % ((count // folder_cap) * folder_cap, (count // folder_cap + 1) * folder_cap - 1)
        filename = '%s_%d.samp' % (FLAGS.mode, count)
        f_train_list.write(train_mixture_path + filefolder + filename + '\n')
    f_train_list.close()
elif FLAGS.mode == 'test':
    if not os.path.isdir(test_mixture_path):
        os.makedirs(test_mixture_path)
    speech_list = open(filelists_path + 'testFileList_LIBRI_CLEAN.txt', 'r')

    filelines_speech = speech_list.readlines()
    speech_list.close()
    FLAGS.corpus = 'LIBRI_CLEAN'
    for noise_name in test_noise_list:
        print('Using %s noise' % (noise_name))
        # read noise
        n, srate_n = sf.read(test_noise_path + noise_name)
        # n = np.memmap(train_noise_path+train_noise,dtype=np.float32,mode='r')
        if len(n.shape) == 2:
            n = n[:, 0]
        print(n.shape)
        # sys.exit()
        # if srate_n != 16000:
        #    raise ValueError('Invalid sample rate!')
        for snr in test_snr_list:
            print('SNR level: %d dB' % snr)
            # write all examples into h5py files
            filename_mix = '%s_%s_snr%d_%s_mix.dat' % (FLAGS.mode, noise_name.split('.')[0], snr, FLAGS.corpus)
            filename_s = '%s_%s_snr%d_%s_s.dat' % (FLAGS.mode, noise_name.split('.')[0], snr, FLAGS.corpus)
            filename = '%s_%s_snr%d_%s.samp' % (FLAGS.mode, noise_name.split('.')[0], snr, FLAGS.corpus)

            f_mix = h5py.File(test_mixture_path + filename_mix, 'w')
            f_s = h5py.File(test_mixture_path + filename_s, 'w')
            writer = h5py.File(test_mixture_path + filename, 'w')
            # custom a progressbar
            widgets = ['[', Timer(), '](', ETA(), ')', Bar('='), '(', Percentage(), ')']
            bar = ProgressBar(widgets=widgets)
            for i in bar(range(len(filelines_speech))):
                speech_name = filelines_speech[i].strip()
                s, srate_s = sf.read(test_speech_path + speech_name)
                if srate_s != 16000:
                    raise ValueError('Invalid sample rate!')
                s = s / np.max(np.abs(s))
                # choose a point where we start to cut
                start_cut_point = random.randint(0, n.size - s.size)
                while np.sum(n[start_cut_point:start_cut_point + s.size] ** 2.0) == 0.0:
                    start_cut_point = random.randint(0, n.size - s.size)
                # cut noise
                n_t = n[start_cut_point:start_cut_point + s.size]
                # mixture = speech + noise
                alpha = np.sqrt(np.sum(s ** 2.0) / (np.sum(n_t ** 2.0) * (10.0 ** (snr / 10.0))))
                snr_check = 10.0 * np.log10(np.sum(s ** 2.0) / np.sum((n_t * alpha) ** 2.0))
                mix = s + alpha * n_t

                f_mix.create_dataset(str(i), data=mix.astype(np.float32), chunks=True)
                f_s.create_dataset(str(i), data=s.astype(np.float32), chunks=True)

                writer_grp = writer.create_group(str(i))
                writer_grp.create_dataset('noisy_raw', data=mix.astype(np.float32), chunks=True)
                writer_grp.create_dataset('clean_raw', data=s.astype(np.float32), chunks=True)

            f_mix.close()
            f_s.close()
            writer.close()

            print('Written into %s' % (filename))

            print('sleep for 3 secs...')
            time.sleep(3)
else:
    raise ValueError('Invalid mode!')

print('[%s] Finish generating mixtures.\n' % FLAGS.mode)
