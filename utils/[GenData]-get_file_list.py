import os
import random

def gen_speech_list(speech_dir):
    wav_l = []
    for root, dirs, files in os.walk(speech_dir, followlinks=True):
        for f in files:
            f = os.path.join(root, f)
            ext = os.path.splitext(f)[1]
            if ext=='.flac':
                wav_l.append(f)
    return wav_l


train_dir = '/home/imu_zhangkanghao/ddn/common_data/LIBRI_datasets/train-other-500'
test_dir = train_dir
all_files = gen_speech_list(train_dir)
print(len(all_files))
random.shuffle(all_files)
train_files = all_files

print(len(train_files))
with open('/home/imu_zhangkanghao/ddn/project_data/SpeechEnhancement/filelists/trainFileList_LIBRI_OTHER_500.txt', 'w') as f:
    for path in train_files:
        s = path.replace(train_dir, '')
        print(s)
        f.write(s+'\n')
