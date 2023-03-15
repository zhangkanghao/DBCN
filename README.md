# A Dual-branch Convolutional Network Architecture Processing on both Frequency and Time domain for Single-channel Speech Enhancement

This repository provides an implementation of the Dual-Branch Convolutional Network for monaural speech enhancement, developed in ["A Dual-branch Convolutional Network Architecture Processing on both Frequency and Time domain for Single-channel Speech Enhancement"](https://blank), which was submitted to APSIPA transaction. In addition, the network is developed based on the pre-work ["DBNet: A Dual-branch Network Architecture Processing on Spectrum and Waveform for Single-channel Speech Enhancement"](https://www.isca-speech.org/archive/pdfs/interspeech_2021/zhang21s_interspeech.pdf), Interspeech Proceedings, pp. 2821-2825, 2021.Proceedings of Interspeech, pp. 2821-2825, 2021. In this work, a dual-banch convolutional network was proposed to perform cross-domain processing, which combines time-domain and frequency-domain. 

## Installation
The program is developed using Python 3.9.
Clone this repo, and install the dependencies:
```
git clone https://github.com/zhangkanghao/DBCN.git
cd DBCN
pip install -r requirements.txt
```

## Data preparation

Note: You will be required to download the WSJ0-SI84 dataset in advance! Or you can use LibriSpeed instead.

To use this program, data and file lists need to be prepared. If configured correctly, the directory tree should look like this:
```
.
├── DBCN_causal_master
│   ├── filelists
│   │   ├── assess_list.txt
│   │   ├── test_list.txt
│   │   └── train_list.txt
│   ├── logs
│   │   └── *.txt
│   ├── models
│   │   └── *.model
│   └── script
│       ├── config.yaml
│       ├── datasets.py
│       ├── metrics.py
│       ├── models.py
│       ├── networks.py
│       ├── runner.py
│       └── tool.py
├── DBCN_non_causal_master
│   ├── filelists
│   │   ├── assess_list.txt
│   │   ├── test_list.txt
│   │   └── train_list.txt
│   ├── logs
│   │   └── *.txt
│   ├── models
│   │   └── *.model
│   └── script
│       ├── config.yaml
│       ├── datasets.py
│       ├── metrics.py
│       ├── models.py
│       ├── networks.py
│       ├── runner.py
│       └── tool.py
├── README.md
└── requirements.txt
```




You will find that some files above are missing in your directory tree. Those are for you to prepare. Don't worry. Follow these instructions:
1. Write your own scripts or use utils/[GenData]-*.py to prepare data for training, validation and testing. 
    - For the training set, We save each example into an HDF5 file, which contains a HDF5 dataset, named ```clean_raw```. ```clean_raw``` stores a clean utterance. 
        - Example code:
          ```
          import os
          import h5py
          import numpy as np
          ...

          for idx in range(n_tr_ex): # n_tr_ex is the number of training examples 
              # save clean example into HDF5 file
              filefolder = '%d-%d/' % ((count // folder_cap) * folder_cap, (count // folder_cap + 1) * folder_cap - 1)
              filename = '%s_%d.samp' % (FLAGS.mode, count)
              ...
              sph, srate_s = sf.read(file_path)
              # normalize
              if srate_s != 16000:
                  raise ValueError('Invalid sample rate!')
              sph = sph / np.max(np.abs(sph))
          
              writer = h5py.File(os.path.join(filepath, filename), 'w')
              writer.create_dataset('clean_raw', data=sph.astype(np.float32), shape=mix.shape, chunks=True)
              writer.close()
          ```
        - We use online strategy to generate the mixture. You can also customize your own code in ```script/datasets```. You can edit ```script/datasets``` to use your own code.
        
    - For the validation set, 
        - We use the examples mixed with babble under -5dB, which is part of the test set.
    - For the test set(s), all examples (in each condition) need to be saved into a single HDF5 file, each of which is stored in a HDF5 group. Each group contains two HDF5 datasets, one named ```clean_raw``` and the other named ```noisy_raw```.
        - Example code:
          ```
          import os

          import h5py
          import numpy as np


          # some settings
          ...
          rms = 1.0
          
          filename = 'tt_snr-5.ex'
          writer = h5py.File(os.path.join(filepath, filename), 'w')
          for idx in range(n_cv_ex):
              # generate a noisy mixture
              ...
              mix = sph + noi
              # normalize
              alpha = np.sqrt(np.sum(sph ** 2.0) / (np.sum(noi ** 2.0) * (10.0 ** (snr / 10.0))))
              mix *= alpha
              sph *= alpha

              writer_grp = writer.create_group(str(count))
              writer_grp.create_dataset('noisy_raw', data=mix.astype(np.float32), shape=mix.shape, chunks=True)
              writer_grp.create_dataset('clean_raw', data=sph.astype(np.float32), shape=sph.shape, chunks=True)
          writer.close()
          ```
    - The training list will be automatically generated during the data set generation process.


## How to run
Take causal DBCN as example:
1. Change the directory: ```cd DBCN_causal_master```. Remember that this is your working directory. All paths and commands below are relative to it.
2. Check ```script/networks.py``` for the DBCN configurations. By default, ```G=2``` (see the original paper) is used for LSTM grouping.
3. Train the model: ```python script/runner.py```. By default, three directorys named ```logs|models|waves``` will be automatically generated. Many model files will be generated under ```models/```: ```model_epoch.model```(the model saved after N epochs), ```model_latest.model```(the model from the latest checkpoint) and ```model_best.model```(the model that performs best on the validation set by far). ```model_best.model``` can be used to resume training if interrupted, and ```model_best.model``` is typically used for testing. You can check the loss values in ```logs/loss_log.txt``` or check metric values in ```logs/metric_log.txt```. In addition, the file has four args: ```--mode=(default=train)```, ```--model=(default="")```, ```--config=(default="config.yaml")``` and ```--dir=(default=path-to-infer-data)```
4. Evaluate the model: ```python script/runner.py --mode=test --model=model_best.model```. WAV files will be generated under ```waves/predictions```. STOI, PESQ and SNR results will be written into the file under ```logs```: ```result.csv```.


## How to cite
```
@inproceedings{zhang21s_interspeech,
  author={Kanghao Zhang and Shulin He and Hao Li and Xueliang Zhang},
  title={{DBNet: A Dual-Branch Network Architecture Processing on Spectrum and Waveform for Single-Channel Speech Enhancement}},
  year=2021,
  booktitle={Proc. Interspeech 2021},
  pages={2821--2825},
  doi={10.21437/Interspeech.2021-1042}
}
```
```
@article{APSIPA transaction,
  to be update
}
```