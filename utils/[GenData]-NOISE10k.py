import os
import numpy as np
import h5py
import timeit

folder_cap = 5000
noise = np.memmap('/data_zkh/NOISE10K_datasets/noise10k.bin', dtype=np.float32, mode='r')
split_size = 50000
noise_dir = '/data_zkh/NOISE10K_datasets/NOISE10K_split_h5_50000/'
if not os.path.isdir(noise_dir):
    os.makedirs(noise_dir)
total_time = 0.
cnt = 1
print(noise.shape)
noise_len = noise.shape[0]
chunk_len = noise_len // split_size
start = 0
end = min(start + chunk_len, noise_len)
print(start, end)
for count in range(split_size):
    start1 = timeit.default_timer()
    filefolder = '%d-%d/' % ((count // folder_cap) * folder_cap, (count // folder_cap + 1) * folder_cap - 1)
    filename = 'noise_%d.samp' % (count)
    if not os.path.isdir(noise_dir + filefolder):
        os.makedirs(noise_dir + filefolder)
    writer = h5py.File(noise_dir + filefolder + filename, 'w')
    print(start, end)
    n_t = noise[start:end]
    print('noise', n_t.shape, start, end)
    writer.create_dataset('noise', data=n_t.astype(np.float32), chunks=True)
    writer.close()
    end1 = timeit.default_timer()
    curr_time = end1 - start1
    total_time += curr_time
    mtime = total_time / cnt
    print(count, split_size, curr_time, mtime, mtime * split_size / 60)
    cnt += 1
    start = end
    end = min(start + chunk_len, noise_len)
