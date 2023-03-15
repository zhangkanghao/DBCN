import os
import timeit
import h5py
import numpy as np
import torch
import time
import soundfile as sf
from pystoi import stoi
from pypesq import pesq
from metrics import normalize_wav, snr
from networks import Net
from datasets import TrainingDataset, TrainCollate, EvalDataset, EvalCollate
from tools import gen_list, Checkpoint, ProgressBar
from torch import distributed as dist
from torch.utils.data import DistributedSampler, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP


class Model(object):
    def __init__(self, args):
        self.args = args
        self.conf = args.conf
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def setup(self, rank, world_size):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12388"
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        self.device = torch.device("cuda:{}".format(rank))
        print(self.device)

    def cleanup(self):
        dist.destroy_process_group()

    def train(self, args):
        self.setup(args.rank, self.conf["num_gpu"])

        # region: Initialize Dataset
        with open(self.conf["dataset"]["train_list"], "r") as train_list_file:
            self.train_list = [line.strip() for line in train_list_file.readlines()]

        trainSet = TrainingDataset(self.train_list, self.conf["dataset"])
        train_sampler = DistributedSampler(
            trainSet,
            num_replicas=self.conf["num_gpu"],
            rank=args.rank,
            shuffle=True,
            drop_last=True
        )
        train_loader = DataLoader(
            trainSet,
            batch_size=self.conf["dataset"]["batch_size"],
            shuffle=False,
            num_workers=self.conf["dataset"]["num_workers"],
            drop_last=True,
            pin_memory=True,
            collate_fn=TrainCollate(),
            sampler=train_sampler,
            prefetch_factor=self.conf["dataset"]["prefetch_factor"],
            persistent_workers=True
        )
        if args.rank == 0:
            evalSet = EvalDataset(self.conf["dataset"]["eval_file"], self.conf["dataset"]["num_eval_sentences"])
            eval_loader = DataLoader(
                evalSet,
                batch_size=self.conf["dataset"]["batch_size"] // 2,
                shuffle=False,
                num_workers=self.conf["dataset"]["num_workers"],
                drop_last=False,
                pin_memory=True,
                collate_fn=EvalCollate(),
                persistent_workers=True,
            )
        # endregion

        # region: Initialize Network and Loss
        network = Net(self.conf["models"]["frame_size"], self.conf["models"]["frame_shift"])
        network = network.to(self.device)
        network = DDP(
            network,
            device_ids=[args.rank],
            output_device=args.rank,
            find_unused_parameters=False,
        )
        criterion = None
        # endregion

        # region: Initialize Optimizer, Scaler and Scheduler
        optimizer = torch.optim.Adam(network.parameters(), lr=self.conf["models"]["lr"])
        scaler = torch.cuda.amp.GradScaler()
        lr_list = [0.0002] * 6 + [0.0001] * 12 + [0.00005] * 6 + [0.00001] * 6
        # endregion

        # region: Load Pretrained Model
        if len(args.model) > 0:
            print('Resume model from "%s"' % args.model)
            checkpoint = Checkpoint()
            checkpoint.load(os.path.join(self.conf["output"]["model_path"], args.model))
            start_epoch = checkpoint.start_epoch
            best_loss = checkpoint.best_loss
            network.load_state_dict(checkpoint.state_dict)
            optimizer.load_state_dict(checkpoint.optimizer)
            scaler.load_state_dict(checkpoint.scaler)
        else:
            self.logging("Training from scratch.", log_type='info')
            start_epoch, best_loss = 0, np.inf
        # endregion

        # region: Print Variables
        # print([(i, lr_list[i]) for i in range(len(lr_list))])
        num_train_sentences = len(trainSet)
        num_train_batches = num_train_sentences // (self.conf["dataset"]["batch_size"] * self.conf["num_gpu"])
        total_train_batch = self.conf["models"]["max_epoch"] * num_train_batches
        print("Number of learnable parameters: {}".format(self.num_params(network)))
        print("num_train_sentences", num_train_sentences)
        print("batches_per_epoch", num_train_batches)
        print("total_train_batch", total_train_batch)
        print("local_batch_size", self.conf["dataset"]["batch_size"])
        print("total_batch_size", self.conf["dataset"]["batch_size"] * self.conf["num_gpu"])
        print("nsamples", trainSet.nsamples)

        # endregion

        # region: Training
        print("Start Traning.")
        # region: ValidateWithMetric Before Training
        dist.barrier()
        if args.rank == 0 and len(args.model) == 0:
            start_time = time.time()
            print("Before training the performance on validationis as follows:")
            mdict = self.validate_with_metrics(network, eval_loader)
            print(mdict)
            end_time = time.time()
            print("Time:{:.4f}".format(end_time - start_time))
        # endregion
        dist.barrier()
        for epoch in range(start_epoch, self.conf["models"]["max_epoch"]):
            cnt, mtime = 0, 0
            # region: Train One Epoch
            accu_train_loss = 0.0
            network.train()
            train_sampler.set_epoch(epoch)
            for param_group in optimizer.param_groups:
                curr_lr = param_group["lr"]
                param_group["lr"] = lr_list[epoch]
            train_bar = ProgressBar(0, len(train_loader), 20)
            train_bar.start()
            start_time = time.time()
            for i, (features, labels, nsamples) in enumerate(train_loader):
                features = features.to(self.device)
                labels = labels.to(self.device)
                nsamples = nsamples.to(self.device)
                for param in network.parameters():
                    param.grad = None

                # Forward propagation
                with torch.cuda.amp.autocast():
                    est, loss = network(features, nsamples, labels)

                # Back propagation
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_value_(network.parameters(), self.conf["models"]["grad_value"])
                scaler.step(optimizer)
                scaler.update()

                # Print Infomation
                running_loss = loss.data.item()
                accu_train_loss += running_loss
                del features, labels, nsamples, loss
                for param_group in optimizer.param_groups:
                    curr_lr = param_group["lr"]
                end_time = time.time()
                cur_time = end_time - start_time
                mtime += cur_time
                cnt += 1
                train_bar.update_progress(
                    progress=i,
                    prefix_message="Train",
                    suffix_message="epoch={}/{}, lr={:.8f}, loss={:.5f}, time/mtime={:.5f}/{:.5f}\n".format(
                        epoch + 1,
                        self.conf["models"]["max_epoch"],
                        curr_lr,
                        running_loss,
                        cur_time,
                        mtime / cnt
                    )
                )
                start_time = time.time()
            # endregion

            # region: Validate One Epoch
            if args.rank == 0:
                print("#" * 50)
                start_time = time.time()
                avg_train_loss = accu_train_loss / cnt
                avg_eval_loss = self.validate(network, eval_loader, criterion)
                self.logging('Epoch [%d/%d], ( TrainLoss: %.4f | EvalLoss: %.4f )' % (
                    epoch + 1,
                    self.conf["models"]["max_epoch"],
                    avg_train_loss,
                    avg_eval_loss
                ), log_type='info')
                is_best = (avg_eval_loss < best_loss)
                best_loss = avg_eval_loss if is_best else best_loss
                checkpoint = Checkpoint(
                    start_epoch=epoch + 1,
                    train_loss=avg_train_loss,
                    eval_loss=avg_eval_loss,
                    best_loss=best_loss,
                    state_dict=network.state_dict(),
                    optimizer=optimizer.state_dict(),
                    scaler=scaler.state_dict(),
                )
                checkpoint.save(
                    is_best,
                    os.path.join(self.conf["output"]["model_path"], "model_latest.model"),
                    os.path.join(self.conf["output"]["model_path"], "model_best.model"),
                )
                if self.conf["save_all"]:
                    checkpoint.save(
                        False,
                        os.path.join(self.conf["output"]["model_path"], "model_{}.model".format(epoch + 1)),
                        os.path.join(self.conf["output"]["model_path"], "model_best.model"),
                    )

                self.logging(checkpoint, None, 'loss')
                end_time = time.time()
                print("Time:{:.4f}".format(end_time - start_time))
                print("#" * 50)
            # endregion

            # region: Validate With Metric One Epoch
            if (epoch + 1) % self.conf["models"]["eval_epoch"] == 0 and args.rank == 0:
                print('********** Started metrics evaluation on validation set ********')
                start_time = time.time()
                mdict = self.validate_with_metrics(network, eval_loader)
                print("After {} epoch the performance on validation set is as follows:".format(epoch + 1))
                print(mdict)
                print("")

                self.logging(epoch + 1, mdict, 'metric')
                end_time = time.time()
                print("Time:{:.4f}".format(end_time - start_time))
                print("#" * 50)
            # endregion

            dist.barrier()
            # scheduler.step(epoch)
        # endregion
        self.cleanup()

    def validate(self, net, eval_loader, criterion):
        net.eval()
        with torch.no_grad():
            cnt = 0
            accu_eval_loss = 0.0
            eval_bar = ProgressBar(0, len(eval_loader), 20)
            eval_bar.start()
            for j, (noisy_batch, clean_batch, nsamples) in enumerate(eval_loader):
                noisy_batch = noisy_batch.to(self.device)
                clean_batch = clean_batch.to(self.device)
                nsamples = nsamples.to(self.device)
                with torch.cuda.amp.autocast():
                    est, loss = net(noisy_batch, nsamples, clean_batch)
                accu_eval_loss += loss.item()
                cnt += 1
                eval_bar.update_progress(
                    progress=j,
                    prefix_message='Eval ',
                    suffix_message='loss={:.5f}/{:.5f}'.format(
                        loss,
                        accu_eval_loss / cnt
                    )
                )
            eval_bar.finish()
            avg_eval_loss = accu_eval_loss / cnt
        print()
        net.train()
        return avg_eval_loss

    def validate_with_metrics(self, net, eval_loader):
        net.eval()
        mdict = {
            "stoi": 0,
            "estoi": 0,
            "pesq_nb": 0,
            "pesq_wb": 0,
            "snr": 0,
            "csig": 0,
            "cbak": 0,
            "covl": 0,
            "count": 0,
        }

        def update(*a):
            curr_d = a[0]
            mdict["stoi"] += curr_d["stoi"]
            mdict["estoi"] += curr_d["estoi"]
            mdict["pesq_nb"] += curr_d["pesq_nb"]
            mdict["pesq_wb"] += curr_d["pesq"]
            mdict["snr"] += curr_d["snr"]
            mdict["csig"] += curr_d["csig"]
            mdict["cbak"] += curr_d["cbak"]
            mdict["covl"] += curr_d["covl"]
            mdict["count"] += 1

        # pool = Pool(processes=8)
        with torch.no_grad():
            eval_bar = ProgressBar(0, self.conf["dataset"]["num_eval_sentences"], 20)
            eval_bar.start()
            eval_count = 0
            for _, (noisy_batch, clean_batch, nsamples) in enumerate(eval_loader):
                noisy_batch = noisy_batch.to(self.device)
                clean_batch = clean_batch.to(self.device)
                nsamples = nsamples.to(self.device)
                for j in range(noisy_batch.shape[0]):
                    noisy_batch_ = noisy_batch[j: j + 1, :, : nsamples[j]]
                    clean_batch_ = clean_batch[j: j + 1, :, : nsamples[j]]
                    nsamples_ = nsamples[j: j + 1]
                    with torch.cuda.amp.autocast():
                        est_batch_ = net(noisy_batch_, nsamples_, clean_batch_)
                    est_s = est_batch_[0][0].float().cpu().numpy()
                    ideal_s = clean_batch_[0][0].float().cpu().numpy()
                    mix_s = noisy_batch_[0][0].float().cpu().numpy()
                    # pool.apply_async(
                    #     self.compute_metrics, args=(ideal_s, est_s), callback=update
                    # )
                    dct = self.compute_metrics(ideal_s, est_s)
                    update(dct)
                    eval_bar.update_progress(
                        progress=eval_count,
                        prefix_message='Metric',
                        suffix_message="",
                    )
                    eval_count += 1
                    if eval_count < 10:
                        sf.write(os.path.join(self.conf["output"]["valiate_path"], "est_{}.wav".format(eval_count)),
                                 normalize_wav(est_s)[0], self.conf["models"]["srate"])
                        sf.write(os.path.join(self.conf["output"]["valiate_path"], "tgt_{}.wav".format(eval_count)),
                                 normalize_wav(ideal_s)[0], self.conf["models"]["srate"])
                        sf.write(os.path.join(self.conf["output"]["valiate_path"], "mix_{}.wav".format(eval_count)),
                                 normalize_wav(mix_s)[0], self.conf["models"]["srate"])
            eval_bar.finish()
        # pool.close()
        # pool.join()
        net.train()
        for key in mdict.keys():
            if key != "count":
                mdict[key] = mdict[key] / mdict["count"]
        mdict.pop("count")
        return mdict

    def test(self):
        samp_list = gen_list(self.conf["dataset"]["test_path"], '.samp')
        net = Net(self.conf["models"]["frame_size"], self.conf["models"]["frame_shift"]).to(self.device)
        net = torch.nn.DataParallel(net)
        checkpoint = Checkpoint()
        checkpoint.load(os.path.join(self.conf["output"]["model_path"], self.args.model))
        net.load_state_dict(checkpoint.state_dict)
        net.eval()

        score_stois = {}
        score_snrs = {}
        score_pesqs = {}
        print('#' * 18 + 'Finish Resume Model For Test ' + '#' * 18)
        for i in range(len(samp_list)):
            filename_input = samp_list[i]
            elements = filename_input.split('_')
            noise_type, snr_value = elements[1], elements[2]
            print('{}/{}, Started working on {}.'.format(i + 1, len(samp_list), samp_list[i]))
            f_mix = h5py.File(os.path.join(self.conf["dataset"]["test_path"], filename_input), 'r')
            ttime, mtime, cnt = 0., 0., 0.
            acc_stoi_mix, acc_snr_mix, acc_pesq_mix = 0., 0., 0.
            acc_stoi_est, acc_snr_est, acc_pesq_est = 0., 0., 0.
            num_clips = len(f_mix)
            for k in range(num_clips):
                start = timeit.default_timer()
                reader_grp = f_mix[str(k)]
                mix = reader_grp['noisy_raw'][:]
                label = reader_grp['clean_raw'][:]
                mix_tensor = torch.from_numpy(mix).reshape(1, 1, -1)
                tgt_tensor = torch.from_numpy(label).reshape(1, 1, -1)
                nsamples = torch.IntTensor([mix_tensor.shape[-1:]])
                est, _ = net(mix_tensor, nsamples, tgt_tensor)
                est = est.cpu().detach().numpy()[0]
                mix = mix[:est.size]
                label = label[:est.size]

                mix_stoi = stoi(label, mix, self.conf["models"]["srate"])
                est_stoi = stoi(label, est, self.conf["models"]["srate"])
                acc_stoi_mix += mix_stoi
                acc_stoi_est += est_stoi

                mix_snr = snr(label, mix)
                est_snr = snr(label, est)
                acc_snr_mix += mix_snr
                acc_snr_est += est_snr

                mix_pesq = pesq(label, mix, self.conf["models"]["srate"])
                est_pesq = pesq(label, est, self.conf["models"]["srate"])
                acc_pesq_mix += mix_pesq
                acc_pesq_est += est_pesq

                cnt += 1
                end = timeit.default_timer()
                curr_time = end - start
                ttime += curr_time
                mtime = ttime / cnt
                print('{}, stoi={}, pesq={}, snr={}, ctime/mtime={:.3f}/{:.3f}'.format(k, est_stoi, est_pesq, est_snr,
                                                                                       curr_time, mtime))

            score_stois[noise_type + '_' + snr_value + '_mix'] = acc_stoi_mix / num_clips
            score_stois[noise_type + '_' + snr_value + '_est'] = acc_stoi_est / num_clips
            score_snrs[noise_type + '_' + snr_value + '_mix'] = acc_snr_mix / num_clips
            score_snrs[noise_type + '_' + snr_value + '_est'] = acc_snr_est / num_clips
            score_pesqs[noise_type + '_' + snr_value + '_mix'] = acc_pesq_mix / num_clips
            score_pesqs[noise_type + '_' + snr_value + '_est'] = acc_pesq_est / num_clips
            f_mix.close()
        self.printResult(self.args.model[6:-6], {'STOI': score_stois, 'PESQ': score_pesqs, 'SNR ': score_snrs})

    def printResult(self, epoch, dicts):
        f = open('../logs/result.csv', 'a')
        noises = ['ADTbabble', 'ADTcafeteria']
        snrs = ['snr-5', 'snr-2', 'snr0', 'snr2', 'snr5']
        f.write('Result For Model {}\n'.format(epoch))
        for metric_type, values in dicts.items():
            noise_list = [metric_type]
            snr_list = [metric_type]
            mix_list = ['MIX\t']
            est_list = ['EST\t']
            for n in noises:
                for r in snrs:
                    domain = n + '_' + r
                    snr_list.append('({})'.format(r))
                    mix_list.append('{:.4f}'.format(round(values[domain + '_mix'], 4)))
                    est_list.append('{:.4f}'.format(round(values[domain + '_est'], 4)))
                noise_list.append('({})'.format(n))
                noise_list.append('\t' * (len(snrs) + 1))
            print('\t'.join(noise_list))
            print('\t'.join(snr_list))
            print('\t'.join(mix_list))
            print('\t'.join(est_list))
            f.write('{}\n'.format('\t'.join(noise_list)))
            f.write('{}\n'.format('\t'.join(snr_list)))
            f.write('{}\n'.format('\t'.join(mix_list)))
            f.write('{}\n'.format('\t'.join(est_list)))
        f.write('\n\n')
        f.close()

    def inference(self):
        net = Net(self.conf["models"]["frame_size"], self.conf["models"]["frame_shift"])
        net = torch.nn.DataParallel(net)
        net.to(self.device)
        net.eval()
        checkpoint = Checkpoint()
        checkpoint.load(os.path.join(self.conf["output"]["model_path"], self.args.model))
        net.load_state_dict(checkpoint.state_dict)
        inference_lst = gen_list(self.args.dir, ".wav")
        inference_bar = ProgressBar(0, len(inference_lst))
        inference_bar.start()
        for infer_idx in range(len(inference_lst)):
            filename = inference_lst[infer_idx]
            mixture, fs = sf.read(os.path.join(self.args.dir, filename))
            len_speech = len(mixture)
            mixture = torch.from_numpy(mixture).float().reshape(1, 1, -1)
            alpha = 1 / torch.sqrt(torch.sum(mixture ** 2) / len_speech)
            mixture = mixture * alpha
            nsample = torch.IntTensor([mixture.shape[-1]])
            est, _ = net(mixture, nsample, mixture)
            est = est.cpu().detach().numpy()[0] / alpha
            sf.write(
                file=os.path.join(self.conf["output"]["inference_path"], filename.replace('mix', 'est')),
                data=est,
                samplerate=fs
            )
            inference_bar.update_progress(infer_idx, 'infer')
        print('Done ~')

    def num_params(self, net):
        num = 0
        for param in net.parameters():
            num += int(np.prod(param.size()))
        return num

    def get_snr(self, ref, est):
        return 10.0 * np.log10(np.sum(ref ** 2) / np.sum((est - ref) ** 2))

    def compute_metrics(self, ideal_s, est_s):
        st1 = stoi(ideal_s, est_s, int(self.conf["models"]["srate"]), extended=False) * 100
        st2 = stoi(ideal_s, est_s, int(self.conf["models"]["srate"]), extended=True) * 100
        sn = self.get_snr(ideal_s, est_s)
        # est_d = eval_composite(ideal_s, est_s)
        est_d = {"csig": 0, "cbak": 0, "covl": 0, "pesq": 0}

        pe_nb = pesq(ideal_s, est_s, int(self.conf["models"]["srate"]))
        est_d["pesq_nb"] = pe_nb
        est_d["stoi"] = st1
        est_d["estoi"] = st2
        est_d["snr"] = sn
        return est_d

    def logging(self, info, mdict=None, log_type='loss'):
        assert log_type in ['loss', 'metric', 'info']
        if log_type == 'loss':
            checkpoint = info
            if checkpoint.start_epoch == 1:
                header = "epoch, best_loss, train_loss, eval_loss\n"
            else:
                header = ""
            with open(os.path.join(self.conf["output"]["log_path"], "loss_log.txt"), "a") as f_log:
                f_log.write(
                    header
                    + "{}, {}, {:.4f}, {:.4f}\n".format(
                        checkpoint.start_epoch,
                        checkpoint.best_loss,
                        checkpoint.train_loss,
                        checkpoint.eval_loss,
                    )
                )
        elif log_type == 'metric':
            epoch = info
            if epoch == 1:
                header = "EPOCH, STOI, ESTOI, PESQ_NB, PESQ_WB, SNR, CSIG, CBAK, COVL\n"
            else:
                header = ""
            with open(os.path.join(self.conf["output"]["log_path"], "metric_log.txt"), "a") as f_log:
                f_log.write(
                    header
                    + "{}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}\n".format(
                        epoch,
                        mdict["stoi"],
                        mdict["estoi"],
                        mdict["pesq_nb"],
                        mdict["pesq_wb"],
                        mdict["snr"],
                        mdict["csig"],
                        mdict["cbak"],
                        mdict["covl"],
                    )
                )
        elif log_type == 'info':
            msg = info
            log_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
            print(msg)
            with open(os.path.join(self.conf["output"]["log_path"], "train_log.txt"), "a") as f_log:
                f_log.write(
                    "{} [INFO]: {}\n".format(
                        log_time,
                        msg
                    )
                )
