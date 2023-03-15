import os
import sys
import gflags
import torch
import yaml
from models import Model
import torch.multiprocessing as mp


class ArgParser(object):
    def __init__(self, argv):
        # region: load sys.argv
        FLAGS = gflags.FLAGS
        gflags.DEFINE_string("mode", "train", "run mode")
        gflags.DEFINE_string("model", "", "resume_model")
        gflags.DEFINE_string("conf", "config.yaml", "config_path")
        gflags.DEFINE_string("dir", "", "inference input dir")
        FLAGS(argv)
        assert FLAGS.mode in ["train", "test", "inference"]
        # endregion

        # region: load configs
        file = open(FLAGS.conf, 'r')
        conf = yaml.load(file, yaml.FullLoader)
        file.close()
        # endregion

        # region: initialize paths
        for path in conf["output"].values():
            if not os.path.isdir(path):
                os.makedirs(path)
        # endregion

        # region: initialize gpus
        if FLAGS.mode == 'train':
            os.environ["CUDA_VISIBLE_DEVICES"] = conf["train_cuda"]
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = conf["test_cuda"]
        # endregion

        self.mode = FLAGS.mode
        self.model = FLAGS.model
        self.dir = FLAGS.dir
        self.conf = conf


def main(rank, args):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    args.rank = rank
    torch.cuda.set_device(args.rank)
    model = Model(args)
    model.train(args)


if __name__ == '__main__':
    args = ArgParser(sys.argv)
    if args.mode == "train":
        mp.set_start_method("spawn")
        mp.spawn(main, (args,), nprocs=args.conf['num_gpu'])
    elif args.mode == "test":
        model = Model(args)
        model.test()
    elif args.mode == "inference":
        model = Model(args)
        model.inference()
