"""
参数和配置信息
"""

import argparse
from collections import OrderedDict


class Config(object):
    def __init__(self):
        self.name = "Pix2Pix model in Tensorflow"

        self.args = OrderedDict(datadir=("./TFRecords/cityscapes", "path of datasets"),
                                imsize=((256, 512), "image size"),
                                clambda=(100.0, "weight on L1 term for generator gradient"),
                                is_training=(True, 'train or test'),
                                lr_g=(2e-4, "learning rate of G"),
                                lr_d=(1e-4, "learning rate of D"),
                                beta1=(0.5, "momentum term of adam"),
                                direction=("B2A", "A2B, B2A"),
                                EPS=(1e-12, "epsilon"),
                                sample_freq=(300, "frequency of updating sample images"),
                                sample_to_file=(True, "save samples to image file"),
                                logdir=("logs", "path to save logs"),
                                sampledir=("results/cityscapes", "path to save examples"),
                                checkpointdir=("checkpoint", "path to save checkpoints")
                                )

    def __call__(self):
        parser = argparse.ArgumentParser(prog=self.name)
        for key, value in self.args.items():
            var, doc = value
            parser.add_argument("--%s" % key, dest=key, type=type(var), default=var, help=doc)

        return parser.parse_args()


if __name__ == "__main__":
    config = Config()
    config = config()
    print(config.datadir)
    print("..........")