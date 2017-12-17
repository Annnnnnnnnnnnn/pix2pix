"""
构建数据
"""

import tensorflow as tf
import glob
import random
import os


def data_reader(input_dir: str, shuffle: bool=True):
    """
    Read images from input_dir then shuffle them
    :param input_dir: path of input dir
    :param shuffle: list of string
    :return:
    """
    image_list = glob.glob(os.path.join(input_dir, "*.jpg"))

    if shuffle:
        # Shuffle the ordering of all image files in order to guarantee random ordering of images with respect to label in the
        # saved TFRecord files. Make the randomization repeatable
        random.seed(12345)
        random.shuffle(image_list)
        pass

    return image_list


def _int64_feature(value: int):
    """
    Wrapper for inserting int 64 features into Example proto.
    :param value:
    :return:
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value: bytes):
    """
    Wrapper for inserting bytes feature into Example proto
    :param value:
    :return:
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(file_path: str, image_buffer: bytes):
    """
    Build on Example proto for an example
    :param file_path: path to an image file
    :param image_buffer: JPG encoding of RGB image
    :return:
    """

    example = tf.train.Example(features=tf.train.Features(feature={
        "image/file_name": _bytes_feature(value=tf.compat.as_bytes(os.path.basename(file_path))),
        "image/encoded_image": _bytes_feature(value=image_buffer)
    }))

    return example


def data_writer(input_dir: str, output_file: str):
    """
    Write data to TFRecords
    :param input_dir: path of input dir
    :param output_file: path of output dir
    :return:
    """
    image_list = data_reader(input_dir)
    output_dir = os.path.dirname(output_file)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # dump to TFRecords file
    writer = tf.python_io.TFRecordWriter(output_file)

    for i, image_path in enumerate(image_list):

        with tf.gfile.FastGFile(image_path, "rb") as f:
            image_data = f.read()

        example = _convert_to_example(image_path, image_data)
        writer.write(example.SerializeToString())

        if i % 500 == 0:
            print("Processed {}/{}.".format(i, len(image_list)))

    writer.close()
    print("Done.")


if __name__ == "__main__":
    print("Convert X data to TFRecords...")
    data_writer("../CycleGAN/datasets/cityscapes/test", "./TFRecords/cityscapes/test.tfrecords")