"""
数据解析
"""
import tensorflow as tf


class Reader(object):
    def __init__(self,
                 tfrecord_file: str,
                 image_size: tuple=(256, 512),
                 min_queue_examples: int=1000,
                 batch_size: int=1,
                 num_threads: int=8,
                 direction: str="A2B",
                 name: str=""):
        """

        :param tfrecord_file: tfrecords file path
        :param image_size:
        :param min_queue_examples: minimum number of samples to retain in the queue that provides of batches of examples
        :param batch_size: number of images per batch
        :param num_threads: number of preprocess threads
        :param direction: "AtoB, BtoA"
        :param name:
        """
        self.tfrecord_file = tfrecord_file
        self.image_size = image_size
        self.min_queue_examples = min_queue_examples
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.direction = direction
        self.name = name
        self.reader = tf.TFRecordReader()

    def feed(self):
        """
        :return: images: 4D tensor [batch_size, image_width, image_height, image_depth]
        """
        with tf.name_scope(self.name):
            file_name_queue = tf.train.string_input_producer([self.tfrecord_file])

            _, serialized_example = self.reader.read(file_name_queue)
            features = tf.parse_single_example(serialized_example, features={
                "image/file_name": tf.FixedLenFeature([], tf.string),
                "image/encoded_image": tf.FixedLenFeature([], tf.string)
            })

            image_buffer = features["image/encoded_image"]
            image = tf.image.decode_jpeg(image_buffer)
            imageA, imageB = self._preprocess(image)

            if self.direction == "B2A":
                imageA, imageB = imageB, imageA

            imagesA, imagesB = tf.train.shuffle_batch([imageA, imageB],
                                                      batch_size=self.batch_size,
                                                      capacity=self.min_queue_examples + 3 * self.batch_size,
                                                      min_after_dequeue=self.min_queue_examples,
                                                      num_threads=self.num_threads)

            return imagesA, imagesB

    def _preprocess(self, image):
        image = tf.image.resize_images(image, size=self.image_size)
        image = self._convert2float(image)
        image.set_shape([self.image_size[0], self.image_size[1], 3])
        # imageA, imageB = image[0:self.image_size // 2, 0:self.image_size // 2, :], image[self.image_size // 2:, self.image_size // 2:, :]
        imageA = tf.slice(image, begin=[0, 0, 0], size=[self.image_size[0], self.image_size[1] // 2, 3])
        imageB = tf.slice(image, begin=[0, self.image_size[1] // 2, 0], size=[self.image_size[0], self.image_size[1] // 2, 3])
        return imageA, imageB
        pass

    def _convert2float(self, image):
        """
        Transfrom from int image ([0,255]) to float tensor ([-1.,1.])
        :param image:
        :return:
        """
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        return (image / 127.5) - 1.0
        pass


def test_reader():
    TRAIN_FILE_1 = "./TFRecords/edges2shoes/test.tfrecords"

    reader1 = Reader(TRAIN_FILE_1, batch_size=2)

    image1_op = reader1.feed()

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            step = 0
            while not coord.should_stop():
                batch_image1, batch_image2 = sess.run(image1_op)
                print(batch_image1.shape, batch_image2.shape)
                print("--------" * 10)
                step += 1

        except KeyboardInterrupt:
            print("Interrupted")
            coord.request_stop()

        except Exception as e:
            coord.request_stop(e)

        finally:
            # When done, ask the threads to stop.
            coord.request_stop()
            coord.join(threads)


if __name__ == "__main__":
    test_reader()