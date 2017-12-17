"""
pix2pix模型
"""

from GANs.pix2pix.modules import generator, discriminator
from GANs.pix2pix.layers import abs_criterion, mae_criterion
from GANs.pix2pix.utils import *
from GANs.pix2pix.Reader import Reader
from GANs.pix2pix.config import Config
import tensorflow as tf
import numpy as np
from scipy.misc import imsave
import os
import time


class Pix2Pix(object):
    def __init__(self, args):
        """
        创建整个模型
        :param args:
        """
        self.sess = tf.Session()
        self.args = args
        self._create_placeholders()
        self._create_model()
        self._create_losses()
        self._collect_vars()
        self._create_opts()
        self._create_summaries()
        self.saver = tf.train.Saver()
        self.writer = tf.summary.FileWriter(self.args.logdir, graph=self.sess.graph, max_queue=1)

    def _create_placeholders(self):
        """
        创建占位符
        :return:
        """
        self.is_training = tf.placeholder(tf.bool, shape=[], name="is_training")

        self.image1_op = Reader(os.path.join(self.args.datadir, "train.tfrecords"), direction=self.args.direction, name="image1_op").feed()
        self.image2_op = Reader(os.path.join(self.args.datadir, "test.tfrecords"), direction=self.args.direction, name="image2_op").feed()

        self.realA, self.realB = tf.cond(self.is_training, lambda: self.image1_op, lambda: self.image2_op)
        pass

    def _create_model(self):
        """
        创建生成器、判别器
        :return:
        """
        self.fakeB = generator(self.realA, reuse=False, name="G")

        self.D_real = discriminator(self.realA, self.realB, reuse=False, name="D")
        self.D_fake = discriminator(self.realA, self.fakeB, reuse=True, name="D")
        pass

    def _create_losses(self):
        """
        创建损失函数
        :return:
        """
        with tf.variable_scope("Loss_G"):
            # predict_fake => 1
            # abs(targets - outputs) => 0
            self.lossG = tf.add(tf.reduce_mean(-tf.log(self.D_fake + self.args.EPS), name="soft_ones"),
                                self.args.clambda * abs_criterion(self.realB, self.fakeB, name="abs_Loss"),
                                name="Total_Loss")

        with tf.variable_scope("Loss_D"):
            with tf.variable_scope("Loss_D_real"):
                # minimizing -tf.log will try to get inputs to 1
                # predict_real => 1
                # predict_fake => 0
                # mae_criterion(self.D_real, tf.ones_like(self.D_real, name="soft_ones"))
                # tf.reduce_mean(-tf.log(self.D_real), name="Loss_D_real")
                self.lossD_real = tf.reduce_mean(-tf.log(self.D_real + self.args.EPS), name="soft_ones")

            with tf.variable_scope("Loss_D_fake"):
                # minimizing -tf.log will try to get inputs to 1
                # predict_real => 1
                # predict_fake => 0
                # mae_criterion(self.D_fake, tf.zeros_like(self.D_fake, name="soft_zeros"))
                # tf.reduce_mean(-tf.log(1 - self.D_fake), name="Loss_D_fake")
                self.lossD_fake = tf.reduce_mean(-tf.log(1 - self.D_fake + self.args.EPS), name="soft_zeros")

            self.lossD = self.lossD_real + self.lossD_fake
        pass

    def _collect_vars(self):
        """
        搜集可训练变量
        :return:
        """
        self.d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="D")
        self.g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="G")

        self.d_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope="D")
        self.g_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope="G")

        # print(self.d_vars)
        # print(self.g_vars)
        # print(self.d_update_ops)
        # print(self.g_update_ops)
        # exit()
        pass

    def _create_opts(self):
        """
        创建优化目标
        :return:
        """

        @namespace("Adam")
        def make_optimizer(loss, lr, var_list):
            """
            创建优化器
            :param loss: 目标loss
            :param lr: 学习率
            :param var_list: 待更新的变量
            :return:
            """
            global_step = tf.Variable(0, trainable=False)
            starter_learning_rate = lr
            end_learning_rate = 0.0
            start_decay_step = 200000
            decay_steps = 100000

            learning_rate = tf.where(tf.greater_equal(global_step, start_decay_step),
                                     tf.train.polynomial_decay(starter_learning_rate,
                                                               global_step - start_decay_step,
                                                               decay_steps,
                                                               end_learning_rate=end_learning_rate,
                                                               power=1.0),
                                     y=starter_learning_rate)

            learning_step = tf.train.AdamOptimizer(learning_rate, beta1=self.args.beta1).minimize(loss,
                                                                                                  global_step=global_step,
                                                                                                  var_list=var_list)

            return learning_rate, learning_step

        with tf.control_dependencies(self.d_update_ops):
            self.d_lr, self.d_optim = make_optimizer(self.lossD, self.args.lr_d, var_list=self.d_vars, name="Adam_D")

        with tf.control_dependencies(self.g_update_ops):
            self.g_lr, self.g_optim = make_optimizer(self.lossG, self.args.lr_g, var_list=self.g_vars, name="Adam_G")
        pass

    def _create_summaries(self):
        """
        创建TensorBoard可记录量
        :return:
        """
        with tf.name_scope("G_summaries"):
            self.lossG_sum = tf.summary.scalar("lossG", self.lossG)

            self.G_lr_sum = tf.summary.scalar("G_lr", self.g_lr)

            self.G_sum = tf.summary.merge([self.lossG_sum, self.G_lr_sum])

        with tf.name_scope("D_summaries"):
            self.lossD_sum = tf.summary.scalar("lossD", self.lossD)
            self.lossD_real_sum = tf.summary.scalar("lossD_real", self.lossD_real)
            self.lossD_fake_sum = tf.summary.scalar("lossD_fake", self.lossD_fake)

            self.D_lr_sum = tf.summary.scalar("D_lr", self.d_lr)

            self.D_sum = tf.summary.merge([self.lossD_sum, self.lossD_real_sum, self.lossD_fake_sum, self.D_lr_sum])

        with tf.name_scope("image_summaries"):
            self.real_A = tf.placeholder(tf.float32, shape=[1, self.args.imsize[0], self.args.imsize[1] // 2, 3], name="real_A_ph")
            self.fake_B = tf.placeholder(tf.float32, shape=[1, self.args.imsize[0], self.args.imsize[1] // 2, 3], name="fake_B_ph")
            self.real_B = tf.placeholder(tf.float32, shape=[1, self.args.imsize[0], self.args.imsize[1] // 2, 3], name="real_B_ph")

            self.real_A_sum = tf.summary.image("1_real_A", self.real_A)
            self.fake_B_sum = tf.summary.image("2_fake_B", self.fake_B)
            self.real_B_sum = tf.summary.image("3_real_B", self.real_B)

            self.img_op = tf.summary.merge([self.real_A_sum, self.fake_B_sum, self.real_B_sum])

        pass

    def train(self):
        """
        训练
        :return:
        """
        self.sess.run(tf.global_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)
        start_time = time.time()
        counter = 0

        if self.load(self.args.checkpointdir):
            print(" [*] Load SUCCESS")

        else:
            print(" [!] load failed...")

        try:
            while not coord.should_stop():
                # Update D network
                _, lossD, summary_str = self.sess.run([self.d_optim, self.lossD, self.D_sum], feed_dict={self.is_training: True})
                self.writer.add_summary(summary_str, counter)

                # Update G network
                _, lossG, summary_str = self.sess.run([self.g_optim, self.lossG, self.G_sum], feed_dict={self.is_training: True})
                self.writer.add_summary(summary_str, counter)

                if counter % 5 == 0:
                    print("step: [%2d] time: %4.4f, lossD: %.8f, lossG: %.8f" % (counter,
                                                                                 time.time() - start_time,
                                                                                 lossD,
                                                                                 lossG))
                counter += 1
                start_time = time.time()

                if counter % self.args.sample_freq == 1:
                    self.sample_model(self.args.sampledir, counter)

                if counter % 1000 == 2:
                    self.save(self.args.checkpointdir, counter)

        except KeyboardInterrupt:
            print("Interrupted!")
            coord.request_stop()

        except Exception as e:
            coord.request_stop(e)

        finally:
            coord.request_stop()
            coord.join(threads)
            pass

    def save(self, path, counter):
        """
        保存checkpoint
        :param path: 路径
        :param counter: 计数
        :return:
        """
        checkpoint_name = "pix2pix.model"
        self.saver.save(self.sess, os.path.join(path, checkpoint_name), global_step=counter)
        pass

    def load(self, path):
        """
        加载checkpoint
        :param path: 路径
        :return:
        """
        print(" [*] Reading checkpoint...")
        ckpt = tf.train.get_checkpoint_state(path)
        if ckpt and ckpt.model_checkpoint_path:
            checkpoint_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(path, checkpoint_name))
            return True

        else:
            return False
        pass

    def sample_model(self, path, counter):
        """
        保存样例
        :param path: 路径
        :param counter: 总计数
        :return:
        """
        real, fake, target = self.sess.run([self.realA, self.fakeB, self.realB], feed_dict={self.is_training: False})

        img = visual_grid(np.concatenate([real, fake, target], axis=0), shape=[1, 3])

        if self.args.sample_to_file:
            imsave(os.path.join(path, '%04d.png' % counter), img, 'png')

        s_img = self.sess.run(self.img_op, feed_dict={self.real_A: real,
                                                      self.fake_B: fake,
                                                      self.real_B: target})
        self.writer.add_summary(s_img, counter)
        pass

    def test(self):
        """
        测试
        :param direction: A2B、B2A
        :return:
        """
        if self.load(self.args.checkpiontdir):
            print(" [*] Load SUCCESS")

        else:
            print(" [!] load failed...")

        save_path = self.args.sampledir
        for i in range(100):
            ret = self.sess.run(self.fakeB, feed_dict={self.is_training: False})
            file_name = "test_sample_" + str(i)
            imsave(os.path.join(save_path, file_name), ret, "png")


if __name__ == "__main__":
    config = Config()
    config = config()

    if not os.path.exists(config.checkpointdir):
        os.makedirs(config.checkpointdir)

    if not os.path.exists(config.sampledir):
        os.makedirs(config.sampledir)

    pix2pix = Pix2Pix(args=config)
    if config.is_training:
        pix2pix.train()

    else:
        pix2pix.test()
