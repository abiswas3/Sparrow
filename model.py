import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import numpy as np
from sklearn.metrics import *

# general purpose shit everythone should have
from utils import *

from jack import Jack

class ImageClassifier(object):

    def __init__(self,
                 num_dimensions,
                 num_classes):

        self.num_dimensions = num_dimensions
        self.num_classes    = num_classes
        self.sess           = None
        self.build_model()



    def oracle(self, xvals):

        if self.sess != None:
            y_probs_noiseless = self.sess.run(self.y_hat, feed_dict={self.input_data: xvals})
            return [np.argmax(i) for i in y_probs_noiseless]
        else:
            print("We have not learned a classifier")
            return []


    def get_gradient(self, xvals, lvals):

        return  self.sess.run(self.gv,
                            feed_dict={self.input_data: xvals,
                                       self.input_labels: lvals})
    
    def build_model(self):

        num_dimensions = self.num_dimensions
        num_classes    = self.num_classes

        self.input_data   = tf.placeholder(tf.float32, [None, num_dimensions])
        self.input_labels = tf.placeholder(tf.float32, [None, num_classes])

        self.weights      = tf.Variable(tf.zeros([num_dimensions, num_classes]),name="WHISKEY") # weights
        self.bias         = tf.Variable(tf.zeros([num_classes]),name="BRAVO") # bias

        W = self.weights
        b = self.bias

        # linear portion of loss function
        self.y_hat = tf.matmul(self.input_data, W) + b

        # loss is monotonic, this is the only bit that really matters
        ce = tf.nn.softmax_cross_entropy_with_logits(labels= self.input_labels,
                                                     logits= self.y_hat)
        self.loss = tf.reduce_mean(ce)

        opter = tf.train.AdamOptimizer()
        self.optimizer = opter.minimize(self.loss)

        # function to get gradient value
        self.gv = tf.gradients(self.loss, [self.input_data])[0]

    def train(self):

        sess =  tf.Session()
        self.sess = sess

        sess.run(tf.global_variables_initializer())

        # Learning
        for step in range(1000):
            # make batches of data and labels
            batch_xs, batch_ys = mnist.train.next_batch(100)

            # Trainining step
            _ = sess.run([self.optimizer],
                         feed_dict={self.input_data  : batch_xs,
                                    self.input_labels: batch_ys})


        print('Training finished')

if __name__ == '__main__':

    # Import data as tensors
    mnist = input_data.read_data_sets("./MNIST_data/",
                                      one_hot=True)
    im = ImageClassifier(784, 10)
    im.train()


    # grads  = im.get_gradient(mnist.test.images,
    #                          mnist.test.labels)

    j = Jack()
    pirates = j.turn_em_into_a_pirate(mnist.test.images,
                                      mnist.test.labels,
                                      im,
                                      eps=0.07,
                                      num_test_images = 'all')

    y_bad   = im.oracle(pirates)
    y_pred  = im.oracle(mnist.test.images)
    y_truth = [np.argmax(i) for i in mnist.test.labels]

    print('Pred: {} Jacked: {}'.format(accuracy_score(y_truth, y_pred),
                                       accuracy_score(y_truth, y_bad)))
                                       
