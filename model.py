import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import numpy as np
from sklearn.metrics import *

# general purpose shit everythone should have
from utils import *


class ImageClassifier(object):

    def __init__(self,
                 num_dimensions,
                 num_classes):

        self.num_dimensions = num_dimensions
        self.num_classes    = num_classes
        self.build_model()


    def badify(self,
               x,
               gr,
               eps = 0.1):

        '''
        Goodfellow badify
        '''
        eta = np.array([eps*sgn(i) for i in gr])

        return x+eta

    def uniform_noisify(self,
                        x,
                        eps = 0.5):

        eta = eps*(np.random.uniform(-1,1, 784))

        return x + eta

    def evaluate(self,
                 test_data,
                 test_labels,
                 session,
                 eps=0.07,
                 num_test_images = 'all'):

        num_test_images = len(test_labels) if num_test_images == 'all' else num_test_images

        xvals = test_data[:num_test_images]
        lvals = test_labels[:num_test_images]
        y_true = lvals[:]

        # Get gradient so i can update eta
        grvar = session.run(self.gv,
                            feed_dict={self.input_data: xvals,
                                       self.input_labels: lvals})

        y_probs_noiseless = session.run(self.y_hat, feed_dict={self.input_data: xvals})

        adv = np.zeros(xvals.shape)
        for i, xval in enumerate(xvals):
            adv[i] = self.badify(xval, grvar[i], eps=eps)

        y_probs_adv = session.run(self.y_hat, feed_dict={self.input_data: adv})

        rand = np.zeros(xvals.shape)
        for i, xval in enumerate(xvals):
            rand[i] = self.uniform_noisify(xval, eps=eps)

        y_probs_rand = session.run(self.y_hat, feed_dict={self.input_data: rand})

        y_true_pred = [np.argmax(i) for i in y_true] 
        y_noiseless_pred = [np.argmax(i) for i in y_probs_noiseless]
        y_adv_pred = [np.argmax(i) for i in y_probs_adv]
        y_rand_pred = [np.argmax(i) for i in y_probs_rand]
 
        s = "Noiseless {}{}{} Adversary {}{}{} Uniformly Random {}{}{}\n"
        print(s.format(WARNING,
                       accuracy_score(y_true_pred, y_noiseless_pred),
                       ENDC,
                       BLUE,
                       accuracy_score(y_true_pred, y_adv_pred),
                       ENDC,
                       GREEN,
                       accuracy_score(y_true_pred, y_rand_pred),
                       ENDC))


    def oracle(self):
        pass

    def get_gradient(self, func, variable):

        return tf.gradients(func, [variable])[0]

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
        self.gv = self.get_gradient(self.loss, self.input_data)

    def train(self):

        sess =  tf.Session()
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
        # Evaluate
        for eps in np.linspace(0.001, 0.1, 20):
            print('EPSILON', eps)

            self.evaluate(mnist.test.images,
                          mnist.test.labels,
                          sess,
                          eps=eps,
                          num_test_images = 10)

        sess.close()

if __name__ == '__main__':

    # Import data as tensors
    mnist = input_data.read_data_sets("./MNIST_data/",
                                      one_hot=True)

    im = ImageClassifier(784, 10)
    im.train()
