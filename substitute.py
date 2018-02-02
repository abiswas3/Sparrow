import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import numpy as np
from sklearn.metrics import *

# general purpose shit everyone should have
from utils import *

from jack import Jack
from model import ImageClassifier

class Substitute(object):

    def __init__(self,
                 data,
                 target_model,
                 num_dimensions,
                 num_classes,
                 learning_rate=0.1,
                 batch_size=100):

        self.data = data
        self.target_model = target_model
        self.num_dimensions = num_dimensions
        self.num_classes    = num_classes
        self.sess           = tf.Session()
        self.learning_rate  = 0.1
        self.batch_size     = batch_size
        self.build_model()

    def oracle(self, xvals):

        if self.sess != None:
            y = self.sess.run(self.y_hat,
                                 feed_dict={self.input_data: xvals})

            out = np.zeros((len(xvals), self.num_classes))
            for i, p in enumerate(y):
                out[i, np.argmax(p)] = 1
            return out
        else:
            print("We have not learned a classifier")
            return []


    def oracle_probs(self, xvals):

        if self.sess != None:
            return self.sess.run(self.y_hat,
                                 feed_dict={self.input_data: xvals})

        else:
            print("We have not learned a classifier")
            return []

    def get_gradient(self, xvals, lvals):

        return  self.sess.run(self.gv,
                              feed_dict={self.input_data: xvals,
                                         self.input_labels: lvals})


    def get_gradients_wrt_logits(self, feed_dict):

        grads = tf.stack([tf.gradients(yi, self.input_data)[0]
                          for yi in tf.unstack(self.y_hat, axis=1)],
                         axis=2)

        return self.sess.run(grads, feed_dict=feed_dict)


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

        self.gv = tf.gradients(self.loss, [self.input_data])[0]

    def craft_sample_aux(self,
                         x,
                         predicted_label, eps=0.07):

        # This is the Jacbian
        grad = self.get_gradients_wrt_logits({self.input_data: x})

        new_samples = np.zeros(x.shape)
        for i in range(len(x)):
            g = grad[i][:, predicted_label[i]]
            new_samples[i] = self.data[i] + np.sign(g)*eps

        return new_samples


    def get_next_batch(self, batch_size):

        # make this dumb for now
        inds = np.arange(len(self.labels))

        if self.batch_size < len(self.labels):
            inds = np.random.choice(inds, size=batch_size, replace=True)
            return self.data[inds], self.labels[inds]
        else:
            return self.data, self.labels

    def _train(self):

        sess =  self.sess

        sess.run(tf.global_variables_initializer())

        # Learning
        for step in range(1000):
            batch_xs, batch_ys = self.get_next_batch(self.batch_size)

            # Trainining step
            _ = sess.run([self.optimizer],
                         feed_dict={self.input_data  : batch_xs,
                                    self.input_labels: batch_ys})


    def train(self):

        for rho in range(1):
            print('Rho')
            # label my data
            y_pred      = self.target_model.oracle(self.data)
            self.labels = y_pred

            self._train()
            new_data   = self.craft_sample_aux(self.data, [np.argmax(i) for i in y_pred])
            new_labels =  self.target_model.oracle(new_data)

            # add em in
            self.data = np.vstack((new_data, self.data))
            self.labels = np.vstack((new_labels, self.labels))

if __name__ == '__main__':

    # Using test data cos it's smaller and fits in RAM better

    # Import data as tensors
    mnist = input_data.read_data_sets("./MNIST_data/",
                                      one_hot=True)

    ##########################################################################
    # Train a regular classifier that we wisht o break
    #########################################################################
    target_model = ImageClassifier(mnist.train.images,
                         mnist.train.labels,
                         784,
                         10)

    target_model.train()

    ##########################################################################
    # Pick some data to test this on
    #########################################################################
    test_x =  mnist.test.images[:50]
    test_y =  mnist.test.labels[:50]
    
    y_truth =  [np.argmax(i) for i in test_y]
    # Prediction by the classifier i am trying to break
    y_pred =  [np.argmax(i) for i in target_model.oracle(test_x)]
    
    print('Prediction accuracy: {}\n'.format(accuracy_score(y_truth, y_pred)))

    ##########################################################################
    # From here on I have no access to any training labels
    # all i have access is to the target model
    #########################################################################
    print('Training substitute')
    # Train a substitute classifier
    inds = np.arange(len(mnist.train.images))
    inds_to_select = np.random.choice(inds, size=5000)
    
    im = Substitute(mnist.train.images[inds_to_select],
                    target_model,
                    784,
                    10)

    im.train()

    ##########################################################################
    # Now I use substitute for all my gradients and hope they're good enough
    #########################################################################
    j  = Jack()

    # I need some fake labels to create bad examples: use the labels
    # by substitute model
    fake_labels = np.zeros(test_y.shape)
    y_pred_sub = [np.argmax(i) for i in im.oracle(test_x)]
    for i,p in enumerate(y_pred_sub):
        fake_labels[i, p] = 1

    # these our the bad ones
    pirates = j.turn_em_into_a_pirate(mnist.test.images,
                                      fake_labels,
                                      im,
                                      eps=0.07,
                                      num_test_images = 'all')

    y_bad   = [np.argmax(i) for i in im.oracle(pirates)]
    
    # finally print out the difference in
    print('Pred: {} Jacked: {}\n'.format(accuracy_score(y_truth, y_pred),
                                       accuracy_score(y_truth, y_bad)))

    print('We have cocked it up')
