import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import numpy as np
from sklearn.metrics import *

# general purpose shit everyone should have
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
            return np.array([np.argmax(i) for i in y_probs_noiseless])
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

    # Train a classifier
    im = ImageClassifier(784, 10)
    im.train()

    # Now i have access to the learned model (this is the stronger assumption)
    # in the next one we'll get a substitute model trained and we will use that instead

    # Jack needs some labels to generate gradients
    # So we use our model to label the data and use those as the true labels
    # (so not using the real labels)
    y_truth = [np.argmax(i) for i in mnist.test.labels]
    y_pred = im.oracle(mnist.test.images)
    
    j  = Jack()    
    fake_labels = np.zeros((mnist.test.labels.shape))    
    for i,p in enumerate(y_pred):
        fake_labels[i, p] = 1

    # these our the bad ones
    pirates = j.turn_em_into_a_pirate(mnist.test.images,
                                      fake_labels,
                                      im,
                                      eps=0.07,
                                      num_test_images = 'all')

    y_bad   = im.oracle(pirates)

    # finally print out the difference in 
    print('Pred: {} Jacked: {}\n'.format(accuracy_score(y_truth, y_pred),
                                       accuracy_score(y_truth, y_bad)))
                                       
    print('We have cocked it up')
