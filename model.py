import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import numpy as np
from sklearn.metrics import *

# general purpose shit everyone should have
from utils import *

from jack import Jack

from get_cifar_data import make_cifar_data

class ImageClassifier(object):

    def __init__(self,
                 data,
                 labels,
                 num_dimensions,
                 num_classes,
                 learning_rate=0.1,
                 batch_size=100):

        self.data = data
        self.labels = labels
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
        print(self.gv)
        
    def get_next_batch(self, batch_size):

        # make this dumb for now
        inds = np.arange(len(self.labels))

        if self.batch_size < len(self.labels):
            inds = np.random.choice(inds, size=batch_size, replace=True)
            return self.data[inds], self.labels[inds]
        else:
            return self.data, self.labels
        
    def train(self, epochs=1000):

        sess =  self.sess

        sess.run(tf.global_variables_initializer())

        # Learning
        for step in range(epochs):
            batch_xs, batch_ys = self.get_next_batch(self.batch_size)

            # Trainining step
            _, loss = sess.run([self.optimizer, self.loss],
                         feed_dict={self.input_data  : batch_xs,
                                    self.input_labels: batch_ys})

            if step % 500 == 0:
                y_truth =  [np.argmax(i) for i in self.labels]
                y_pred =  [np.argmax(i) for i in self.oracle(self.data)]    
                print('{} training_accuracy: {}\n'.format(step, accuracy_score(y_truth, y_pred)))

        print('Training finished')

if __name__ == '__main__':

    # Import data as tensors
    mnist = input_data.read_data_sets("./MNIST_data/",
                                      one_hot=True)

    # data, description, wide_labels = make_cifar_data()
    
    # Train a classifier
    data = mnist.train.images
    labels = mnist.train.labels
    im = ImageClassifier( data,
                          labels,
                          len(data[0]),
                          10)
    
    im.train(epochs=5000)

    # Now i have access to the learned model (this is the stronger assumption)
    # in the next one we'll get a substitute model trained and we will use that instead

    # Jack needs some labels to generate gradients
    # So we use our model to label the data and use those as the true labels
    # (so not using the real labels)
    y_truth =  [np.argmax(i) for i in mnist.test.labels]
    y_pred =  [np.argmax(i) for i in im.oracle(mnist.test.images[:])]
    print('Pred: {}\n'.format(accuracy_score(y_truth, y_pred)))
    
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

    y_bad   = [np.argmax(i) for i in im.oracle(pirates)]

    # finally print out the difference in 
    print('Pred: {} Jacked: {}\n'.format(accuracy_score(y_truth, y_pred),
                                       accuracy_score(y_truth, y_bad)))
                                       
    print('We have cocked it up')
