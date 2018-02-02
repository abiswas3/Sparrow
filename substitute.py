import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import numpy as np
from sklearn.metrics import *

# general purpose shit everyone should have
from utils import *

from jack import Jack

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
        self.sess           = None
        self.learning_rate  = 0.1
        self.batch_size     = batch_size
        self.build_model()

    def oracle(self, xvals):

        if self.sess != None:
            y_probs_noiseless = self.sess.run(self.y_hat,
                                              feed_dict={self.input_data: xvals})
            
            return np.array([np.argmax(i) for i in y_probs_noiseless])
        else:
            print("We have not learned a classifier")
            return []

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


    def craft_sample_aux(self,
                         x,
                         predicted_label, eps=0.07):

        print("Getting Jacobian")
        grad = self.get_gradients_wrt_logits({self.input_data: x})
        
        new_samples = np.zeros(len(x))
        for i in range(len(self.data)):
            g = grad.shape
            print(g.shape)
            break
            # new_samples[i,:] = self.data[i] + eps*[sgn(k) for k in ]
            
        return new_samples


    def get_next_batch(self, batch_size):

        # make this dumb for now
        inds = np.arange(len(self.labels))

        if self.batch_size < len(self.labels):
            inds = np.random.choice(inds, size=batch_size, replace=True)
            return self.data[inds], self.labels[inds]
        else:
            return self.data, self.labels
        
    def train(self):

        sess =  tf.Session()
        self.sess = sess

        sess.run(tf.global_variables_initializer())

        # Learning
        for step in range(1000):
            # make batches of data and labels
            # batch_xs, batch_ys = mnist.train.next_batch(100)
            batch_xs, batch_ys = self.get_next_batch(self.batch_size)
            
            # Trainining step
            _ = sess.run([self.optimizer],
                         feed_dict={self.input_data  : batch_xs,
                                    self.input_labels: batch_ys})

        print('Training finished')

        test_x = mnist.test.images[:]
        test_y = mnist.test.labels[:]
        
        y_truth = [np.argmax(i) for i in test_y]
        y_pred  = self.oracle(test_x)
        print('Pred: {}\n'.format(accuracy_score(y_truth, y_pred)))
        return self.craft_sample_aux(batch_xs[:10], y_pred)


if __name__ == '__main__':

    # Import data as tensors
    mnist = input_data.read_data_sets("./MNIST_data/",
                                      one_hot=True)

    # Train a classifier
    im = ImageClassifier(mnist.train.images,
                         mnist.train.labels,
                         784,
                         10)
    x = im.train()
    
