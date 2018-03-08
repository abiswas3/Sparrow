import numpy as np
from utils import *

class Jack(object):

    def __init__(self):

        pass

    def badify(self,
               x,
               gr,
               eps = 0.1):

        '''
        Goodfellow badify
        '''
        eta = np.sign(gr)*eps
        
        return x+eta

    def uniform_noisify(self,
                        x,
                        eps = 0.5):

        eta = eps*(np.random.uniform(-1,1, 784))

        return x + eta

    def turn_em_into_a_pirate(self,
                              test_data,
                              test_labels,
                              model,
                              eps=0.07,
                              num_test_images = 'all'):

        num_test_images = len(test_labels) if num_test_images == 'all' else num_test_images

        xvals = test_data[:num_test_images]
        lvals = test_labels[:num_test_images]

        # Get gradient so i can update eta
        grvar = model.get_gradient(xvals, lvals)

        adv = np.zeros(xvals.shape)
        for i, xval in enumerate(xvals):
            adv[i] = self.badify(xval, grvar[i], eps=eps)

        return adv
