#!/usr/bin/env python

import numpy as np


class Neuron(object):

    def __init__(self, ntype="linear"):
        
        neuron = dict()
        neuron = {
            "linear": [self._linear_eval, self._linear_grad],
                "logistic": [self._logistic_eval, self._logistic_grad]
        }
        self.eval = neuron[ntype][0]
        self.grad = neuron[ntype][1]

    def _linear_eval(self, inputs, weights, bias, bias_weights):
        ''' compute the output of linear neuron(s)
        x - input array/matrix of size n x k where n is the  to the layer
            and k is the number of input vectors
        w - weight matrix of size n x m where n is the number of inputs to the layer
            and m is the number of neurons in the layer so that w(i,j) is the weight from i_th node
            from previous layer to j_th node of current layer.
        b - bias vector of size m
        '''
        assert(len(weights.T) == len(bias))
        assert(len(inputs) == len(weights))
        assert(len(bias) == len(bias_weights))
        if(len(bias.shape) > 1):
            assert(bias.shape[0] == 1 or bias.shape[1] == 1)
        else:
            assert(len(bias.shape) == 1)

        layer_size = weights.shape[1]
        bias = np.multiply(bias, bias_weights)
        if(len(inputs.shape) > 1):
            n_inputs = inputs.shape[1]
            bias = np.repeat(
                bias.reshape([layer_size, 1]), n_inputs, axis=1)  # makes 'b' a  k x m matrix

        response = bias + np.dot(weights.T, inputs)

        return response

    def _linear_grad(self, inputs, weights= None, bias= None, bias_weights= None):

        return np.ones(inputs.shape)

    def _logistic_eval(self, inputs, weights, bias, bias_weights):
        
        linear_sum  = self._linear_eval(inputs, weights, bias, bias_weights)
        exp_1 = 1 + np.exp(-linear_sum)
        response = 1/exp_1
        return response
    
    #TODO
    def _logistic_grad(self, y):
        pass
