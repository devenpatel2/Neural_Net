#!/usr/bin/env python
import numpy as np
from neuron import Neuron

class Layer(object):

    """
    Layer class
    """

    def __init__(self, nodes, initial_weights=None, bias=None, neuron_type="linear"):
        self._ntype = neuron_type 
        self._neurons = Neuron(neuron_type)
        assert(len(nodes) == 2), "param nodes is a tuple/list of length 2"
        assert(type(nodes[0]) == int)
        assert(type(nodes[1]) == int)
        self._nodes = nodes
        self._init_weights = initial_weights
        self._bias = bias
        grad_fn = dict()
        grad_fn = {"linear" : self._linear_layer_grad,
                   "logistic": self._logistic_layer_grad
                   }
        self.grad = grad_fn[neuron_type]
        
        if self._init_weights is None:
            self._init_weights = np.random.rand(nodes[0], nodes[1])
        if self._bias is None:
            self._bias = np.ones(self._nodes[1])
        
        assert(len(self._init_weights.T) == self._nodes[1])
        assert(len(self._bias) == self._nodes[1])
        self._state = np.zeros(self._nodes[1]) 
        self._bias_weights = np.ones(self._nodes[1])
        self._weights = self._init_weights

    def __call__(self):
       pass;

    def _linear_layer_grad(self):
        return np.ones(self._state.shape)

    def _logistic_layer_grad(self):

        grad = np.multiply(self._state, 1-self._state)
        return grad

    def forward(self, x):
        layer_out = self._neurons.eval(x, self._weights, self._bias, self._bias_weights)
        #store state of layer
        self._state = layer_out
        return layer_out
    
    def backward(self, x):
        b = self.bias 
        if(len(x.shape) > 1):
            n_inputs = x.shape[1]
            b = np.repeat(
                b.reshape([1, self._nodes[1]]), n_inputs, axis=0)  # makes 'b' a  k x m matrix
        x = x + b.T
        bias = np.zeros(len(self._weights))
        layer_out = self._neurons.eval(x, self._weights.T, bias)
        return layer_out
    
    @property
    def nodes(self):
        return self._nodes

    @property
    def ntype(self):
        return self._ntype

    @property
    def init_weights(self):
        return self._init_weights

    @init_weights.setter
    def init_weights(self, init_weights):
        assert(len(init_weights.T) == self._nodes[1])
        self._init_weights = init_weights

    @property
    def bias(self):
        return self._bias

    @bias.setter
    def bias(self, bias):
        assert(len(bias) == self._nodes[1])
        self._bias = bias

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, weights):
        assert(self._weights.shape == weights.shape)
        self._weights = weights
    
    @property
    def bias_weights(self):
        return bias_weights

    @bias_weights.setter
    def bias_weights(self, weights):
        assert(len(weights) == len(self._bias))
        self._bias_weights = weights

    @property
    def state(self):
        return self._state
   
  
if __name__ == "__main__":


    l = Layer([3, 4])
    # single input
    x = np.asarray([1, 2, 3])
    print(l.forward(x))

    # batch input
    x = np.repeat(x.reshape([3, 1]), 2, axis=1)
    print(l.forward(x))
