#!/usr/bin/env python

import numpy as np
from cost_function import Cost


class Network(object):

    def __init__(self, layers):
        self._layers = layers
        self._check_layers()
        # residual squared error cost
        self.cost = Cost("rse")

    @property
    def layers(self):
        return self._layers

    def pairwise(self, list_to_pair):
        paired = zip(list_to_pair[:-1], list_to_pair[1:])
        return paired

    def _check_layers(self):
        pairwise_layers = self.pairwise(self._layers)
        for l_previous, l_current in pairwise_layers:
            assert(l_previous.nodes[1] == l_current.nodes[
                   0]), "Nodes do not match in network"

    def feed_forward(self, inputs):
        assert(self._layers[0].nodes[0] == len(inputs)),\
            "Length of input does not match with number of nodes in first layer"
        x = inputs.copy()
        for l in self._layers:
            x = l.forward(x)
        return x

    def backpropogation(self, inputs, outputs,  targets, learning_rate=0.0003):
        # reverse layers
        layers_rev = self._layers[::-1]
        assert(layers_rev[0].nodes[1] == len(targets)),\
                "Number of nodes in last layer do not match with target size"

        pairwise_layers_rev = self.pairwise(layers_rev)
        
        #y = self.feed_forward(inputs)
        # err_grad wrt output
        de_dy = self.cost.grad(targets, outputs, ax=0, K =-0.5)
        for index, l_current  in enumerate(layers_rev):

            # output_grad wrt input
            dy_dz = l_current.grad()

            # err_grad wrt input
            de_dz = np.multiply(de_dy, dy_dz)

            # input_grad wrt weights
            if(index < len(self._layers)-1):
                dz_dw = layers_rev[index+1].state
            else:
                dz_dw = inputs
            # err_grad wrt weights
            if(len(inputs.shape) == 1 and len(inputs.shape) == 1):
                de_dw = np.outer(dz_dw, de_dz)
                de_db = np.dot(l_current.bias, de_dz)
            else:
                de_dw = np.dot(dz_dw, de_dz.T)
                de_db = np.dot(l_current.bias, np.sum(de_dz,1))
        
            #print(" wieghts", l_current.weights)
            
            # update weights
            l_current.weights = l_current.weights - learning_rate * de_dw

            #bias_weights update
            l_current.bias_weights = l_current.bias_weights - learning_rate * de_db
            # err_grad wrt output of previous layer

            de_dy = np.dot(l_current.weights, de_dz)

    def train(self, inputs, targets, epochs = 10):
        for i in range(epochs):
            outputs = self.feed_forward(inputs)
            err = self.cost.eval(targets, outputs, K= 0.5)
            print("Cost error %s"%err)
            self.backpropogation(inputs, outputs, targets)

    def feed_backward(self, x):
        assert(self._layers[-1].nodes[1] == len(x))
        for l in reversed(self._layers):
            x = l.backward(x)

        return x

if __name__ == "__main__":
    from layer import Layer

    l1 = Layer([3, 4], neuron_type="linear")
    l2 = Layer([5, 5], neuron_type="linear")
    l3 = Layer([4, 2], neuron_type="linear")

    net = Network([l1, l3])
    #x = np.asarray([1, 2, 3])
    #t = np.asarray([3,5])
    x = np.asarray([[3, 1, 2],[ 3, 2 ,5] , [6,3,4], [1,5,6], [9,3,6 ] ,[3,6,1]] ).T
    t = np.asarray([[4,3],[5,7],[9,7],[6,11],[12,9],[9,7]] ).T
 
    #print(t.shape)
    net.train(x,t, epochs =20 )
    #print(net.feed_forward(x))
    print(net.feed_forward(np.asarray([1,5,7])))
    
    # batch input
    #x = np.repeat(x.reshape([3, 1]), 3, axis=1)
    #t = np.repeat(t.reshape([2, 1]), 3, axis=1)
    #net.train(x,t )
    #print(net.feed_forward(x))
