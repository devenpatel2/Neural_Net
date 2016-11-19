#!/usr/bin/env python

import numpy as np


class Cost(object):

    def __init__(self, cost="mse"):

        cost_functions = dict()
        cost_functions = {
                "mse": [self._mean_squared_eval, self._mean_squared_grad],
                #"mean_squared": [self._mean_squared_eval, self._mean_squared_grad],
                "rse": [self._residual_squared_eval, self._residual_squared_grad]
                #"residual_squared": [self._residual_squared_eval, self._residual_squared_grad]
            }
        self.eval = cost_functions[cost][0]
        self.grad = cost_functions[cost][1]

    def _mean_squared_eval(self, x, y, ax=None, K=1):
        '''
        Compute Mean Square Error

            with ax=0 the average is performed along the row, for each column, returning an array
            with ax=1 the average is performed along the column, for each row, returning an array
            with ax=None the average is performed element-wise along the array, returning a single value
        '''
        assert(x.shape == y.shape),"cannot sub arrays of shape %s and %s"%(x.shape,y.shape)
        mse = ((x - y)**2).mean(axis=ax)
        mse = mse * K
        return mse

    def _mean_squared_grad(self, x, y, ax=None, K=1):

        assert(x.shape == y.shape),"cannot sub arrays of shape %s and %s"%(x.shape,y.shape)
        grad = 2 * (x - y).mean(axis=ax)
        grad = grad * K
        return grad

    def _residual_squared_eval(self, x, y, ax=None, K=1):
        '''
        Compute Mean Square Error

            with ax=0 the average is performed along the row, for each column, returning an array
            with ax=1 the average is performed along the column, for each row, returning an array
            with ax=None the average is performed element-wise along the array, returning a single value
        '''

        assert(x.shape == y.shape),"cannot sub arrays of shape %s and %s"%(x.shape,y.shape)
        rse = ((x - y)**2).sum(axis=ax)
        rse = rse * K
        return rse

    def _residual_squared_grad(self, x, y, ax=None, K=1):

        assert(x.shape == y.shape),"cannot sub arrays of shape %s and %s"%(x.shape,y.shape)
        grad = 2 * (x - y).sum(axis=ax)
        grad = grad * K
        return grad

if __name__ == "__main__":
    a = np.ones([2,4])
    b = np.zeros([2,4])

    c = Cost()
    print(c.eval(a,a, ax =1))
    print(c.eval(a,b, ax = 0))
