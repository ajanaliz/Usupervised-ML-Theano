import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from util import relu, error_rate, getKaggleMNIST, init_weights

# new additions used to compare purity measure using GMM
import os
import sys
sys.path.append(os.path.abspath('..'))
from Class1.kmeans_mnist import purity
from Class1.gmm import gmm

class Layer(object):
    def __init__(self, m1, m2): # m1: input size & m2: output size
        W = init_weights((m1, m2))
        bi = np.zeros(m2)
        bo = np.zeros(m1)
        self.W = theano.shared(W)
        self.bi = theano.shared(bi) # input bias
        self.bo = theano.shared(bo) # output bias
        self.params = [self.W, self.bi, self.bo]

    def forward(self, X):
        return T.nnet.sigmoid(X.dot(self.W) + self.bi)

    def forwardT(self, X):
        return T.nnet.sigmoid(X.dot(self.W.T) + self.bo)


class DeepAutoEncoder(object):
    def __init__(self, hidden_layer_sizes):
        self.hidden_layer_sizes = hidden_layer_sizes

    def fit(self, X, learning_rate=0.5, mu=0.99, epochs=50, batch_sz=100, show_fig=False):
        N, D = X.shape
        n_batches = N / batch_sz

        # create the hidden layers
        mi = D
        self.layers = []
        self.params = []
        for mo in self.hidden_layer_sizes:
            layer = Layer(mi, mo)
            self.layers.append(layer)
            self.params += layer.params
            mi = mo

        X_in = T.matrix('X')
        X_hat = self.forward(X_in)


        # cross entropy cost function
        cost = -(X_in * T.log(X_hat) + (1 - X_in) * T.log(1 - X_hat)).mean()
        cost_op = theano.function(
            inputs=[X_in],
            outputs=cost,
        )

        dparams = [theano.shared(p.get_value()*0) for p in self.params]
        grads = T.grad(cost, self.params)

        updates = [
            (p, p + mu*dp - learning_rate*g) for p, dp, g in zip(self.params, dparams, grads)
        ] + [
            (dp, mu*dp - learning_rate*g) for dp, g in zip(dparams, grads)
        ]
        train_op = theano.function(
            inputs=[X_in],
            outputs=cost,
            updates=updates,
        )

        costs = []
        for i in range(epochs):
            print("epoch:", i)
            X = shuffle(X)
            for j in range(n_batches):
                batch = X[j*batch_sz:(j*batch_sz + batch_sz)]
                c = train_op(batch)
                if j % 100 == 0:
                    print("j / n_batches:", j, "/", n_batches, "cost:", c)
                costs.append(c)
        if show_fig:
            plt.plot(costs)
            plt.show()

    def forward(self, X):
        Z = X
        for layer in self.layers:
            Z = layer.forward(Z)

        self.map2center = theano.function(
            inputs=[X],
            outputs=Z,
        )# returns middle layer

        for i in range(len(self.layers)-1, -1, -1):
            Z = self.layers[i].forwardT(Z)

        return Z


def main():
    Xtrain, Ytrain, Xtest, Ytest = getKaggleMNIST()
    dae = DeepAutoEncoder([500, 300, 2]) # 500 300 2 300 500
    dae.fit(Xtrain)
    mapping = dae.map2center(Xtrain)
    plt.scatter(mapping[:,0], mapping[:,1], c=Ytrain, s=100, alpha=0.5)
    plt.show()

    # purity measure from unsupervised machine learning pt 1
    _, Rfull = gmm(X, 10, max_iter=30)
    print("full purity:", purity(Y, Rfull))
    _, Rreduced = plot_k_means(Z, 10, max_iter=30)
    print("reduced purity:", purity(Y, Rreduced))


if __name__ == '__main__':
    main()

