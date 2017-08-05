import theano
import numpy as np
from theano import tensor as T
from theano import config
import lasagne
from lasagne_average_layer import lasagne_average_layer
import logging
import cPickle

class ppdb_avgcnn_hinge_model(object):

    def __init__(self, We_initial, regfile=None, layersize=300, num_filters=4,
                    filter_size=11, margin=0.4, LC=1e-6, LW=1e-6, updatewords=True,
                    clip=1.0, eta=0.025):

        initial_We = theano.shared(np.asarray(We_initial, dtype = config.floatX))
        We = theano.shared(np.asarray(We_initial, dtype = config.floatX))

        g1batchindices = T.imatrix(); g2batchindices = T.imatrix()
        p1batchindices = T.imatrix(); p2batchindices = T.imatrix()
        g1mask = T.matrix(); g2mask = T.matrix()
        p1mask = T.matrix(); p2mask = T.matrix()

        l_in = lasagne.layers.InputLayer((None, None))
        l_mask = lasagne.layers.InputLayer(shape=(None, None))
        if regfile is None:
            l_emb = lasagne.layers.EmbeddingLayer(l_in, input_size=We.get_value().shape[0], output_size=We.get_value().shape[1], W=We)
            l_average = lasagne_average_layer([l_emb, l_mask])
            l_reshape = lasagne.layers.ReshapeLayer(l_average, ([0], 1, -1))
            l_conv = lasagne.layers.Conv1DLayer(l_reshape,
                                                num_filters=num_filters,
                                                filter_size=filter_size,
                                                stride=1,
                                                pad=filter_size/2,
                                                nonlinearity=lasagne.nonlinearities.rectify)
            l_pooling = lasagne.layers.MaxPool1DLayer(l_conv,
                                                        pool_size=2)
            # (batch_size, params.layersize)
            l_proj = lasagne.layers.DenseLayer(l_pooling, layersize, nonlinearity=lasagne.nonlinearities.tanh)
        else:
            logging.debug("loading regfile: {}".format(regfile))
            p = cPickle.load(file(regfile, 'rb'))
            
            We = theano.shared(np.asarray(p[0].get_value(), dtype = config.floatX))
            W_conv = np.asarray(p[1].get_value(), dtype = config.floatX)
            b_conv = np.asarray(p[2].get_value(), dtype = config.floatX)
            W_proj = np.asarray(p[3].get_value(), dtype = config.floatX)
            b_proj = np.asarray(p[4].get_value(), dtype = config.floatX)
            
            l_emb = lasagne.layers.EmbeddingLayer(l_in, input_size=We.get_value().shape[0], output_size=We.get_value().shape[1], W=We)
            l_average = lasagne_average_layer([l_emb, l_mask])
            l_reshape = lasagne.layers.ReshapeLayer(l_average, ([0], 1, -1))
            l_conv = lasagne.layers.Conv1DLayer(l_reshape,
                                                num_filters=num_filters,
                                                filter_size=filter_size,
                                                stride=1,
                                                pad=filter_size/2,
                                                nonlinearity=lasagne.nonlinearities.rectify,
                                                W=W_conv,
                                                b=b_conv)
            l_pooling = lasagne.layers.MaxPool1DLayer(l_conv,
                                                        pool_size=2)
            # (batch_size, params.layersize)
            l_proj = lasagne.layers.DenseLayer(l_pooling, layersize, nonlinearity=lasagne.nonlinearities.tanh, W=W_proj, b=b_proj)

        embg1 = lasagne.layers.get_output(l_proj, {l_in:g1batchindices, l_mask:g1mask})
        embg2 = lasagne.layers.get_output(l_proj, {l_in:g2batchindices, l_mask:g2mask})
        embp1 = lasagne.layers.get_output(l_proj, {l_in:p1batchindices, l_mask:p1mask})
        embp2 = lasagne.layers.get_output(l_proj, {l_in:p2batchindices, l_mask:p2mask})

        g1g2 = (embg1*embg2).sum(axis=1)
        g1g2norm = T.sqrt(T.sum(embg1**2,axis=1)) * T.sqrt(T.sum(embg2**2,axis=1))
        g1g2 = g1g2 / g1g2norm

        p1g1 = (embp1*embg1).sum(axis=1)
        p1g1norm = T.sqrt(T.sum(embp1**2,axis=1)) * T.sqrt(T.sum(embg1**2,axis=1))
        p1g1 = p1g1 / p1g1norm

        p2g2 = (embp2*embg2).sum(axis=1)
        p2g2norm = T.sqrt(T.sum(embp2**2,axis=1)) * T.sqrt(T.sum(embg2**2,axis=1))
        p2g2 = p2g2 / p2g2norm

        costp1g1 = margin - g1g2 + p1g1
        costp1g1 = costp1g1*(costp1g1 > 0)

        costp2g2 = margin - g1g2 + p2g2
        costp2g2 = costp2g2*(costp2g2 > 0)

        cost = costp1g1 + costp2g2
        network_params = lasagne.layers.get_all_params(l_proj, trainable=True)
        network_params.pop(0)
        self.all_params = lasagne.layers.get_all_params(l_proj, trainable=True)

        l2 = 0.5*LC*sum(lasagne.regularization.l2(x) for x in network_params)
        if updatewords:
            word_reg = 0.5*LW*lasagne.regularization.l2(We-initial_We)
            cost = T.mean(cost) + l2 + word_reg
        else:
            cost = T.mean(cost) + l2

        self.feedforward_function = theano.function([g1batchindices,g1mask], embg1)
        self.cost_function = theano.function([g1batchindices, g2batchindices, p1batchindices, p2batchindices,
                             g1mask, g2mask, p1mask, p2mask], cost)

        prediction = g1g2

        self.scoring_function = theano.function([g1batchindices, g2batchindices,
                             g1mask, g2mask],prediction)

        self.train_function = None
        if updatewords:
            grads = theano.gradient.grad(cost, self.all_params)
            if clip:
                grads = [lasagne.updates.norm_constraint(grad, clip, range(grad.ndim)) for grad in grads]
            updates = lasagne.updates.adam(grads, self.all_params, eta)
            self.train_function = theano.function([g1batchindices, g2batchindices, p1batchindices, p2batchindices,
                             g1mask, g2mask, p1mask, p2mask], cost, updates=updates)
        else:
            self.all_params = network_params
            grads = theano.gradient.grad(cost, self.all_params)
            if clip:
                grads = [lasagne.updates.norm_constraint(grad, clip, range(grad.ndim)) for grad in grads]
            updates = lasagne.updates.adam(grads, self.all_params, eta)
            self.train_function = theano.function([g1batchindices, g2batchindices, p1batchindices, p2batchindices,
                             g1mask, g2mask, p1mask, p2mask], cost, updates=updates)
