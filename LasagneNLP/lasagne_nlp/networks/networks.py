__author__ = 'max'

import lasagne
import lasagne.nonlinearities as nonlinearities
from lasagne.layers import Gate, SliceLayer
from lasagne_nlp.networks.crf import CRFLayer
from lasagne_nlp.networks.highway import HighwayDenseLayer


def build_BiRNN(incoming, num_units, mask=None, grad_clipping=0, nonlinearity=nonlinearities.tanh,
                precompute_input=True, dropout=True, in_to_out=False):
    # construct the forward and backward rnns. Now, Ws are initialized by He initializer with default arguments.
    # Need to try other initializers for specific tasks.

    # dropout for incoming
    if dropout:
        incoming = lasagne.layers.DropoutLayer(incoming, p=0.5)

    rnn_forward = lasagne.layers.RecurrentLayer(incoming, num_units,
                                                mask_input=mask, grad_clipping=grad_clipping,
                                                nonlinearity=nonlinearity, precompute_input=precompute_input,
                                                W_in_to_hid=lasagne.init.GlorotUniform(),
                                                W_hid_to_hid=lasagne.init.GlorotUniform(), name='forward')
    rnn_backward = lasagne.layers.RecurrentLayer(incoming, num_units,
                                                 mask_input=mask, grad_clipping=grad_clipping,
                                                 nonlinearity=nonlinearity, precompute_input=precompute_input,
                                                 W_in_to_hid=lasagne.init.GlorotUniform(),
                                                 W_hid_to_hid=lasagne.init.GlorotUniform(), backwards=True,
                                                 name='backward')

    # concatenate the outputs of forward and backward RNNs to combine them.
    concat = lasagne.layers.concat([rnn_forward, rnn_backward], axis=2, name="bi-rnn")

    # dropout for output
    if dropout:
        concat = lasagne.layers.DropoutLayer(concat, p=0.5)

    if in_to_out:
        concat = lasagne.layers.concat([concat, incoming], axis=2)

    # the shape of BiRNN output (concat) is (batch_size, input_length, 2 * num_hidden_units)
    return concat


def build_BiLSTM(incoming, num_units, mask=None, grad_clipping=0, precompute_input=True, peepholes=False, dropout=True,
                 in_to_out=False):
    # construct the forward and backward rnns. Now, Ws are initialized by Glorot initializer with default arguments.
    # Need to try other initializers for specific tasks.

    # dropout for incoming
    if dropout:
        incoming = lasagne.layers.DropoutLayer(incoming, p=0.5)

    ingate_forward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(),
                          W_cell=lasagne.init.Uniform(range=0.1))
    outgate_forward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(),
                           W_cell=lasagne.init.Uniform(range=0.1))
    # according to Jozefowicz et al.(2015), init bias of forget gate to 1.
    forgetgate_forward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(),
                              W_cell=lasagne.init.Uniform(range=0.1), b=lasagne.init.Constant(1.))
    # now use tanh for nonlinear function of cell, need to try pure linear cell
    cell_forward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(), W_cell=None,
                        nonlinearity=nonlinearities.tanh)
    lstm_forward = lasagne.layers.LSTMLayer(incoming, num_units, mask_input=mask, grad_clipping=grad_clipping,
                                            nonlinearity=nonlinearities.tanh, peepholes=peepholes,
                                            precompute_input=precompute_input,
                                            ingate=ingate_forward, outgate=outgate_forward,
                                            forgetgate=forgetgate_forward, cell=cell_forward, name='forward')

    ingate_backward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(),
                           W_cell=lasagne.init.Uniform(range=0.1))
    outgate_backward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(),
                            W_cell=lasagne.init.Uniform(range=0.1))
    # according to Jozefowicz et al.(2015), init bias of forget gate to 1.
    forgetgate_backward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(),
                               W_cell=lasagne.init.Uniform(range=0.1), b=lasagne.init.Constant(1.))
    # now use tanh for nonlinear function of cell, need to try pure linear cell
    cell_backward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(), W_cell=None,
                         nonlinearity=nonlinearities.tanh)
    lstm_backward = lasagne.layers.LSTMLayer(incoming, num_units, mask_input=mask, grad_clipping=grad_clipping,
                                             nonlinearity=nonlinearities.tanh, peepholes=peepholes,
                                             precompute_input=precompute_input, backwards=True,
                                             ingate=ingate_backward, outgate=outgate_backward,
                                             forgetgate=forgetgate_backward, cell=cell_backward, name='backward')

    # concatenate the outputs of forward and backward RNNs to combine them.
    concat = lasagne.layers.concat([lstm_forward, lstm_backward], axis=2, name="bi-lstm")

    # dropout for output
    if dropout:
        concat = lasagne.layers.DropoutLayer(concat, p=0.5)

    if in_to_out:
        concat = lasagne.layers.concat([concat, incoming], axis=2)

    # the shape of BiRNN output (concat) is (batch_size, input_length, 2 * num_hidden_units)
    #print 'Bi-LSTM'
    #print concat.output_shape
    return concat


def build_BiLSTM_char(incoming, num_units, mask=None, grad_clipping=0, precompute_input=True, peepholes=False, dropout=True,
                 in_to_out=False):
    # construct the forward and backward rnns. Now, Ws are initialized by Glorot initializer with default arguments.
    # Need to try other initializers for specific tasks.

    # dropout for incoming
    if dropout:
        incoming = lasagne.layers.DropoutLayer(incoming, p=0.5)

    ingate_forward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(),
                          W_cell=lasagne.init.Uniform(range=0.1))
    outgate_forward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(),
                           W_cell=lasagne.init.Uniform(range=0.1))
    # according to Jozefowicz et al.(2015), init bias of forget gate to 1.
    forgetgate_forward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(),
                              W_cell=lasagne.init.Uniform(range=0.1), b=lasagne.init.Constant(1.))
    # now use tanh for nonlinear function of cell, need to try pure linear cell
    cell_forward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(), W_cell=None,
                        nonlinearity=nonlinearities.tanh)
    lstm_forward = lasagne.layers.LSTMLayer(incoming, num_units, mask_input=mask, grad_clipping=grad_clipping,
                                            nonlinearity=nonlinearities.tanh, peepholes=peepholes,
                                            precompute_input=precompute_input,
                                            ingate=ingate_forward, outgate=outgate_forward,
                                            forgetgate=forgetgate_forward, cell=cell_forward, name='forward')

    ingate_backward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(),
                           W_cell=lasagne.init.Uniform(range=0.1))
    outgate_backward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(),
                            W_cell=lasagne.init.Uniform(range=0.1))
    # according to Jozefowicz et al.(2015), init bias of forget gate to 1.
    forgetgate_backward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(),
                               W_cell=lasagne.init.Uniform(range=0.1), b=lasagne.init.Constant(1.))
    # now use tanh for nonlinear function of cell, need to try pure linear cell
    cell_backward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(), W_cell=None,
                         nonlinearity=nonlinearities.tanh)
    lstm_backward = lasagne.layers.LSTMLayer(incoming, num_units, mask_input=mask, grad_clipping=grad_clipping,
                                             nonlinearity=nonlinearities.tanh, peepholes=peepholes,
                                             precompute_input=precompute_input, backwards=True,
                                             ingate=ingate_backward, outgate=outgate_backward,
                                             forgetgate=forgetgate_backward, cell=cell_backward, name='backward')
    print lstm_forward.output_shape
    print lstm_backward.output_shape
    lstm_forward = SliceLayer(lstm_forward, indices=-1, axis=1)
    lstm_backward = SliceLayer(lstm_backward, indices=-1, axis=1)
    print lstm_forward.output_shape
    print lstm_backward.output_shape
    # concatenate the outputs of forward and backward RNNs to combine them.
    concat = lasagne.layers.concat([lstm_forward, lstm_backward], axis=1, name="bi-lstm")
    print concat.output_shape
    # dropout for output
    if dropout:
        concat = lasagne.layers.DropoutLayer(concat, p=0.5)

    if in_to_out:
        concat = lasagne.layers.concat([concat, incoming], axis=2)

    # the shape of BiRNN output (concat) is (batch_size, input_length, 2 * num_hidden_units)
    #print 'Bi-LSTM'
    #print concat.output_shape
    return concat



def build_BiLSTM_2(incoming, num_units, mask=None, grad_clipping=0, precompute_input=True, peepholes=False, dropout=True,
                 in_to_out=False):
    # construct the forward and backward rnns. Now, Ws are initialized by Glorot initializer with default arguments.
    # Need to try other initializers for specific tasks.

    # dropout for incoming
    if dropout:
        incoming = lasagne.layers.DropoutLayer(incoming, p=0.5)

    ingate_forward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(),
                          W_cell=lasagne.init.Uniform(range=0.1))
    outgate_forward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(),
                           W_cell=lasagne.init.Uniform(range=0.1))
    # according to Jozefowicz et al.(2015), init bias of forget gate to 1.
    forgetgate_forward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(),
                              W_cell=lasagne.init.Uniform(range=0.1), b=lasagne.init.Constant(1.))
    # now use tanh for nonlinear function of cell, need to try pure linear cell
    cell_forward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(), W_cell=None,
                        nonlinearity=nonlinearities.tanh)
    lstm_forward = lasagne.layers.LSTMLayer(incoming, num_units, mask_input=mask, grad_clipping=grad_clipping,
                                            nonlinearity=nonlinearities.tanh, peepholes=peepholes,
                                            precompute_input=precompute_input,
                                            ingate=ingate_forward, outgate=outgate_forward,
                                            forgetgate=forgetgate_forward, cell=cell_forward, name='forward')

    ingate_backward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(),
                           W_cell=lasagne.init.Uniform(range=0.1))
    outgate_backward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(),
                            W_cell=lasagne.init.Uniform(range=0.1))
    # according to Jozefowicz et al.(2015), init bias of forget gate to 1.
    forgetgate_backward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(),
                               W_cell=lasagne.init.Uniform(range=0.1), b=lasagne.init.Constant(1.))
    # now use tanh for nonlinear function of cell, need to try pure linear cell
    cell_backward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(), W_cell=None,
                         nonlinearity=nonlinearities.tanh)
    lstm_backward = lasagne.layers.LSTMLayer(incoming, num_units, mask_input=mask, grad_clipping=grad_clipping,
                                             nonlinearity=nonlinearities.tanh, peepholes=peepholes,
                                             precompute_input=precompute_input, backwards=True,
                                             ingate=ingate_backward, outgate=outgate_backward,
                                             forgetgate=forgetgate_backward, cell=cell_backward, name='backward')

    # concatenate the outputs of forward and backward RNNs to combine them.
    concat = lasagne.layers.concat([lstm_forward, lstm_backward], axis=2, name="bi-lstm")

    # dropout for output
    if dropout:
        concat = lasagne.layers.DropoutLayer(concat, p=0.5)

    if in_to_out:
        concat = lasagne.layers.concat([concat, incoming], axis=2)

    if dropout:
        incoming = lasagne.layers.DropoutLayer(concat, p=0.5)

    ingate_forward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(),
                          W_cell=lasagne.init.Uniform(range=0.1))
    outgate_forward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(),
                           W_cell=lasagne.init.Uniform(range=0.1))
    # according to Jozefowicz et al.(2015), init bias of forget gate to 1.
    forgetgate_forward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(),
                              W_cell=lasagne.init.Uniform(range=0.1), b=lasagne.init.Constant(1.))
    # now use tanh for nonlinear function of cell, need to try pure linear cell
    cell_forward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(), W_cell=None,
                        nonlinearity=nonlinearities.tanh)
    lstm_forward = lasagne.layers.LSTMLayer(incoming, num_units, mask_input=mask, grad_clipping=grad_clipping,
                                            nonlinearity=nonlinearities.tanh, peepholes=peepholes,
                                            precompute_input=precompute_input,
                                            ingate=ingate_forward, outgate=outgate_forward,
                                            forgetgate=forgetgate_forward, cell=cell_forward, name='forward')

    ingate_backward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(),
                           W_cell=lasagne.init.Uniform(range=0.1))
    outgate_backward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(),
                            W_cell=lasagne.init.Uniform(range=0.1))
    # according to Jozefowicz et al.(2015), init bias of forget gate to 1.
    forgetgate_backward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(),
                               W_cell=lasagne.init.Uniform(range=0.1), b=lasagne.init.Constant(1.))
    # now use tanh for nonlinear function of cell, need to try pure linear cell
    cell_backward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(), W_cell=None,
                         nonlinearity=nonlinearities.tanh)
    lstm_backward = lasagne.layers.LSTMLayer(incoming, num_units, mask_input=mask, grad_clipping=grad_clipping,
                                             nonlinearity=nonlinearities.tanh, peepholes=peepholes,
                                             precompute_input=precompute_input, backwards=True,
                                             ingate=ingate_backward, outgate=outgate_backward,
                                             forgetgate=forgetgate_backward, cell=cell_backward, name='backward')

    # concatenate the outputs of forward and backward RNNs to combine them.
    concat = lasagne.layers.concat([lstm_forward, lstm_backward], axis=2, name="bi-lstm")

    # dropout for output
    if dropout:
        concat = lasagne.layers.DropoutLayer(concat, p=0.5)

    if in_to_out:
        concat = lasagne.layers.concat([concat, incoming], axis=2)

    # the shape of BiRNN output (concat) is (batch_size, input_length, 2 * num_hidden_units)
    return concat



def build_BiGRU(incoming, num_units, mask=None, grad_clipping=0, precompute_input=True, dropout=True, in_to_out=False):
    # construct the forward and backward grus. Now, Ws are initialized by Glorot initializer with default arguments.
    # Need to try other initializers for specific tasks.

    # dropout for incoming
    if dropout:
        incoming = lasagne.layers.DropoutLayer(incoming, p=0.5)

    # according to Jozefowicz et al.(2015), init bias of forget gate to 1.
    resetgate_forward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(),
                             W_cell=lasagne.init.Uniform(range=0.1), b=lasagne.init.Constant(1.))
    updategate_forward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(),
                              W_cell=lasagne.init.Uniform(range=0.1))
    # now use tanh for nonlinear function of hidden gate
    hidden_forward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(), W_cell=None,
                          nonlinearity=nonlinearities.tanh)
    gru_forward = lasagne.layers.GRULayer(incoming, num_units, mask_input=mask, grad_clipping=grad_clipping,
                                          precompute_input=precompute_input,
                                          resetgate=resetgate_forward, updategate=updategate_forward,
                                          hidden_update=hidden_forward, name='forward')

    # according to Jozefowicz et al.(2015), init bias of forget gate to 1.
    resetgate_backward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(),
                              W_cell=lasagne.init.Uniform(range=0.1), b=lasagne.init.Constant(1.))
    updategate_backward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(),
                               W_cell=lasagne.init.Uniform(range=0.1))
    # now use tanh for nonlinear function of hidden gate
    hidden_backward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(), W_cell=None,
                           nonlinearity=nonlinearities.tanh)
    gru_backward = lasagne.layers.GRULayer(incoming, num_units, mask_input=mask, grad_clipping=grad_clipping,
                                           precompute_input=precompute_input, backwards=True,
                                           resetgate=resetgate_backward, updategate=updategate_backward,
                                           hidden_update=hidden_backward, name='backward')

    # concatenate the outputs of forward and backward GRUs to combine them.
    concat = lasagne.layers.concat([gru_forward, gru_backward], axis=2, name="bi-gru")

    # dropout for output
    if dropout:
        concat = lasagne.layers.DropoutLayer(concat, p=0.5)

    if in_to_out:
        concat = lasagne.layers.concat([concat, incoming], axis=2)

    # the shape of BiRNN output (concat) is (batch_size, input_length, 2 * num_hidden_units)
    return concat


def build_BiRNN_CNN(incoming1, incoming2, num_units, mask=None, grad_clipping=0, nonlinearity=nonlinearities.tanh,
                    precompute_input=True, num_filters=20, dropout=True, in_to_out=False):
    # first get some necessary dimensions or parameters
    conv_window = 3
    _, sent_length, _ = incoming2.output_shape

    # dropout before cnn?
    if dropout:
        incoming1 = lasagne.layers.DropoutLayer(incoming1, p=0.5)

    # construct convolution layer
    cnn_layer = lasagne.layers.Conv1DLayer(incoming1, num_filters=num_filters, filter_size=conv_window, pad='full',
                                           nonlinearity=lasagne.nonlinearities.tanh, name='cnn')
    # infer the pool size for pooling (pool size should go through all time step of cnn)
    _, _, pool_size = cnn_layer.output_shape
    # construct max pool layer
    pool_layer = lasagne.layers.MaxPool1DLayer(cnn_layer, pool_size=pool_size)
    # reshape the layer to match rnn incoming layer [batch * sent_length, num_filters, 1] --> [batch, sent_length, num_filters]
    output_cnn_layer = lasagne.layers.reshape(pool_layer, (-1, sent_length, [1]))

    # finally, concatenate the two incoming layers together.
    incoming = lasagne.layers.concat([output_cnn_layer, incoming2], axis=2)

    return build_BiRNN(incoming, num_units, mask=mask, grad_clipping=grad_clipping, nonlinearity=nonlinearity,
                       precompute_input=precompute_input, dropout=dropout, in_to_out=in_to_out)


def build_BiLSTM_CNN(incoming1, incoming2, num_units, mask=None, grad_clipping=0, precompute_input=True,
                     peepholes=False, num_filters=20, dropout=True, in_to_out=False):
    # first get some necessary dimensions or parameters
    conv_window = 3
    _, sent_length, _ = incoming2.output_shape

    # dropout before cnn?
    if dropout:
        incoming1 = lasagne.layers.DropoutLayer(incoming1, p=0.5)

    # construct convolution layer
    cnn_layer = lasagne.layers.Conv1DLayer(incoming1, num_filters=num_filters, filter_size=conv_window, pad='full',
                                           nonlinearity=lasagne.nonlinearities.tanh, name='cnn')
    print 'Incomming'
    print incoming1.output_shape
    print incoming2.output_shape
    print 'CNN'
    print cnn_layer.output_shape
    # infer the pool size for pooling (pool size should go through all time step of cnn)
    _, _, pool_size = cnn_layer.output_shape
    # construct max pool layer
    pool_layer = lasagne.layers.MaxPool1DLayer(cnn_layer, pool_size=pool_size)
    print pool_layer.output_shape
    # reshape the layer to match lstm incoming layer [batch * sent_length, num_filters, 1] --> [batch, sent_length, num_filters]
    output_cnn_layer = lasagne.layers.reshape(pool_layer, (-1, sent_length, [1]))
    print output_cnn_layer.output_shape
    # finally, concatenate the two incoming layers together.
    incoming = lasagne.layers.concat([output_cnn_layer, incoming2], axis=2)
    print 'CNN-LSTM'
    print incoming.output_shape

    return build_BiLSTM(incoming, num_units, mask=mask, grad_clipping=grad_clipping, peepholes=peepholes,
                        precompute_input=precompute_input, dropout=dropout, in_to_out=in_to_out)


def build_BiLSTM_LSTM(incoming1, incoming2, num_units_word, num_units_char, mask=None, grad_clipping=0,
                      precompute_input=True, peepholes=False, dropout=True, in_to_out=False):
    # first get some necessary dimensions or parameters
    _, sent_length, _ = incoming2.output_shape
    print 'Incomming'
    print incoming1.output_shape
    print incoming2.output_shape

    # dropout before cnn?
    if dropout:
        incoming1 = lasagne.layers.DropoutLayer(incoming1, p=0.5)

    output_lstm_layer = build_BiLSTM_char(incoming1, num_units_char, mask=mask, grad_clipping=grad_clipping, peepholes=peepholes,
                                     precompute_input=precompute_input, dropout=dropout, in_to_out=in_to_out)
    print 'Char-LSTM'
    print output_lstm_layer.output_shape
    output_lstm_layer = lasagne.layers.reshape(output_lstm_layer, (-1, sent_length, [1]))

    # finally, concatenate the two incoming layers together.
    incoming = lasagne.layers.concat([output_lstm_layer, incoming2], axis=2)
    print incoming.output_shape

    return build_BiLSTM(incoming, num_units_word, mask=mask, grad_clipping=grad_clipping, peepholes=peepholes,
                        precompute_input=precompute_input, dropout=dropout, in_to_out=in_to_out)


def build_BiLSTM_2_CNN(incoming1, incoming2, num_units, mask=None, grad_clipping=0, precompute_input=True,
                     peepholes=False, num_filters=20, dropout=True, in_to_out=False):
    # first get some necessary dimensions or parameters
    conv_window = 3
    _, sent_length, _ = incoming2.output_shape

    # dropout before cnn?
    if dropout:
        incoming1 = lasagne.layers.DropoutLayer(incoming1, p=0.5)

    # construct convolution layer
    cnn_layer = lasagne.layers.Conv1DLayer(incoming1, num_filters=num_filters, filter_size=conv_window, pad='full',
                                           nonlinearity=lasagne.nonlinearities.tanh, name='cnn')
    # infer the pool size for pooling (pool size should go through all time step of cnn)
    _, _, pool_size = cnn_layer.output_shape
    # construct max pool layer
    pool_layer = lasagne.layers.MaxPool1DLayer(cnn_layer, pool_size=pool_size)
    # reshape the layer to match lstm incoming layer [batch * sent_length, num_filters, 1] --> [batch, sent_length, num_filters]
    output_cnn_layer = lasagne.layers.reshape(pool_layer, (-1, sent_length, [1]))

    # finally, concatenate the two incoming layers together.
    incoming = lasagne.layers.concat([output_cnn_layer, incoming2], axis=2)

    return build_BiLSTM_2(incoming, num_units, mask=mask, grad_clipping=grad_clipping, peepholes=peepholes,
                        precompute_input=precompute_input, dropout=dropout, in_to_out=in_to_out)


def build_BiGRU_CNN(incoming1, incoming2, num_units, mask=None, grad_clipping=0, precompute_input=True,
                    num_filters=20, dropout=True, in_to_out=False):
    # first get some necessary dimensions or parameters
    conv_window = 3
    _, sent_length, _ = incoming2.output_shape

    # dropout before cnn?
    if dropout:
        incoming1 = lasagne.layers.DropoutLayer(incoming1, p=0.5)

    # construct convolution layer
    cnn_layer = lasagne.layers.Conv1DLayer(incoming1, num_filters=num_filters, filter_size=conv_window, pad='full',
                                           nonlinearity=lasagne.nonlinearities.tanh, name='cnn')
    # infer the pool size for pooling (pool size should go through all time step of cnn)
    _, _, pool_size = cnn_layer.output_shape
    # construct max pool layer
    pool_layer = lasagne.layers.MaxPool1DLayer(cnn_layer, pool_size=pool_size)
    # reshape the layer to match lstm incoming layer [batch * sent_length, num_filters, 1] --> [batch, sent_length, num_filters]
    output_cnn_layer = lasagne.layers.reshape(pool_layer, (-1, sent_length, [1]))

    # finally, concatenate the two incoming layers together.
    incoming = lasagne.layers.concat([output_cnn_layer,     incoming2], axis=2)

    return build_BiGRU(incoming, num_units, mask=mask, grad_clipping=grad_clipping, precompute_input=precompute_input,
                       dropout=dropout, in_to_out=in_to_out)


def build_BiLSTM_CNN_CRF(incoming1, incoming2, num_units, num_labels, mask=None, grad_clipping=0, precompute_input=True,
                         peepholes=False, num_filters=20, dropout=True, in_to_out=False):
    bi_lstm_cnn = build_BiLSTM_CNN(incoming1, incoming2, num_units, mask=mask, grad_clipping=grad_clipping,
                                   precompute_input=precompute_input, peepholes=peepholes,
                                   num_filters=num_filters, dropout=dropout, in_to_out=in_to_out)
    print bi_lstm_cnn.output_shape
    return CRFLayer(bi_lstm_cnn, num_labels, mask_input=mask)


def build_BiLSTM_LSTM_CRF(incoming1, incoming2, num_units_word, num_units_char, num_labels, mask=None, grad_clipping=0,
                          precompute_input=True, peepholes=False, dropout=True, in_to_out=False):
    bi_lstm_lstm = build_BiLSTM_LSTM(incoming1, incoming2, num_units_word, num_units_char, mask=mask,
                                    grad_clipping=grad_clipping, precompute_input=precompute_input, peepholes=peepholes,
                                    dropout=dropout, in_to_out=in_to_out)
    print bi_lstm_lstm.output_shape
    #return CRFLayer(bi_lstm_cnn, num_labels, mask_input=mask)
    return bi_lstm_lstm


def build_BiLSTM_2_CNN_CRF(incoming1, incoming2, num_units, num_labels, mask=None, grad_clipping=0, precompute_input=True,
                         peepholes=False, num_filters=20, dropout=True, in_to_out=False):
    bi_lstm_cnn = build_BiLSTM_2_CNN(incoming1, incoming2, num_units, mask=mask, grad_clipping=grad_clipping,
                                   precompute_input=precompute_input, peepholes=peepholes,
                                   num_filters=num_filters, dropout=dropout, in_to_out=in_to_out)

    return CRFLayer(bi_lstm_cnn, num_labels, mask_input=mask)


def build_BiLSTM_CRF(incoming, num_units, num_labels, mask=None, grad_clipping=0, precompute_input=True,
                         peepholes=False, dropout=True, in_to_out=False):
    bi_lstm = build_BiLSTM(incoming, num_units, mask=mask, grad_clipping=grad_clipping,
                           precompute_input=precompute_input, peepholes=peepholes, dropout=dropout, in_to_out=in_to_out)

    return CRFLayer(bi_lstm, num_labels, mask_input=mask)


def build_BiLSTM_HighCNN(incoming1, incoming2, num_units, mask=None, grad_clipping=0, precompute_input=True,
                         peepholes=False, num_filters=20, dropout=True, in_to_out=False):
    # first get some necessary dimensions or parameters
    conv_window = 3
    _, sent_length, _ = incoming2.output_shape

    # dropout before cnn
    if dropout:
        incoming1 = lasagne.layers.DropoutLayer(incoming1, p=0.5)

    # construct convolution layer
    cnn_layer = lasagne.layers.Conv1DLayer(incoming1, num_filters=num_filters, filter_size=conv_window, pad='full',
                                           nonlinearity=lasagne.nonlinearities.tanh, name='cnn')
    # infer the pool size for pooling (pool size should go through all time step of cnn)
    _, _, pool_size = cnn_layer.output_shape
    # construct max pool layer
    pool_layer = lasagne.layers.MaxPool1DLayer(cnn_layer, pool_size=pool_size)
    # reshape the layer to match highway incoming layer [batch * sent_length, num_filters, 1] --> [batch * sent_length, num_filters]
    output_cnn_layer = lasagne.layers.reshape(pool_layer, ([0], -1))

    # dropout after cnn?
    # if dropout:
    # output_cnn_layer = lasagne.layers.DropoutLayer(output_cnn_layer, p=0.5)

    # construct highway layer
    highway_layer = HighwayDenseLayer(output_cnn_layer, nonlinearity=nonlinearities.rectify)

    # reshape the layer to match lstm incoming layer [batch * sent_length, num_filters] --> [batch, sent_length, number_filters]
    output_highway_layer = lasagne.layers.reshape(highway_layer, (-1, sent_length, [1]))

    # finally, concatenate the two incoming layers together.
    incoming = lasagne.layers.concat([output_highway_layer, incoming2], axis=2)

    return build_BiLSTM(incoming, num_units, mask=mask, grad_clipping=grad_clipping, peepholes=peepholes,
                        precompute_input=precompute_input, dropout=dropout, in_to_out=in_to_out)


def build_BiLSTM_HighCNN_CRF(incoming1, incoming2, num_units, num_labels, mask=None, grad_clipping=0,
                             precompute_input=True, peepholes=False, num_filters=20, dropout=True, in_to_out=False):
    bi_lstm_cnn = build_BiLSTM_HighCNN(incoming1, incoming2, num_units, mask=mask, grad_clipping=grad_clipping,
                                       precompute_input=precompute_input, peepholes=peepholes,
                                       num_filters=num_filters, dropout=dropout, in_to_out=in_to_out)

    return CRFLayer(bi_lstm_cnn, num_labels, mask_input=mask)
