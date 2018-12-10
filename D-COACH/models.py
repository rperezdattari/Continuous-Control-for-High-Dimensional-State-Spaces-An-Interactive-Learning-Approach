import tensorflow.contrib.layers as lays
import tensorflow as tf


def autoencoder(ae_inputs):
    # encoder
    net = lays.conv2d(ae_inputs, 16, [3, 3], stride=2, padding='SAME')
    net = lays.conv2d(net, 8, [3, 3], stride=2, padding='SAME')
    net = lays.conv2d(net, 4, [3, 3], stride=2, padding='SAME')
    latent_space = tf.identity(net, name='conv_part')

    # decoder
    net = lays.conv2d_transpose(latent_space, 8, [3, 3], stride=2, padding='SAME')
    net = lays.conv2d_transpose(net, 16, [3, 3], stride=2, padding='SAME')
    net = lays.conv2d_transpose(net, 1, [3, 3], stride=2, padding='SAME', activation_fn=tf.nn.sigmoid)
    ae_output = tf.identity(net, name='ae_output')

    # calculate the loss and optimize the network
    loss = tf.reduce_mean(tf.square(ae_output - ae_inputs))

    return loss, latent_space, ae_output


def fully_connected_layers(x, dim_a, fc_layers_neurons, loss_function_type):
    # Fully connected layer (in tf contrib folder for now)
    fc1 = tf.layers.dense(x, fc_layers_neurons, activation=tf.nn.tanh)
    fc2 = tf.layers.dense(fc1, fc_layers_neurons, activation=tf.nn.tanh)

    # Output layer, class prediction
    y = tf.layers.dense(fc2, dim_a, activation=tf.nn.tanh, name='action')

    y_ = tf.placeholder(tf.float32, [None, dim_a], name='policy_label')

    # define the loss function
    if loss_function_type == 'cross_entropy':
        loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    elif loss_function_type == 'mean_squared':
        loss = 0.5 * tf.reduce_mean(tf.square(y - y_))
    else:
        loss = None
        print('No existing loss function was selected, please try mean_squared or cross_entropy')
        exit()

    return y, loss


def recurrent_layers(x, dim_a, loss_function_type, h_size):
    x_shape = x.get_shape()
    batch_size = tf.placeholder(tf.int32, [], name='batch_size')
    sequence_length = tf.placeholder(tf.int32, name='sequence_length')

    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
    #x = tf.reshape(x, shape=[batch_size, sequence_length, x_shape[1], x_shape[2], x_shape[3]])
    x = tf.reshape(x, shape=[batch_size, sequence_length, x_shape[2]])

    # Flatten the data to a 1-D vector for the fully connected layer
    #x_flatten = tf.contrib.layers.flatten(x)
    #x_flatten = tf.reshape(x_flatten, shape=[batch_size, sequence_length, x_shape[1] * x_shape[2] * x_shape[3]])

    # LSTM cell
    cell = tf.nn.rnn_cell.LSTMCell(h_size, state_is_tuple=False)
    state_in = cell.zero_state(batch_size, tf.float32)
    state_in = tf.identity(state_in, name='state_in')

    # Recurrence
    rnn_output, state_out = tf.nn.dynamic_rnn(
            cell,
            x,
            initial_state=state_in,
            dtype=tf.float32)

    state_out = tf.identity(state_out, name='state_out')

    y = tf.reshape(tf.layers.dense(rnn_output[:, -1, :],
                                   dim_a, activation=tf.nn.tanh), [-1, dim_a], name='action')
    y_ = tf.placeholder(tf.float32, [None, dim_a], name='model_label')

    # define the loss function
    if loss_function_type == 'cross_entropy':
        loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    elif loss_function_type == 'mean_squared':
        loss = 0.5 * tf.reduce_mean(tf.square(y - y_))
    else:
        loss = None
        print('No existing loss function was selected, please try mean_squared or cross_entropy')
        exit()

    return y, loss, state_out
