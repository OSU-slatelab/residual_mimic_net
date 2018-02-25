import tensorflow as tf

def conv_layer(inputs, filters, training, dropout, strides=1):
    """
    One convolutional layer, with convolutional dropout.

    Parameters
    ----------
    inputs : Tensor
        input to convolutional layer
    filters : int
        size of convolutional layer
    training : boolean placeholder
        whether or not we're training now
    dropout : float
        fraction of filters to drop
    strides : int
        number of input dimensions to shift by for convolution

    Returns
    -------
    Tensor
        outputs of convolutional layer
    """

    # Apply convolution
    layer = tf.layers.conv2d(
        inputs      = inputs,
        filters     = filters,
        kernel_size = 3,
        strides     = strides,
        padding     = 'same',
        data_format = 'channels_first',
        kernel_regularizer = tf.contrib.layers.l2_regularizer(0.001),
    )

    # Dropout
    shape = tf.shape(layer)
    layer = tf.layers.dropout(
        inputs      = layer,
        rate        = dropout,
        noise_shape = [shape[0], filters, 1, 1],
        training    = training,
    )

    return layer


def conv_block(inputs, filters, training, dropout, activation=tf.nn.relu):
    """
    A residual block of three conv layers.

    First layer down-samples using a stride of 2. Output of the third
    layer is added to the first layer as a residual connection.

    Parameters
    ----------
    inputs : Tensor
        Input to this conv block
    filters : int
        Width of this conv block
    training : boolean placeholder
        Whether or not we're in training phase
    dropout : float
        Fraction of filters to drop
    activation : function
        Function to apply after each conv layer

    Returns
    -------
    Tensor
        Output of this conv block
    """
    
    # Down-sample layer
    reduced = conv_layer(inputs, filters, training, dropout, strides=2)
    conv_in = activation(reduced)

    # 1st conv layer
    with tf.variable_scope("conv1"):
        conv1 = conv_layer(conv_in, filters, training, dropout)
        conv1 = activation(conv1)

    # 2nd conv layer
    with tf.variable_scope("conv2"):
        conv2 = conv_layer(conv1, filters, training, dropout)
        conv2 = activation(conv1)

    return reduced + conv2


class Dropnet():
    """
    Dropnet is a resnet-style architecture with convolutional dropout instead of batch norm.
    """

    def __init__(self,
        inputs,
        output_size, 
        filters    = [128, 128, 256, 256],
        fc_layers  = 2,
        fc_nodes   = 2048,
        activation = tf.nn.relu,
        dropout    = 0.3,
        training   = False,
    ):
        """
        Build the graph.

        Parameters
        ----------
        inputs : Placeholder
            Spectral inputs to this model, of the shape (batchsize, frames, frequencies)
        output_size : int
            Size of the output
        filters : list of ints
            Size of each block
        fc_layers : int
            Number of fully-connected hidden layers
        fc_nodes : int
            Number of units to put in each fc hidden layer
        dropout : float
            Fraction of filters and nodes to drop
        training : boolean placeholder
            Whether we are in training or test mode

        Returns
        -------
        Tensor
            Outputs of the dropnet model
        """
        
        # Add channels dimension
        shape = inputs.get_shape().as_list()#tf.shape(inputs)
        block = tf.reshape(inputs, [-1, 1, shape[1], shape[2]])

        # Convolutional part
        for i, f in enumerate(filters):
            with tf.variable_scope("block{0}".format(i)):
                block = conv_block(block, f, training, dropout, activation)

        # Prepare for fully-connected part
        fc = tf.contrib.layers.flatten(block)
        fc = tf.layers.dropout(fc, rate=dropout, training=training)

        # Fully conntected part
        for i in range(fc_layers):
            with tf.variable_scope("fc{0}".format(i)):
                fc = tf.layers.dense(fc, fc_nodes, activation)
                fc = tf.layers.dropout(fc, rate=dropout, training=training)

        self.output = tf.layers.dense(fc, output_size)

