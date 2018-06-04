import tensorflow as tf

class ResNet():
    """
    ResNet-style architecture for speech denoising.
    """

    def __init__(self,
        inputs,
        output_size, 
        filters    = [128, 128, 256, 256],
        fc_layers  = 2,
        fc_nodes   = 2048,
        activation = tf.nn.relu,
        dropout    = 0.3,
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
        activation : function
            Function to apply before conv layers as an activation
        dropout : float
            Fraction of filters and nodes to drop

        Returns
        -------
        Tensor
            Outputs of the dropnet model
        """

        # Store hyperparameters
        self.inputs = inputs
        self.activation = activation
        self.dropout = dropout
        self.training = tf.placeholder(tf.bool)
        
        # Add channels dimension
        shape = inputs.get_shape().as_list()
        block = tf.reshape(inputs, [-1, 1, shape[1], shape[2]])

        # Convolutional part
        for i, f in enumerate(filters):
            with tf.variable_scope("block{0}".format(i)):
                block = self.conv_block(block, f)

        # Prepare for fully-connected part
        fc = tf.contrib.layers.flatten(block)
        fc = tf.layers.dropout(fc, rate=dropout, training=self.training)

        # Fully conntected part
        for i in range(fc_layers):
            with tf.variable_scope("fc{0}".format(i)):
                fc = tf.layers.dense(fc, fc_nodes, activation)
                fc = tf.layers.dropout(fc, rate=dropout, training=self.training)

        self.outputs = tf.layers.dense(fc, output_size)


    def conv_layer(self, inputs, filters, downsample=False):
        """
        One convolutional layer, with convolutional dropout.

        Parameters
        ----------
        inputs : Tensor
            input to convolutional layer
        filters : int
            size of convolutional layer
        downsample : boolean
            Whether or not this is a downsampling layer

        Returns
        -------
        Tensor
            outputs of convolutional layer
        """

        # DO pre-activation if we're not downsampling
        if not downsample:
            inputs = self.activation(inputs)

        # Apply convolution
        layer = tf.layers.conv2d(
            inputs      = inputs,
            filters     = filters,
            kernel_size = 3,
            strides     = 2 if downsample else 1,
            padding     = 'same',
            data_format = 'channels_first',
            kernel_regularizer = tf.contrib.layers.l2_regularizer(0.0001),
        )

        # Dropout
        shape = tf.shape(layer)
        dropped = tf.layers.dropout(
            inputs      = layer,
            rate        = self.dropout,
            training    = self.training,
            noise_shape = [shape[0], filters, 1, 1],
        )

        return layer, dropped


    def conv_block(self, inputs, filters):
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

        Returns
        -------
        Tensor
            Output of this conv block
        """
        
        # Down-sample layer
        with tf.variable_scope("downsample"):
            reduced, dropped = self.conv_layer(inputs, filters, downsample=True)

        # 1st conv layer
        with tf.variable_scope("conv1"):
            _, conv1 = self.conv_layer(dropped, filters)

        # 2nd conv layer
        with tf.variable_scope("conv2"):
            _, conv2 = self.conv_layer(conv1, filters)

        return reduced + conv2


