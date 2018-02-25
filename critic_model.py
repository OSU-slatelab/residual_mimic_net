import tensorflow as tf
from dropnet import Dropnet

class Critic:
    """
    This critic model takes clean speech as input, and outputs senone labels.

    As part of the actor-critic model, this is trained jointly with
    the actor, by freezing the weights after it has been
    trained on clean speech.
    """

    def __init__(self,
            inputs,
            output_size = 1999,
            filters     = [128, 128, 256, 256],
            fc_layers   = 2,
            fc_nodes    = 2048,
            activation  = tf.nn.relu,
            dropout     = 0.3):
        """
        Create critic model.

        Parameters
        ----------
        inputs : Tensor
            The input placeholder or tensor from actor
        output_size : int
            Number of classes to output
        filters : list
            The number of filters in each block
        fc_layers : int
            Number of fully-connected hidden layers
        fc_nodes : int
            Number of units per fully connected hidden layer
        activation : function
            Function to apply to each layer as an activation
        dropout : float
            Proportion of filters and units to drop at train time
        """

        # Placeholders
        self.inputs = inputs
        self.training = tf.placeholder(dtype = tf.bool, name = "training")
        self.labels = tf.placeholder(dtype = tf.float32, shape = (None, output_size), name = "labels")

        self.dropnet = Dropnet(
            inputs      = inputs,
            output_size = output_size,
            filters     = filters,
            fc_layers   = fc_layers,
            fc_nodes    = fc_nodes,
            activation  = activation,
            dropout     = dropout,
        )

        self.outputs = self.dropnet.output


