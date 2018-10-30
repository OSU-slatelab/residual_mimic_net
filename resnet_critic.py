#!/u/drspeech/opt/anaconda3/bin/python3

"""
Critic model for actor-critic noisy speech recognition.

Author: Deblin Bagchi and Peter Plantinga
Date:   Fall 2017
"""

import tensorflow as tf
from resnet import ResNet

class Critic:
    """
    This critic model takes clean speech as input, and outputs senone labels.

    As part of the actor-critic model, this is trained jointly with
    the actor, by freezing the weights after it has been
    trained on clean speech.
    """

    def __init__(self,
            inputs,
            fc_nodes    = 2048,
            fc_layers   = 2,
            output_size = 1999,
            dropout     = 0.5,
            context     = 5,
        ):
        """
        Create critic model.

        Params:
         * inputs : Tensor
            The input placeholder or tensor from actor
         * fc_nodes : int
            The size of the DNN layers
         * fc_layers : int
            Number of layers
         * output_size : int
            Number of classes to output
         * dropout : float
            Proportion of neurons to drop
         * context : int
            Number of past and future frames to include in the input
        """

        self.inputs = inputs
        
        critic_inputs = tf.stack([inputs[i:-(2*context-i)] for i in range(2*context)] + [inputs[2*context:]])
        critic_inputs = tf.transpose(critic_inputs, [1, 0, 2])

        resnet = ResNet(
            inputs = critic_inputs,
            fc_nodes = fc_nodes,
            fc_layers = fc_layers,
            output_size = output_size,
            dropout = dropout,
        )

        self.training = resnet.training

        self.outputs = resnet.outputs
