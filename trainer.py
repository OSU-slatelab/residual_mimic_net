"""
Trainer creates train ops and goes through all data to train or test.

Author: Peter Plantinga
Date: Fall 2017
"""

import tensorflow as tf
import time
import sys

def update_progressbar(progress):
    """ Make a very basic progress bar """

    length = 30
    intprog = int(round(progress * length))
    sys.stdout.write("\r[{0}] {1:2.1f}%".format("#"*intprog + "-"*(length-intprog), progress*100))
    sys.stdout.flush()

class Trainer:
    """ Train a model """

    def __init__(self,
        critic,
        learn_rate = 1e-4,
        lr_decay   = 0,
        max_norm   = 5.0,
        alpha      = 1,
        actor      = None,
        teacher    = None,
    ):
        """ 
        Parameters
        ----------
        critic : Critic
            Model to train on clean speech. If actor is passed, train that instead
        learn_rate : float
            Rate of gradient descent
        lr_decay : float
            Amount of decay for learning rate
        max_norm : float
            For clipping norm
        actor : Actor
            (optional) model to train. If passed, critic is frozen.
        teacher : Critic
            (optional) model generating posteriors as labels
        """
        
        self.feed_dict = {}

        # Set this to a placeholder if clean speech is input
        self.clean = None

        # Whether we're training critic or not
        self.critic_train = actor is None

        # Actor is none if we're training critic
        if actor is None:
            self.inputs = critic.inputs
            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')
            self.training = critic.training
            self.labels = critic.labels

            loss = tf.nn.softmax_cross_entropy_with_logits(logits=critic.outputs, labels=critic.labels)
            self.loss = tf.reduce_mean(loss)

        # Training actor
        else:
            self.inputs = actor.inputs
            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='dropnet')
            self.training = actor.training
            
            self.feed_dict[critic.training] = False
            self.feed_dict[teacher.training] = False

            self.clean = teacher.inputs
            self.labels = critic.labels
            
            labels = teacher.outputs
            predictions = critic.outputs

            loss = tf.losses.mean_squared_error(labels=labels, predictions=predictions)
            self.mimic_loss = tf.reduce_mean(loss)

            loss = tf.losses.mean_squared_error(labels=self.clean, predictions=actor.outputs)
            self.fidelity_loss = tf.reduce_mean(loss)

            self.loss = alpha * self.mimic_loss + self.fidelity_loss

        self.learn_rate = learn_rate
        self.lr_decay = lr_decay
        self.max_norm = max_norm

        self._create_train_op()

    def _create_train_op(self):
        """ Define the training op """
        
        grads = tf.gradients(self.loss, self.var_list)
        grads, _ = tf.clip_by_global_norm(grads, clip_norm=self.max_norm)
        grad_var_pairs = zip(grads, self.var_list)
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.inverse_time_decay(self.learn_rate, global_step, 1, self.lr_decay)
        optim = tf.train.AdamOptimizer(learning_rate)
        self.train = optim.apply_gradients(grad_var_pairs, global_step=global_step)

    def run_ops(self, sess, loader, training = True):

        tot_loss = 0
        tot_fidelity_loss = 0
        tot_mimic_loss = 0
        frames = 0
        start_time = time.time()
        self.feed_dict[self.training] = training

        # Iterate dataset
        for batch in loader.batchify():

            self.feed_dict[self.inputs] = batch['frame']
            self.feed_dict[self.labels] = batch['label']

            # Count the frames in the batch
            batch_frames = batch['frame'].shape[0]

            if self.clean is not None:
                self.feed_dict[self.clean] = batch['clean']

            # If we're combining mse and critic loss, report both independently
            if not self.critic_train:
                self.feed_dict[self.mse_weight] = self.current_mse_weight

                ops = [self.fidelity_loss, self.mimic_loss, self.loss]

                if training:
                    fidelity_loss, mimic_loss, batch_loss, _ = sess.run(ops + [self.train], self.feed_dict)
                else:
                    fidelity_loss, mimic_loss, batch_loss = sess.run(ops, self.feed_dict)

                tot_fidelity_loss += batch_frames * fidelity_loss
                tot_mimic_loss += batch_frames * mimic_loss
            
            # Just critic loss
            elif training:
                batch_loss, _ = sess.run([self.loss, self.train], feed_dict = self.feed_dict)
            else:
                batch_loss = sess.run(self.loss, feed_dict = self.feed_dict)

            tot_loss += batch_frames * batch_loss

            # Update the progressbar
            frames += batch_frames
            update_progressbar(frames / loader.frame_count)

        # Compute loss
        losses = {'avg_loss': float(tot_loss) / frames}
        duration = time.time() - start_time

        loader.reset()

        if not self.critic_train:
            losses['fidelity_loss'] = tot_fidelity_loss / frames
            losses['mimic_loss'] = tot_mimic_loss / frames
            
        return losses, duration

