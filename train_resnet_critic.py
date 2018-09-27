import tensorflow as tf
import argparse
import os

from resnet import ResNet
from data_io_old import DataLoader
from trainer import Trainer

def run_training(a):
    """ Define our model and train it """

    # Create directory for saving models
    if not os.path.isdir(a.save_dir):
        os.makedirs(a.save_dir)

    with tf.Graph().as_default():
        shape = (None, 2*a.context + 1, a.frequencies)
        frame_placeholder = tf.placeholder(tf.float32, shape=shape, name="frame_placeholder")

        # Define our critic model
        with tf.variable_scope('critic'):
            critic = ResNet(
                inputs      = frame_placeholder,
                output_size = a.senones,
                filters     = a.filters,
                fc_layers   = a.fc_layers,
                fc_nodes    = a.fc_nodes,
                dropout     = a.dropout,
            )

        critic.labels = tf.placeholder(tf.float32, shape=(None, a.senones), name = "labels")

        # Create loader for train data
        train_loader = DataLoader(
            base_dir    = a.base_dir,
            frame_file  = a.clean_train,
            senone_file = a.senone_train,
            batch_size  = a.batch_size,
            buffer_size = a.buffer_size,
            context     = a.context,
            out_frames  = 1,
            shuffle     = True,
        )

        # Create loader for test data
        dev_loader = DataLoader(
            base_dir    = a.base_dir,
            frame_file  = a.clean_dev,
            senone_file = a.senone_dev,
            batch_size  = a.batch_size,
            buffer_size = a.buffer_size,
            context     = a.context,
            out_frames  = 1,
            shuffle     = False,
        )

        # Class for training
        with tf.variable_scope('trainer'):
            trainer = Trainer(critic, learn_rate=a.learn_rate, lr_decay=a.lr_decay, max_norm=a.max_global_norm)

        # Save all variables
        saver = tf.train.Saver()

        # Begin session
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess = tf.Session()
        sess.run(init)

        # Perform training
        min_loss = float('inf')
        for epoch in range(1, 200):
            print('Epoch %d' % epoch)

            #print(sess.run(trainer.learning_rate))
            loss, duration = trainer.run_ops(sess, train_loader, training = True)
            #print(sess.run(trainer.learning_rate))
            train_loss = loss['avg_loss']
            print ('\nTrain loss: %.6f (%.3f sec)' % (train_loss, duration))

            loss, duration = trainer.run_ops(sess, dev_loader, training = False)
            eval_loss = loss['avg_loss']
            print('\nEval loss: %.6f (%.3f sec)' % (eval_loss, duration))

            if eval_loss < min_loss:
                min_loss = eval_loss
                save_file = os.path.join(a.save_dir, "model-%.4f.ckpt" % eval_loss)
                save_path = saver.save(sess, save_file, global_step=epoch)

def main():
    parser = argparse.ArgumentParser()

    # Files
    parser.add_argument("--base_dir", default=os.getcwd(), help="The directory the data is in")
    parser.add_argument("--clean_train", help="The input clean feature file")
    parser.add_argument("--clean_dev", help="The input dev feature file")
    parser.add_argument("--senone_train", help="The senone file for clean labels")
    parser.add_argument("--senone_dev", help="The senone file for dev labels")
    parser.add_argument("--save_dir", help="directory to save the weights of our model")

    # Training
    parser.add_argument("--learn_rate", type=float, default=1e-4, help="initial learning rate")
    parser.add_argument("--lr_decay", type=float, default=0.95, help="learning rate decay")
    parser.add_argument("--max_global_norm", type=float, default=5.0, help="global max norm for clipping")
    parser.add_argument("--dropout", type=float, default=0.3, help="fraction of filters and neurons to drop")
    parser.add_argument("--buffer_size", default=10, type=int)
    parser.add_argument("--batch_size", default=1024, type=int)
    
    # Model
    parser.add_argument("--filters", type=int, default=[128,128,256,256], nargs="+")
    parser.add_argument("--fc_layers", type=int, default=2)
    parser.add_argument("--fc_nodes", type=int, default=2048)

    # Input
    parser.add_argument("--frequencies", type=int, default=257)
    parser.add_argument("--senones", type=int, default=1999)
    parser.add_argument("--context", type=int, default=5)
    a = parser.parse_args()

    run_training(a)
    
if __name__=='__main__':
    main()    
    

