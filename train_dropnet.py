import tensorflow as tf
import argparse
import os

from data_io import DataLoader
from dropnet import Dropnet

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Path params
    parser.add_argument("--base_dir", default=os.getcwd())
    parser.add_argument("--load_dir", default=None)
    parser.add_argument("--save_dir", default=None)
    parser.add_argument("--noisy_train_file")
    parser.add_argument("--clean_train_file")
    parser.add_argument("--noisy_dev_file")
    parser.add_argument("--clean_dev_file")

    # Model params
    parser.add_argument("--filters", default=[128, 128, 256, 256], nargs="+", type=int)
    parser.add_argument("--fc_nodes", default=2048, type=int)
    parser.add_argument("--fc_layers", default=2, type=int)
    parser.add_argument("--dropout", default=0.3, type=float)
    parser.add_argument("--context", default=5, type=int)
    parser.add_argument("--frequencies", default=257, type=int)

    # Training params
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--buffer_size", default=40, type=int)
    parser.add_argument("--learn_rate", default=1e-4, type=float)
    parser.add_argument("--lr_decay", default=1e-6, type=float)
    args = parser.parse_args()

    train_loader = DataLoader(
        base_dir    = args.base_dir,
        frame_file  = args.noisy_train_file,
        clean_file  = args.clean_train_file,
        batch_size  = args.batch_size,
        buffer_size = args.buffer_size,
        context     = args.context,
        out_frames  = 1,
        shuffle     = True,
    )

    dev_loader = DataLoader(
        base_dir    = args.base_dir,
        frame_file  = args.noisy_dev_file,
        clean_file  = args.clean_dev_file,
        batch_size  = args.batch_size,
        buffer_size = args.buffer_size,
        context     = args.context,
        out_frames  = 1,
        shuffle     = False,
    )

    noisy_frames = tf.placeholder(tf.float32, shape=(None, 2*args.context + 1, args.frequencies))
    clean_frames = tf.placeholder(tf.float32, shape=(None, args.frequencies))
    training = tf.placeholder(tf.bool)

    with tf.variable_scope('actor'):
        dropnet = Dropnet(
            inputs      = noisy_frames,
            output_size = args.frequencies,
            filters     = args.filters,
            fc_layers   = args.fc_layers,
            fc_nodes    = args.fc_nodes,
            activation  = tf.nn.relu,
            dropout     = args.dropout,
            training    = training,
        )

    loss = tf.losses.mean_squared_error(clean_frames, dropnet.output)
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.inverse_time_decay(args.learn_rate, global_step, 1, args.lr_decay)
    train = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        if args.load_dir:
            saver.restore(sess, tf.train.latest_checkpoint(args.load_dir))

        min_loss = 1
        for epoch in range(0, 100):
            print("Epoch", epoch)

            train_loss = 0
            count = 0
            for batch in train_loader.batchify():
                noisy, clean = batch['frame'], batch['clean']
                fd = {noisy_frames: noisy, clean_frames: clean, training: True}
                batch_loss, _ = sess.run([loss, train], fd)
                train_loss += batch_loss
                count += 1

            print("Train loss:", train_loss / count)

            test_loss = 0
            count = 0
            for batch in dev_loader.batchify():
                noisy, clean = batch['frame'], batch['clean']
                fd = {noisy_frames: noisy, clean_frames: clean, training: False}
                test_loss += sess.run(loss, fd)
                count += 1

            test_loss = test_loss / count

            print("Test loss:", test_loss)

            if test_loss < min_loss and args.save_dir:
                saver.save(sess, os.path.join(args.save_dir, "model-%.4f.ckpt" % test_loss))
