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
        base_dir = args.base_dir,
        in_frame_file = "data-spectrogram/train_si84_delta_noisy_global_normalized/feats.scp.mod",
        out_frame_file = "data-spectrogram/train_si84_clean_global_normalized/feats.scp.mod",
        batch_size = args.batch_size,
        buffer_size = args.buffer_size,
        context = args.context,
        out_frame_count = 1,
        shuffle = True,
    )

    dev_loader = DataLoader(
        base_dir = args.base_dir,
        in_frame_file = "data-spectrogram/dev_dt_05_delta_noisy_global_normalized/feats.scp.mod",
        out_frame_file = "data-spectrogram/dev_dt_05_clean_global_normalized/feats.scp.mod",
        batch_size = args.batch_size,
        buffer_size = args.buffer_size,
        context = args.context,
        out_frame_count = 1,
        shuffle = False,
    )

    in_frames = tf.placeholder(tf.float32, shape=(None, 2*args.context + 1, args.frequencies))
    out_frames = tf.placeholder(tf.float32, shape=(None, args.frequencies))
    training = tf.placeholder(tf.bool)

    dropnet = Dropnet(
        inputs      = in_frames,
        output_size = args.frequencies,
        filters     = args.filters,
        fc_layers   = args.fc_layers,
        fc_nodes    = args.fc_nodes,
        activation  = tf.nn.relu,
        dropout     = args.dropout,
        training    = training,
    )

    loss = tf.losses.mean_squared_error(out_frames, dropnet.output)
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
            for in_frame_batch, out_frame_batch in train_loader.batchify(shuffle_batches=False,include_deltas=False):
                fd = {in_frames: in_frame_batch, out_frames: out_frame_batch, training: True}
                batch_loss, _ = sess.run([loss, train], fd)
                train_loss += batch_loss
                count += 1

            print("\nTrain loss:", train_loss / count)

            test_loss = 0
            count = 0
            for in_frame_batch, out_frame_batch in dev_loader.batchify(include_deltas=False):
                fd = {in_frames: in_frame_batch, out_frames: out_frame_batch, training: False}
                test_loss += sess.run(loss, fd)
                count += 1

            test_loss = test_loss / count

            print("Test loss:", test_loss)

            if test_loss < min_loss and args.save_dir:
                saver.save(sess, os.path.join(args.save_dir, "model-{0}.ckpt".format(test_loss)))
