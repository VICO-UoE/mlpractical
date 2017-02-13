#!/disk/scratch/mlp/miniconda2/bin/python

import os
import datetime
import numpy as np
import tensorflow as tf
import mlp.data_providers as data_providers

# check necessary environment variables are defined
assert 'MLP_DATA_DIR' in os.environ, (
    'An environment variable MLP_DATA_DIR must be set to the path containing'
    ' MLP data before running script.')
assert 'OUTPUT_DIR' in os.environ, (
    'An environment variable OUTPUT_DIR must be set to the path to write'
    ' output to before running script.')

# load data
train_data = data_providers.MNISTDataProvider('train', batch_size=50)
valid_data = data_providers.MNISTDataProvider('valid', batch_size=50)
valid_inputs = valid_data.inputs
valid_targets = valid_data.to_one_of_k(valid_data.targets)

# define model graph
with tf.name_scope('data'):
    inputs = tf.placeholder(tf.float32, [None, 784], name='inputs')
    targets = tf.placeholder(tf.float32, [None, 10], name='targets')
with tf.name_scope('parameters'):
    weights = tf.Variable(tf.zeros([784, 10]), name='weights')
    biases = tf.Variable(tf.zeros([10]), name='biases')
with tf.name_scope('model'):
    outputs = tf.matmul(inputs, weights) + biases
with tf.name_scope('error'):
    error = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(outputs, targets))
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(error)
with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(tf.equal(
        tf.argmax(outputs, 1), tf.argmax(targets, 1)), tf.float32))

# add summary operations
tf.summary.scalar('error', error)
tf.summary.scalar('accuracy', accuracy)
summary_op = tf.summary.merge_all()

# create objects for writing summaries and checkpoints during training
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
exp_dir = os.path.join(os.environ['OUTPUT_DIR'], timestamp)
checkpoint_dir = os.path.join(exp_dir, 'checkpoints')
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
train_writer = tf.summary.FileWriter(os.path.join(exp_dir, 'train-summaries'))
valid_writer = tf.summary.FileWriter(os.path.join(exp_dir, 'valid-summaries'))
saver = tf.train.Saver()

# create arrays to store run train / valid set stats
num_epoch = 5
train_accuracy = np.zeros(num_epoch)
train_error = np.zeros(num_epoch)
valid_accuracy = np.zeros(num_epoch)
valid_error = np.zeros(num_epoch)

# create session and run training loop
sess = tf.Session()
sess.run(tf.global_variables_initializer())
step = 0
for e in range(num_epoch):
    for b, (input_batch, target_batch) in enumerate(train_data):
        # do train step with current batch
        _, summary, batch_error, batch_acc = sess.run(
            [train_step, summary_op, error, accuracy],
            feed_dict={inputs: input_batch, targets: target_batch})
        # add symmary and accumulate stats
        train_writer.add_summary(summary, step)
        train_error[e] += batch_error
        train_accuracy[e] += batch_acc
        step += 1
    # normalise running means by number of batches
    train_error[e] /= train_data.num_batches
    train_accuracy[e] /= train_data.num_batches
    # evaluate validation set performance
    valid_summary, valid_error[e], valid_accuracy[e] = sess.run(
        [summary_op, error, accuracy],
        feed_dict={inputs: valid_inputs, targets: valid_targets})
    valid_writer.add_summary(valid_summary, step)
    # checkpoint model variables
    saver.save(sess, os.path.join(checkpoint_dir, 'model.ckpt'), step)
    # write stats summary to stdout
    print('Epoch {0:02d}: err(train)={1:.2f} acc(train)={2:.2f}'
          .format(e + 1, train_error[e], train_accuracy[e]))
    print('          err(valid)={0:.2f} acc(valid)={1:.2f}'
          .format(valid_error[e], valid_accuracy[e]))

# close writer and session objects
train_writer.close()
valid_writer.close()
sess.close()

# save run stats to a .npz file
np.savez_compressed(
    os.path.join(exp_dir, 'run.npz'),
    train_error=train_error,
    train_accuracy=train_accuracy,
    valid_error=valid_error,
    valid_accuracy=valid_accuracy
)
