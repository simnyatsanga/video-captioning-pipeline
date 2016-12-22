# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Trains and Evaluates the 3d convolutional neural network using a feed 
    dictionary.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os
import re
import time


from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import math
import numpy as np
import input_data
import c3d_model

# Basic model parameters as external flags.
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', './result',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('gpu_num', 1, 
                            """How many GPUs to use""")
tf.app.flags.DEFINE_integer('max_steps', 100000, 
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('batch_size', 10,
                            """Batch size.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

def placeholder_inputs(batch_size):
  """Generate placeholder variables to represent the input tensors.

  These placeholders are used as inputs by the rest of the model building
  code and will be fed from the downloaded data in the .run() loop, below.

  Args:
    batch_size: The batch size will be baked into both placeholders.

  Returns:
    images_placeholder: Images placeholder.
    labels_placeholder: Labels placeholder.
  """
  # Note that the shapes of the placeholders match the shapes of the full
  # image and label tensors, except the first dimension is now batch_size
  # rather than the full size of the train or test data sets.
  images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                         c3d_model.NUM_FRAMES_PER_CLIP,
                                                         c3d_model.CROP_SIZE,
                                                         c3d_model.CROP_SIZE,
                                                         c3d_model.CHANNELS))
  labels_placeholder = tf.placeholder(tf.int64, shape=(batch_size))
  return images_placeholder, labels_placeholder


def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.

  Note that this function provides a synchronization point across all towers.

  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat_v2(grads, 0)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads


def tower_loss(scope, images, labels):
  """Calculate the total loss on a single tower running the model.

  Args:
    scope: unique prefix string identifying the tower, e.g. 'tower_0'

  Returns:
     Tensor of shape [] containing the total loss for a batch of data
  """  
  # Build the inference Graph
  logits = c3d_model.inference_c3d(images, 0.5, FLAGS.batch_size)

  # Build the portion of the Graph calculating the losses. Note that we will
  # assemble the total_loss using a custom function below.
  _ = c3d_model.loss(logits, labels)

  # Assemble all of the losses for the current tower only.
  losses = tf.get_collection('losses', scope)

  # Calculate the total loss for the current tower
  total_loss = tf.add_n(losses, name='total_loss')

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    loss_name = re.sub('%s_[0-9]*/' % c3d_model.TOWER_NAME, '', l.op.name)
    tf.summary.scalar(loss_name, l)

  return total_loss


def tower_acc(logit, labels):
  correct_pred = tf.equal(tf.argmax(logit, 1), labels)
  accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
  return accuracy

def _variable_on_cpu(name, shape, initializer):
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer)
  return var

def _variable_with_weight_decay(name, shape, wd):
  var = _variable_on_cpu(name, shape, tf.contrib.layers.xavier_initializer())
  if wd is not None:
    weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


def run_training():
  # Create model directory
  if not os.path.exists(FLAGS.train_dir):
      os.makedirs(FLAGS.train_dir)
  use_pretrained_model = False
  model_filename = "./sports1m_finetuning_ucf101.model"

  with tf.Graph().as_default(), tf.device('/cpu:0'):
    #Create a variable to count the number of train() calls. This equals the
    # number of batches processed * FLAGS.num_gpus.
    global_step = tf.get_variable(
        'global_step', [],
        initializer=tf.constant_initializer(0), trainable=False)

    # Get the image and the labels placeholder
    images_placeholder, labels_placeholder = placeholder_inputs(
        FLAGS.batch_size)

    # Calculate the learning rate schedule.
    num_batches_per_epoch = (c3d_model.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN /
                              FLAGS.batch_size)
    decay_steps = int(num_batches_per_epoch * c3d_model.NUM_EPOCHS_PER_DECAY)

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(c3d_model.INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    c3d_model.LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)

    # Create an optimizer that perfrom Adam algorithm
    opt = tf.train.AdamOptimizer(lr)

    # Calculate the gradients for each model tower
    tower_grads = []

    for gpu_index in xrange(FLAGS.gpu_num):
      with tf.device('/gpu:%d' % gpu_index):
        with tf.name_scope('%s_%d' % (c3d_model.TOWER_NAME, gpu_index)) as scope:
          # Calculate the loss for one tower fo the model. This function 
          # constructs the entire model but shares the variables across
          # all towers.
          loss = tower_loss(scope, images_placeholder, labels_placeholder)

          # Reuse variables for the next tower.
          tf.get_variable_scope().reuse_variables()

          # Retain the summaries from the final tower
          summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

          # Calculate the gradients for the batch of data on this tower
          grads = opt.compute_gradients(loss)

          # Keep track of the graidents across all towers
          tower_grads.append(grads)

    # We must calculate the mean of each gradient. Note that this is the
    # synchronization point across all tower
    grads = average_gradients(tower_grads)

    # Add a summary to track the learning rate
    summaries.append(tf.summary.scalar('learning_rate', lr))

    # Add histograms for gradients.
    for grad, var in grads:
      if grad is not None:
        summaries.append(
            tf.summary.histogram(var.op.name + '/gradients',
                                                    grad))

    # Apply the gradients to adjust the shared variables.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
      summaries.append(
          tf.summary.histogram(var.op.name, var))

    # Track the moving averages of all trainable variables
    variable_averages = tf.train.ExponentialMovingAverage(
        c3d_model.MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # Group all the updates into a single train op
    train_op = tf.group(apply_gradient_op, variables_averages_op)

    # Create a saver
    saver = tf.train.Saver(tf.global_variables())

    # Build the summary operation from the last tower summaries
    summary_op = tf.summary.merge(summaries)

    # Build an initialization operation to run below
    init = tf.global_variables_initializer()

    # Start running operations on the Graph. allow_soft_placement must be set to
    # True to build towers on GPU, as some of the ops do not have GPU
    # implementations.
    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=FLAGS.log_device_placement))
    sess.run(init)

    train_writer = tf.summary.FileWriter(
        os.path.join(FLAGS.train_dir, 'visual_logs', 'train'),
        sess.graph)
    test_writer = tf.summary.FileWriter(
        os.path.join(FLAGS.train_dir, 'visual_logs', 'test'),
        sess.graph)


    for step in xrange(FLAGS.max_steps):
      start_time = time.time()
      # Get the input data
      train_images, train_labels, _, _, _ = input_data.read_clip_and_label(
          filename='list/train.list',
          batch_size=FLAGS.batch_size,
          num_frames_per_clip=c3d_model.NUM_FRAMES_PER_CLIP,
          crop_size=c3d_model.CROP_SIZE,
          shuffle=True)

      # Train the network
      sess.run(train_op, feed_dict={
                            images_placeholder: train_images,
                            labels_placeholder: train_labels})
      duration = time.time() - start_time
      print('Step %d: %.3f sec' % (step, duration))

      # Evaluate the model periodically
      if step % 10 == 0:
        # Training Evaluation
        print('Training Data Eval:')
        summary, loss_value = sess.run(
            [summary_op, loss], feed_dict={
                                    images_placeholder: train_images,
                                    labels_placeholder: train_labels})
        assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
        train_writer.add_summary(summary, step)
        num_examples_per_step = FLAGS.batch_size * FLAGS.gpu_num
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = duration / FLAGS.gpu_num

        format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch)')
        print (format_str % (datetime.now(), step, loss_value,
                             examples_per_sec, sec_per_batch))

        # Test Evaluation
        print('Training Data Eval:')
        val_images, val_labels, _, _, _ = input_data.read_clip_and_label(
            filename='list/test.list',
            batch_size=FLAGS.batch_size,
            num_frames_per_clip=c3d_model.NUM_FRAMES_PER_CLIP,
            crop_size=c3d_model.CROP_SIZE,
            shuffle=True)
        summary, loss_value = sess.run(
            [summary_op, loss], feed_dict={
                                    images_placeholder: val_images,
                                    labels_placeholder: val_labels})
        assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
        test_writer.add_summary(summary, step)
        num_examples_per_step = FLAGS.batch_size * FLAGS.gpu_num
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = duration / FLAGS.gpu_num

        format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch)')
        print (format_str % (datetime.now(), step, loss_value,
                             examples_per_sec, sec_per_batch))

      # Save the model checkpoint periodically.
      if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
  print('Done')


def main(_):
  run_training()


if __name__ == '__main__':
  tf.app.run()
