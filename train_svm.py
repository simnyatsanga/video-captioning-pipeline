from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import c3d_model
import input_data


# Basic model parameters as external flags.
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 10,
                            """Batch size.""")
tf.app.flags.DEFINE_string('checkpoint_dir', 'result',
                            """Check point directory.""")
tf.app.flags.DEFINE_integer('num_examples', 10000,
                            """Number of examples to run.""")

def svm():
  """Define the SVM model
  """


def train():
  """Train the SVM classifier
  """
  with tf.Graph().as_default() as g:
    variable_averages = tf.train.ExponentialMovingAverage(
        c3d_model.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)
    with tf.Session() as sess:
      # Get the checkpoint file
      ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
      if ckpt and ckpt.model_checkpoint_path:
        # Restores from checkpoint
        saver.restore(sess, ckpt.model_checkpoint_path)
        # Assuming model_checkpoint_path looks something like:
        #   /my-favorite-path/cifar10_train/model.ckpt-0,
        # extract global_step from it.
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
      else:
        print('No checkpoint file found')
        return

      # Start the queue runners.
      coord = tf.train.Coordinator()
      try:
        threads = []
        for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
          threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                          start=True))

        num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
        total_sample_count = num_iter * FLAGS.batch_size
        step = 0
        while step < num_iter and not coord.should_stop():
          # train_images size: 
          #   [batch_size, num_frames_per_clip, crop_size, crop_size, channels]
          # train_labels size: [batch_size]
          train_images, train_labels, _, _, _ = input_data.read_clip_and_label(
              filename='list/train.list',
              batch_size=FLAGS.batch_size,
              num_frames_per_clip=c3d_model.NUM_FRAMES_PER_CLIP,
              crop_size=c3d_model.CROP_SIZE,
              shuffle=True)

          # Extract the video feature according to the pretrained C3D model.
          with tf.variable_scope('c3d_var'):
            # features size: [batch_size, 4096]
            features_tf = c3d_model.inference_c3d(train_images,
                                              batch_size=FLAGS.batch_size,
                                              features=True)

          # Extract the feature data and the label data
          # features size: [batch_size, 4096]
          # labels size: [batch_size]
          features, labels = sess.run([features_tf, train_labels])

          # TODO: train the SVM 

          step += 1

      except Exception as e:  # pylint: disable=broad-except
        coord.request_stop(e)

      coord.request_stop()
      coord.join(threads, stop_grace_period_secs=10)


def main(_):
  train()


if __name__ == '__main__':
  tf.app.run()