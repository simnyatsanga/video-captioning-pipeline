from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import sklearn
from sklearn import cross_validation, grid_search
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.externals import joblib
import math
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
tf.app.flags.DEFINE_string('model_output_path', 'result/svm',
                            """SVM model directory.""")
tf.app.flags.DEFINE_integer('num_examples', 5000,
                            """Number of examples to run.""")


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
  return images_placeholder


def eval_svm(features, labels):
    """eval_svm will eval the pretrained SVM and report the classification 
      performance

    Args:
      features: array of input features
      labels: array of labels associated with the input features
    """
    if os.path.exists(FLAGS.model_output_path):
        clf = joblib.load(FLAGS.model_output_path)
        predict = clf.predict(features)
        labels = sorted(list(set(labels)))
        print("\nConfusion matrix:")
        print("Labels: {0}\n".format(",".join(labels)))
        print(confusion_matrix(labels, predict, labels=labels))
        print("\nClassification report:")
        print(classification_report(y_test, y_predict))
    else:
        print("Cannot load trained svm model from {0}."
          .format(FLAGS.model_output_path))


def get_data(saver, features_op, images_placeholder):
  """Grap the video features according to the C3D fc6 layer
  
  Args
    saver: tensorflow saver
    features_op: features operation
  
  Returns:
    features: [num_examples, 4096], num_examples of video features
    labels: [num_examples]
  """
  features = []
  labels = []
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
        # test_images size: 
        #   [batch_size, num_frames_per_clip, crop_size, crop_size, channels]
        # test_labels size: [batch_size]
        test_images, test_labels, _, _, _ = input_data.read_clip_and_label(
            filename='list/test.list',
            batch_size=FLAGS.batch_size,
            num_frames_per_clip=c3d_model.NUM_FRAMES_PER_CLIP,
            crop_size=c3d_model.CROP_SIZE,
            shuffle=True)
        # Extract the feature data and the label data
        #   features size: [batch_size, 4096]
        #   labels size: [batch_size]
        batch_features = sess.run(
          features_op,
          feed_dict={
            images_placeholder: test_images
          })
        # Concatinate all the features and the labels
        if step == 0:
          features = np.squeeze(batch_features)
        else:
          features = np.concatenate((features, np.squeeze(batch_features)), 
                                    axis=0)
        labels = np.concatenate((labels, test_labels), axis=0)

        step += 1
        print('%.2f: data preprocessing' % (step/num_iter))

    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)

  return features, labels


def eval():
  """Evaluate the SVM classifier
  """
  with tf.Graph().as_default() as g:
    # Get the image and the labels placeholder
    images_placeholder = placeholder_inputs(FLAGS.batch_size)
    # Build the Graph that computes the lc6 feature
    with tf.variable_scope('c3d_var'):
      # Extract the video feature according to the pretrained C3D model.
      features_op = c3d_model.inference_c3d(images_placeholder,
                                            batch_size=FLAGS.batch_size,
                                            features=True)
      # Apply the L2 normalization function to all features
      features_op = tf.nn.l2_normalize(features_op, 1)
    # Restore the moving average version of the learned variables for evaluation.
    variable_averages = tf.train.ExponentialMovingAverage(
        c3d_model.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    print('Start processing the Data')
    # Get the features and the labels from the testing dataset
    features, labels = get_data(saver, features_op, images_placeholder)
  print('Done processing the Data, Start evaluating SVM')
  # Evaluate the svm
  eval_svm(features, labels)


def main(_):
  eval()


if __name__ == '__main__':
  tf.app.run()

