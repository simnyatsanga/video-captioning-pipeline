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
tf.app.flags.DEFINE_integer('num_examples', 10000,
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


def train_linear_svm_classifier(features, labels):
  # save 20% of data for performance evaluation
  X_train, X_test, y_train, y_test = cross_validation.train_test_split(
                                       features, labels, test_size=0.2
                                     )
  # Define the classifier
  clf = SVC(kernel='linear', C = 10.0)
  clf.fit(X_train, y_train)

  # Store the svm model
  if os.path.exists(FLAGS.model_output_path):
    joblib.dump(clf, FLAGS.model_output_path)
  else:
    print("Cannot save trained svm model to {0}."
          .format(FLAGS.model_output_path))

  y_predict = clf.predict(X_test)
  labels = sorted(list(set(labels)))
  print("\nConfusion matrix:")
  print("Labels: {0}\n".format(",".join(labels)))
  print(confusion_matrix(y_test, y_predict, labels=labels))

  print("\nClassification report:")
  print(classification_report(y_test, y_predict))


def train_svm_classifer(features, labels):
    """train_svm_classifer will train a SVM, saved the trained and SVM model 
    and report the classification performance

    Args:
      features: array of input features
      labels: array of labels associated with the input features
    """
    # save 20% of data for performance evaluation
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
                                         features, labels, test_size=0.2
                                       )

    param = [
        {
            "kernel": ["linear"],
            "C": [1, 10, 100, 1000]
        },
        {
            "kernel": ["rbf"],
            "C": [1, 10, 100, 1000],
            "gamma": [1e-2, 1e-3, 1e-4, 1e-5]
        }
    ]

    # request probability estimation
    svm = SVC(probability=True)

    # 10-fold cross validation, use 4 thread as each fold and each parameter
    # set can be train in parallel
    clf = grid_search.GridSearchCV(svm, param,
            cv=10, n_jobs=4, verbose=3)

    clf.fit(X_train, y_train)

    if os.path.exists(FLAGS.model_output_path):
        joblib.dump(clf.best_estimator_, FLAGS.model_output_path)
    else:
        print("Cannot save trained svm model to {0}."
          .format(FLAGS.model_output_path))

    print("\nBest parameters set:")
    print(clf.best_params_)

    y_predict = clf.predict(X_test)

    labels = sorted(list(set(labels)))
    print("\nConfusion matrix:")
    print("Labels: {0}\n".format(",".join(labels)))
    print(confusion_matrix(y_test, y_predict, labels=labels))

    print("\nClassification report:")
    print(classification_report(y_test, y_predict))


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
        # train_images size: 
        #   [batch_size, num_frames_per_clip, crop_size, crop_size, channels]
        # train_labels size: [batch_size]
        train_images, train_labels, _, _, _ = input_data.read_clip_and_label(
            filename='list/train.list',
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
            images_placeholder: train_images
          })
        # Concatinate all the features and the labels
        if step == 0:
          features = np.squeeze(batch_features)
        else:
          features = np.concatenate((features, np.squeeze(batch_features)), 
                                    axis=0)
        labels = np.concatenate((labels, train_labels), axis=0)

        step += 1
        print('%.2f: data preprocessing' % (step/num_iter))

    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)

  return features, labels


def train():
  """Train the SVM classifier
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
    # Get the features and the labels from the training dataset
    features, labels = get_data(saver, features_op, images_placeholder)
  print('Done processing the Data, Start Training SVM')
  # train the svm
  # train_svm_classifer(features, labels)
  # train the linear svm
  train_linear_svm_classifier(features, labels)

def main(_):
  train()


if __name__ == '__main__':
  tf.app.run()

