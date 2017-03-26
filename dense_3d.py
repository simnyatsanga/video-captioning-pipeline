import numpy as np
import tensorflow as tf

import input_data


BATCH_SIZE = 10

def run_in_batch_avg(session, tensors, batch_placeholders, feed_dict={}, batch_size=100):
  res = [ 0 ] * len(tensors)
  batch_tensors = [ (placeholder, feed_dict[ placeholder ]) for placeholder in batch_placeholders ]
  total_size = len(batch_tensors[0][1])
  batch_count = (total_size + batch_size - 1) / batch_size
  for batch_idx in xrange(batch_count):
    current_batch_size = None
    for (placeholder, tensor) in batch_tensors:
      batch_tensor = tensor[ batch_idx*batch_size : (batch_idx+1)*batch_size ]
      current_batch_size = len(batch_tensor)
      feed_dict[placeholder] = tensor[ batch_idx*batch_size : (batch_idx+1)*batch_size ]
    tmp = session.run(tensors, feed_dict=feed_dict)
    res = [ r + t * current_batch_size for (r, t) in zip(res, tmp) ]
  return [ r / float(total_size) for r in res ]

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.01)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.01, shape=shape)
  return tf.Variable(initial)

def conv3d(input, in_features, out_features, kernel_size, with_bias=False):
  W = weight_variable([ kernel_size, kernel_size, kernel_size, in_features, out_features ])
  conv = tf.nn.conv3d(input, W, [ 1, 1, 1, 1, 1 ], padding='SAME')
  if with_bias:
    return conv + bias_variable([ out_features ])
  return conv

def batch_activ_conv(current, in_features, out_features, kernel_size, is_training, keep_prob):
  current = tf.contrib.layers.batch_norm(current, scale=True, is_training=is_training, updates_collections=None)
  current = tf.nn.relu(current)
  current = conv3d(current, in_features, out_features, kernel_size)
  current = tf.nn.dropout(current, keep_prob)
  return current

def block(input, layers, in_features, growth, is_training, keep_prob):
  current = input
  features = in_features
  for idx in xrange(layers):
    # Size tmp: [batch_size, sequence_length, height, weight, out_channel]
    tmp = batch_activ_conv(current, features, growth, 3, is_training, keep_prob)
    # Concatinate all the feature along the out_channel axis
    current = tf.concat((current, tmp), 4)
    # Accumulate the output channel number
    features += growth
  return current, features

def avg_pool(input, s):
  return tf.nn.avg_pool3d(input, [ 1, 2, s, s, 1 ], [1, 2, s, s, 1 ], 'VALID')

def run_model(depth=40):
  weight_decay = 1e-4
  layers = (depth - 4) / 3
  graph = tf.Graph()
  with graph.as_default():
    xs = tf.placeholder("float", shape=[BATCH_SIZE, 8, 32, 32, 3])
    ys = tf.placeholder(tf.int64, shape=[BATCH_SIZE])
    ys_onehot = tf.one_hot(ys, 5)
    lr = tf.placeholder("float", shape=[])
    keep_prob = tf.placeholder(tf.float32)
    is_training = tf.placeholder("bool", shape=[])

    current = xs
    # Firt convolution layer
    current = conv3d(current, 3, 16, 3)
    
    # First block
    current, features = block(current, layers, 16, 12, is_training, keep_prob)
    # Second convolution layer
    current = batch_activ_conv(current, features, features, 1, is_training, keep_prob)
    # First pooling layer
    current = avg_pool(current, 2)
    
    # Second block
    current, features = block(current, layers, features, 12, is_training, keep_prob)
    # Third convolution layer
    current = batch_activ_conv(current, features, features, 1, is_training, keep_prob)
    # Second pooling layer
    current = avg_pool(current, 2)

    # Third block
    #current, features = block(current, layers, features, 12, is_training, keep_prob)
    # Fourth convolution layer
    #current = batch_activ_conv(current, features, features, 1, is_training, keep_prob)
    # Third pooling layer
    #current = avg_pool(current, 4)

    # Fourth block
    current, features = block(current, layers, features, 12, is_training, keep_prob)
    current = tf.contrib.layers.batch_norm(current, scale=True, is_training=is_training, updates_collections=None)
    current = tf.nn.relu(current)
    # Fourth pooling layer
    current = avg_pool(current, 8)
    final_dim = features
    current = tf.reshape(current, [ -1, final_dim ])
    
    # Fully connected layer
    Wfc = weight_variable([ final_dim, 5 ])
    bfc = bias_variable([ 5 ])
    ys_ = tf.nn.softmax( tf.matmul(current, Wfc) + bfc )

    cross_entropy = -tf.reduce_mean(ys_onehot * tf.log(ys_ + 1e-12))
    l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
    train_step = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True).minimize(cross_entropy + l2 * weight_decay)
    correct_prediction = tf.equal(tf.argmax(ys_, 1), tf.argmax(ys_onehot, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

  with tf.Session(graph=graph) as session:
    batch_size = BATCH_SIZE
    learning_rate = 0.1
    session.run(tf.initialize_all_variables())
    saver = tf.train.Saver()
    for epoch in xrange(1, 1+300):
      if epoch == 150: learning_rate = 0.01
      if epoch == 225: learning_rate = 0.001
      for batch_idx in xrange(500):
        # Get the training data and training label
        train_images, train_labels, _, _, _ = input_data.read_clip_and_label(
          filename='list/train.list',
          batch_size=batch_size,
          num_frames_per_clip=8,
          crop_size=32,
          shuffle=True)
        # Input normalization
        train_images = train_images/256

        batch_res = session.run([ train_step, cross_entropy, accuracy ],
          feed_dict = { xs: train_images, ys: train_labels, lr: learning_rate, is_training: True, keep_prob: 0.8 })
        if batch_idx % 100 == 0: print epoch, batch_idx, batch_res[1:]

      save_path = saver.save(session, 'densenet_%d.ckpt' % epoch)

      # Get the test data and test label
      test_images, test_labels, _, _, _ = input_data.read_clip_and_label(
        filename='list/test.list',
        batch_size=batch_size,
        num_frames_per_clip=8,
        crop_size=32,
        shuffle=True)
      test_labels = tf.one_hot(test_labels, 5)
      # Input normalization
      test_images = test_images/256
      test_results = session.run([ accuracy ],
        feed_dict = { xs: test_images, ys: test_labels, lr: learning_rate, is_training: False, keep_prob: 1 })
      print epoch, test_results

def run():
  run_model()

run()