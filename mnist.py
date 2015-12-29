import tensorflow as tf
import input_data

# Basic setup
sess = tf.InteractiveSession()

# Read datasets
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Create model
x = tf.placeholder(tf.float32, [None, 784], name="x-input")
W = tf.Variable(tf.zeros([784, 10]), name="weights")
b = tf.Variable(tf.zeros([10]), name="biases")

with tf.name_scope("Wx_b") as scope:
  y = tf.nn.softmax(tf.matmul(x, W) + b)

# Add summary ops to collect data
w_hist = tf.histogram_summary("weights", W)
b_hist = tf.histogram_summary("biases", b)
y_hist = tf.histogram_summary("y", y)

# Reshape the weight tensor to [batch_size, width, height, channel]
w_reshaped = tf.expand_dims(tf.transpose(tf.reshape(W, [28, 28, 10])), 3)
tf.image_summary("weights", w_reshaped, max_images=10)

# Define loss and optimizer
y_ = tf.placeholder("float", [None, 10], name="y-input")

# compute entropy
with tf.name_scope("xent") as scope:
  cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
with tf.name_scope("train") as scope:
  train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
with tf.name_scope("test") as scope:
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
  accuracy_summary = tf.scalar_summary("accuracy", accuracy)

# Output summary
merged = tf.merge_all_summaries()
writer = tf.train.SummaryWriter("/tmp/mnist_logs", sess.graph_def)
tf.initialize_all_variables().run()

# Train and record summaries every 20 steps
for i in range(1000):
  if i % 20 == 0:
    feed = {x: mnist.test.images, y_: mnist.test.labels}
    summary_str, acc = sess.run([merged, accuracy], feed_dict=feed)
    writer.add_summary(summary_str, i)
    print("Accuracy at step %s: %s" % (i, acc))
  else:
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))


