import tensorflow as tf
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

x = tf.Variable(3, name="x")
y = tf.Variable(4, name="x")
f = x * x * y + y + 2

init = tf.global_variables_initializer()

with tf.Session() as sess:
  sess.run(init)
  result = f.eval()

tf.reset_default_graph()

x1 = tf.Variable(0)

x1.graph is tf.get_default_graph()

graph = tf.Graph()
with graph.as_default():
  x2 = tf.Variable(2)

x2.graph is graph
x2.graph is tf.get_default_graph()

tf.reset_default_graph()

w = tf.constant(3)
x = w + 2
y = x + 5
z = x + 3

with tf.Session() as sess:
  y.eval()
  z.eval()

reset_graph()

with tf.Session() as sess:
  y_out, z_out = sess.run([y, z])
  y_out
  z_out

# Linear regerssion
import numpy as np
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()

m, n = housing.data.shape

housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]
# test = np.zeros((3, 2))
# np.c_[np.ones((test.shape[0], 1)), test]

X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name='X')
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name='y')

XT = tf.transpose(X)
# (X^T * X)^-1 * X^T * y
theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)

with tf.Session() as sess:
  theta_value = theta.eval()

reset_graph()
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing

n_epochs = 1000
learning_rate = 0.01

housing = fetch_california_housing()
scalar = StandardScaler()
scaled_housing_data = scalar.fit_transform(housing.data)

m, n = scaled_housing_data.shape
scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
theta = tf.Variable(tf.random_uniform((n + 1, 1)), name='theta')
y_pred = tf.matmul(X, theta, name='prediction')
error = y_pred - y

mse = tf.reduce_mean(tf.square(error), name='mse')
# gradients = 2/m * tf.matmul(tf.transpose(X), error)
gradients = tf.gradients(mse, [theta])[0]
training_op = tf.assign(theta, theta - learning_rate * gradients)

init = tf.global_variables_initializer()

with tf.Session() as sess:
  sess.run(init)
  for epoch in range(n_epochs):
    if epoch % 100 == 0:
      print("Epoch", epoch, "mse: ", mse.eval())
    _ = sess.run(training_op)
  best_theta = theta.eval()

reset_graph()

from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
scaler = StandardScaler()
scaled_housing_data = scaler.fit_transform(housing.data)
m, n = scaled_housing_data.shape
scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]
X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
theta = tf.Variable(tf.random_uniform((n + 1, 1), dtype=tf.float32, name="theta"))
y_pred = tf.matmul(X, theta)
error = y_pred - y 
mse = tf.reduce_mean(tf.square(error), name="mse")
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()


with tf.Session() as sess:
  sess.run(init)
  for epcoh in range(n_epochs):
    if epcoh % 100 == 0:
      print("Epoch ", epoch, "mse", mse.eval())
    _ = sess.run(training_op)
  theta_val = theta.eval()

A = tf.placeholder(tf.float32, shape=(None, 3))
B = A + 5
with tf.Session() as sess:
  sess.run(B, feed_dict={A: [[1, 2, 3]]})
  sess.run(B, feed_dict={A: [[1, 2, 3], [4, 5, 6]]})


reset_graph()

from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
scaler = StandardScaler()
scaled_housing_data = scaler.fit_transform(housing.data)
m, n = scaled_housing_data.shape
scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]
X = tf.placeholder(tf.float32, shape=(None, n + 1), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")
theta = tf.Variable(tf.random_uniform((n + 1, 1), dtype=tf.float32, name="theta"))
y_pred = tf.matmul(X, theta)
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
training_op = optimizer.minimize(mse)
init = tf.global_variables_initializer()
n_epochs = 10
batch_size = 100
n_batchs = int(n_epochs / batch_size)
learning_rate = 0.01


def fetch_batch(epoch, batch_index, batch_size):
  np.random.seed(epoch * n_batchs + batch_index)
  indices = np.random.randint(m, size=batch_size)
  X_batch = scaled_housing_data_plus_bias[indices]
  y_batch = housing.target.reshape(-1, 1)[indices]
  return X_batch, y_batch


with tf.Session() as sess:
  sess.run(init)
  for epoch in range(n_epochs):
    for batch_index in range(n_batchs):
      X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
      _ = sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
  theta.eval()


reset_graph()

from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
scaler = StandardScaler()
scaled_housing_data = scaler.fit_transform(housing.data)
m, n = scaled_housing_data.shape
scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]
X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
theta = tf.Variable(tf.random_uniform((n + 1, 1), dtype=tf.float32, name="theta"))
y_pred = tf.matmul(X, theta)
error = y_pred - y 
mse = tf.reduce_mean(tf.square(error), name="mse")
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
  sess.run(init)
  for epcoh in range(n_epochs):
    if epcoh % 100 == 0:
      print("Epoch ", epoch, "mse", mse.eval())
      save_path = saver.save(sess, "/tmp/my_model.ckpt")
    _ = sess.run(training_op)
  save_path = saver.save(sess, "/tmp/my_model_final.ckpt")
  theta_val = theta.eval()

with tf.Session() as sess:
  saver.restore(sess, "/tmp/my_model_final.ckpt")
  best_theta_val = theta.eval()

np.allclose(best_theta_val, theta_val)



reset_graph()
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
from datetime import datetime
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "./tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)

housing = fetch_california_housing()
scaler = StandardScaler()
scaled_housing_data = scaler.fit_transform(housing.data)
m, n = scaled_housing_data.shape
scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]
X = tf.placeholder(tf.float32, shape=(None, n + 1), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")
theta = tf.Variable(tf.random_uniform((n + 1, 1), dtype=tf.float32, name="theta"))
y_pred = tf.matmul(X, theta)
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
training_op = optimizer.minimize(mse)
init = tf.global_variables_initializer()
n_epochs = 10
batch_size = 10
n_batchs = int(np.ceil(n_epochs / batch_size))
learning_rate = 0.01

mse_summary = tf.summary.scalar("MSE", mse)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())


def fetch_batch(epoch, batch_index, batch_size):
  np.random.seed(epoch * n_batchs + batch_index)
  indices = np.random.randint(m, size=batch_size)
  X_batch = scaled_housing_data_plus_bias[indices]
  y_batch = housing.target.reshape(-1, 1)[indices]
  return X_batch, y_batch


with tf.Session() as sess:
  sess.run(init)
  for epoch in range(n_epochs):
    for batch_index in range(n_batchs):
      X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
      if batch_index % 10 == 0:
        summary_str = mse_summary.eval(feed_dict={X: X_batch, y: y_batch})
        step = epoch * n_batchs + batch_index
        file_writer.add_summary(summary_str, step)
      _ = sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

file_writer.close()

reset_graph()

def fetch_batch(epoch, batch_index, batch_size):
    np.random.seed(epoch * n_batches + batch_index)  # not shown in the book
    indices = np.random.randint(m, size=batch_size)  # not shown
    X_batch = scaled_housing_data_plus_bias[indices] # not shown
    y_batch = housing.target.reshape(-1, 1)[indices] # not shown
    return X_batch, y_batch

from datetime import datetime
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)

n_epochs = 1
batch_size = 10
n_batches = int(np.ceil(m / batch_size))
learning_rate = 0.01


X = tf.placeholder(tf.float32, shape=(None, n + 1), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")

theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")

y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)
init = tf.global_variables_initializer()

mse_summary = tf.summary.scalar('MSE', mse)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

with tf.Session() as sess:                                                        # not shown in the book
    sess.run(init)                                                                # not shown
    for epoch in range(n_epochs):                                                 # not shown
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            if batch_index % 10 == 0:
                summary_str = mse_summary.eval(feed_dict={X: X_batch, y: y_batch})
                step = epoch * n_batches + batch_index
                file_writer.add_summary(summary_str, step)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
    best_theta = theta.eval()                                                     # not shown

file_writer.close()


reset_graph()
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
from datetime import datetime
import tensorflow as tf
import numpy as np
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "./tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)
housing = fetch_california_housing()
scaler = StandardScaler()
scaled_housing_data = scaler.fit_transform(housing.data)
m, n = scaled_housing_data.shape
scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]

n_epoches = 100
batch_size = 100
n_batches = int(np.ceil(m / batch_size))
learning_rate = 0.5

X = tf.placeholder(tf.float32, shape=(None, n + 1), name='X')
y = tf.placeholder(tf.float32, shape=(None, 1), name='y')
theta = tf.Variable(tf.random_uniform((n + 1, 1)), name='theta')
y_pred = tf.matmul(X, theta, name='predictions')

with tf.name_scope("loss") as scope:
  error = y - y_pred
  mse = tf.reduce_mean(tf.square(error), name='mse')

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(mse)
init = tf.global_variables_initializer()

mse_summary = tf.summary.scalar("MSE", mse)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())


def fetch_batch(epoch, batch_index, batch_size):
  np.random.seed(epoch * n_batches + batch_index)
  indices = np.random.randint(m, size=batch_size)
  X_batch = scaled_housing_data_plus_bias[indices]
  y_batch = housing.target.reshape(-1, 1)[indices]
  return X_batch, y_batch


with tf.Session() as sess:
  sess.run(init)
  for epoch in range(n_epoches):
    for batch_index in range(n_batches):
      X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
      if batch_size % 10 == 0:
        summary_str = sess.run(mse_summary, feed_dict={X: X_batch, y: y_batch})
        step = epoch * n_batches + batch_index
        file_writer.add_summary(summary_str, step)
      sess.run(train_op, feed_dict={X: X_batch, y: y_batch})

file_writer.close()

n_features = 3
X = tf.placeholder(tf.float32, shape=(None, n_features), name='X')
w = tf.Variable(tf.random_normal((n_features, 1)), name='weights')
b = tf.Variable(0.0, name='bias')
z = tf.add(tf.matmul(X, w), b, name='z')
relu = tf.maximum(z, 0.)

def relu(X):
  w_shape = (int(X.get_shape()[1]), 1)
  w = tf.Variable(tf.random_normal(w_shape), name='weights')
  b = tf.Variable(0.0, name='bias')
  z = tf.add(tf.matmul(X, w), b, name='z')
  return tf.maximum(z, 0.)


reset_graph()
n_features = 3
X = tf.placeholder(tf.float32, shape=(None, n_features), name='X')
relus = [relu(X) for _ in range(5)]
out_put = tf.add_n(relus, name='output')

file_writer = tf.summary.FileWriter('logs/relu1', tf.get_default_graph())

reset_graph()
def relu2(X):
  with tf.name_scope('relu'):
    w_shape = (int(X.get_shape()[1]), 1)
    w = tf.Variable(tf.random_normal(w_shape), name='weights')
    b = tf.Variable(0.0, name='bias')
    z = tf.add(tf.matmul(X, w), b, name='z')
    return tf.maximum(z, 0.)
n_features = 3
X = tf.placeholder(tf.float32, shape=(None, n_features), name='X')
relus = [relu2(X) for _ in range(5)]
out_put = tf.add_n(relus, name='output')
file_writer = tf.summary.FileWriter('logs/relu2', tf.get_default_graph())

file_writer.close()

reset_graph()

with tf.variable_scope('relu'):
  thread_hold = tf.get_variable('threshold', shape=(), initializer=tf.constant_initializer(0.0))

with tf.variable_scope('relu', reuse=True):
  thread_hold = tf.get_variable('threshold')

with tf.variable_scope('relu') as scope:
  scope.reuse_variables()
  thread_hold = tf.get_variable('threshold')

reset_graph()

n_features = 3
def relu(X):
    with tf.variable_scope("relu", reuse=True):
        threshold = tf.get_variable("threshold")
        w_shape = int(X.get_shape()[1]), 1                          # not shown
        w = tf.Variable(tf.random_normal(w_shape), name="weights")  # not shown
        b = tf.Variable(0.0, name="bias")                           # not shown
        z = tf.add(tf.matmul(X, w), b, name="z")                    # not shown
        return tf.maximum(z, threshold, name="max")

X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")

with tf.variable_scope("relu"):
    threshold = tf.get_variable("threshold", shape=(),
                                initializer=tf.constant_initializer(0.0))

relus = [relu(X) for relu_index in range(5)]
output = tf.add_n(relus, name="output")
file_writer = tf.summary.FileWriter("logs/relu7", tf.get_default_graph())
file_writer.close()

