"""
dx/dt = -x
x(0) = 0
"""
import numpy as np
import tensorflow as tf
import random
from tensorflow.keras import layers
from tensorflow import keras
# In[]

trange = np.linspace(0,10,101)
X0 = [1+0.01*i for i in range(100)]
data = []
for x0 in X0:
    for t in trange:
        data.append([x0,t,(np.e**(-1*t))])
# In[]
data = np.array(data)
# In[]
t_test = tf.constant(random.uniform(0, 2))
with tf.GradientTape() as g:
    g.watch(t_test)
    x_test = tf.math.exp(-1*t_test)
print(g.gradient(x_test, t_test)+x_test)
# In[]
batch_size = 64
input_train = data[:,0:2]
output_train = data[:,2]
train_dataset = tf.data.Dataset.from_tensor_slices((input_train, output_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
# In[]
inputs = keras.Input(shape=(2,), name="X0_t")
x1 = layers.Dense(64, activation="relu")(inputs)
x2 = layers.Dense(64, activation="relu")(x1)
outputs = layers.Dense(1, name="predictions")(x2)
model = keras.Model(inputs=inputs, outputs=outputs)
optimizer = keras.optimizers.Adam(learning_rate=1e-3)
# Instantiate a loss function.
loss_fn = keras.losses.MeanSquaredError()
# In[]
epochs = 100
for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))

    # Iterate over the batches of the dataset.
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

        # Open a GradientTape to record the operations run
        # during the forward pass, which enables auto-differentiation.
        with tf.GradientTape(persistent=True) as tape:

            # Run the forward pass of the layer.
            # The operations that the layer applies
            # to its inputs are going to be recorded
            # on the GradientTape.

            logits = model(x_batch_train, training=True)  # Logits for this minibatch

            # Compute the loss value for this minibatch.
            loss_value = loss_fn(y_batch_train, logits)
            with tf.GradientTape(persistent=True) as tape2:

            # Run the forward pass of the layer.
            # The operations that the layer applies
            # to its inputs are going to be recorded
            # on the GradientTape.
                tape2.watch(x_batch_train)
                Y_hat = model(x_batch_train)
                #print(Y_hat)
            #print(tf.reduce_mean(tape2.gradient(Y_hat,x_batch_train),axis=0)[1].dtype,tf.reduce_mean(Y_hat).dtype)
            dx_dt = tf.reduce_mean(tape2.gradient(Y_hat,x_batch_train),axis=0)[1]
            x = tf.cast(tf.reduce_mean(Y_hat),dtype='float64')
            error = dx_dt+x
            l2 = tf.square(error)
        #print('l2:',l2)
        #print('loss_value:',loss_value)
        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        g2 = tape.gradient(l2,model.trainable_weights)
        #print(g2)
        grads = tape.gradient(loss_value, model.trainable_weights)+g2
        #+tf.cast(l2,dtype='float32')
        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
    print('epoch%s'%epoch,error)
# In[]
x0 = 1.5
x = []
for t in trange:
    x.append(model.predict(np.array([[x0,t]]))[0][0])
# In[]
import matplotlib.pyplot as plt
plt.plot(trange,x)
plt.plot(trange,[(np.e**(-1*t)) for t in trange])
