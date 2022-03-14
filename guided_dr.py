import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from model import ResNetBase
import matplotlib.pyplot as plt

dataset = tf.data.TFRecordDataset('./data/real/images.tfrecords')
data_description = {
    'image': tf.io.FixedLenFeature([], tf.string),
    'yaw': tf.io.FixedLenFeature([], tf.float32),
    'forward_accel': tf.io.FixedLenFeature([], tf.float32)
}

brightness = tfp.distributions.Normal(tf.Variable(0., name='b_mu'), tf.Variable(0.33, name='b_sigma'), name='brightness')
contrast = tfp.distributions.Normal(tf.Variable(0., name='c_mu'), tf.Variable(1., name='c_sigma'), name='contrast')
hue = tfp.distributions.Normal(tf.Variable(0., name='h_mu'), tf.Variable(0.33, name='h_sigma'), name='hue')

def parse_data(example):
    return tf.io.parse_single_example(example, data_description)

def structure_data(example):
    image = tf.io.decode_png(example['image'])
    image = tf.image.resize(image, [299, 299])
    image = tf.image.grayscale_to_rgb(image)
    image = tf.cast(image, tf.float16)
    image = image / 255.0
    image = tf.reshape(image, [299, 299, 3])
    responses = [example['yaw'], example['forward_accel']]
    return image, responses

def perturb_data(example, b, c, h):
    image, response = example
    image = tf.image.adjust_brightness(image, b)
    image = tf.image.adjust_contrast(image, c)
    image = tf.image.adjust_hue(image, h)
    return image, response

def plot_dist(ax, alpha):
    b_mu, b_sigma = brightness.loc.numpy(), brightness.scale.numpy()
    x = np.linspace(b_mu - 3 * b_sigma, b_mu + 3 * b_sigma)
    ax[0].plot(x, brightness.prob(x).numpy(), color='b', alpha=alpha)

    c_mu, c_sigma = contrast.loc.numpy(), contrast.scale.numpy()
    x = np.linspace(c_mu - 3 * c_sigma, c_mu + 3 * c_sigma)
    ax[1].plot(x, contrast.prob(x).numpy(), color='r', alpha=alpha)
    
    h_mu, h_sigma = hue.loc.numpy(), hue.scale.numpy()
    x = np.linspace(h_mu - 3 * h_sigma, h_mu + 3 * h_sigma)
    ax[2].plot(x, hue.prob(x).numpy(), color='g', alpha=alpha)
    
parsed_dataset = dataset.map(parse_data)

structured_dataset = parsed_dataset.map(structure_data).shuffle(10)
train, test = structured_dataset.take(300).batch(3), structured_dataset.skip(300).batch(3)

DR_EPOCHS, ETA = 10, 1e-2

optimizer = tf.keras.optimizers.SGD(ETA)
model = ResNetBase(3)
model.compile(optimizer='Adam', loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])

losses = []
    
# fig, ax = plt.subplots(1, 2)
# ax[0].set_title('Target v/s Source Loss', loc='center')
# ax[1].set_title('Transfer Loss', loc='center')
# ax[2].set_title('Hue', loc='center')

for epoch in range(DR_EPOCHS):
    B, C, H = brightness.sample(), contrast.sample(), hue.sample()
    dr_train, dr_test = train.map(lambda i1, r1: perturb_data((i1, r1), B, C, H)), test.map(lambda i2, r2: perturb_data((i2, r2), B, C, H))

    # plot_dist(ax, 0.5 + ((epoch + 1) / (2 * DR_EPOCHS)))

    model.fit(dr_train, epochs=2)

    source_loss, source_metric = model.evaluate(dr_test)
    target_loss, target_metric = model.evaluate(test)

    with tf.GradientTape(persistent=True) as tape:
        transfer_loss = (target_loss - source_loss) ** 2
        losses.append((transfer_loss, source_loss, target_loss))
        b_loss = -tf.math.log(brightness.prob(B) * transfer_loss)
        c_loss = -tf.math.log(contrast.prob(C) * transfer_loss)
        h_loss = -tf.math.log(hue.prob(H) * transfer_loss)

    b_gradient = tape.gradient(b_loss, brightness.trainable_variables)
    c_gradient = tape.gradient(c_loss, contrast.trainable_variables)
    h_gradient = tape.gradient(h_loss, hue.trainable_variables)

    
    optimizer.apply_gradients(zip(b_gradient, brightness.trainable_variables))
    optimizer.apply_gradients(zip(c_gradient, contrast.trainable_variables))
    optimizer.apply_gradients(zip(h_gradient, hue.trainable_variables))

    print()
    print(f'EPOCH #{epoch+1} SUMMARY===============================')
    print('Source Loss:', source_loss)
    print('Target Loss:', target_loss)
    print('Transfer Loss:', transfer_loss)
    print(brightness.loc.numpy(), brightness.scale.numpy(), contrast.loc.numpy(), contrast.scale.numpy(), hue.loc.numpy(), hue.scale.numpy())
    print()

    del tape

# losses = np.array(losses)
# print(losses)
# ax[0].plot(range(DR_EPOCHS), losses[:, 1], color='r')
# ax[0].plot(range(DR_EPOCHS), losses[:, 2], color='g')
# ax[1].plot(range(DR_EPOCHS), losses[:, 0], color='b')
# plt.savefig('./results/fig-2-losses-over-time.png')

print(brightness.loc.numpy(), brightness.scale.numpy(), contrast.loc.numpy(), contrast.scale.numpy(), hue.loc.numpy(), hue.scale.numpy())
    


