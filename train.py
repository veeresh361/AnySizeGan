import config
import numpy as np
import cv2
import tensorflow as tf
import keras
from utils import (train_step,imageData,getBatchDict)
from generator import GeneratorTwo
from discriminator import Discriminator

# data,countDict,uniq=imageData(config.PATH)
# batchData=getBatchDict(data,uniq,countDict)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
y_train=np.squeeze(y_train)


labelOne=x_train[y_train==0]
labelOne=labelOne[:1000]
labelOne=tf.convert_to_tensor(labelOne)
labelOne=tf.image.resize(labelOne,[40,40])

labelTwo=x_train[y_train==1]
labelTwo=labelTwo[:1000]
labelTwo=tf.convert_to_tensor(labelTwo)
labelTwo=tf.image.resize(labelTwo,[48,48])

labelThree=x_train[y_train==2]
labelThree=labelTwo[:1000]
labelThree=tf.convert_to_tensor(labelThree)
labelThree=tf.image.resize(labelThree,[48,48])

data=[labelOne,labelTwo,labelThree]

tf.config.run_functions_eagerly(True)
latent_dim=100
generator=GeneratorTwo().getGeneratorModel()
untrainedGenerator=GeneratorTwo().getGeneratorModel()
discriminator=Discriminator().getDiscriminatorModel()
d_optimizer = keras.optimizers.Adam(learning_rate=0.0003)
g_optimizer = keras.optimizers.Adam(learning_rate=0.0004)
loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)

tf.config.run_functions_eagerly(True)
@tf.function
def train_step(real_images):
    """This function train the any size gan model
    with different size images.

    Args:
        real_images (_type_): _description_

    Returns:
        _type_: retun loss, accuracy and generated image by
        the generator
    """    
    #global batch_size
    # try:
    # Sample random points in the latent space
    random_latent_vectors = tf.random.normal(shape=(config.batch_size, 100))#Dynamic batch size
    #print(random_latent_vectors.shape)
    # Decode them to fake images
    generated_images = generator(random_latent_vectors)
    # Combine them with real images
    #print(generated_images.shape)
    combined_images = tf.concat([generated_images, real_images], axis=0)

    # Assemble labels discriminating real from fake images
    labels = tf.concat(
        [tf.ones((config.batch_size, 1)), tf.zeros((real_images.shape[0], 1))], axis=0
    )
    # Add random noise to the labels - important trick!
    labels += 0.05 * tf.random.uniform(labels.shape)

    # Train the discriminator
    with tf.GradientTape() as tape:
        predictions = discriminator(combined_images)
        d_loss = loss_fn(labels, predictions)
    grads = tape.gradient(d_loss, discriminator.trainable_weights)
    #print(grads)
    d_optimizer.apply_gradients(zip(grads, discriminator.trainable_weights))

    # Sample random points in the latent space
    random_latent_vectors = tf.random.normal(shape=(config.batch_size, 100))
    # Assemble labels that say "all real images"
    misleading_labels = tf.zeros((config.batch_size, 1))

    # Train the generator (note that we should *not* update the weights
    # of the discriminator)!
    with tf.GradientTape() as tape:
        predictions = discriminator(generator(random_latent_vectors))
        g_loss = loss_fn(misleading_labels, predictions)
    grads = tape.gradient(g_loss, generator.trainable_weights)
    g_optimizer.apply_gradients(zip(grads, generator.trainable_weights))
    return d_loss, g_loss, generated_images

epochs=3
for epoch in range(epochs):
    print("\nStart epoch", epoch)
    for index,datt in enumerate(data):
      config.shapeOne=datt.shape[1]//8 #Size changing according to the different batch
      config.shapeTwo=datt.shape[2]//8
      #print(shapeOne,shapeTwo)

      config.secondShapeOne=datt.shape[1]//4
      config.secondShapeTwo=datt.shape[2]//4

      config.ThirdShapeOne=datt.shape[1]//2
      config.ThirdShapeTwo=datt.shape[2]//2
      #print(datt.shape)
      for i in range(0,datt.shape[0],32):
        batchImages=datt[i:i+32,:,:,:]
        batch_size=batchImages.shape[0]
        batchImages=tf.cast(batchImages, tf.float32)
        d_loss, g_loss, generated_images = train_step(batchImages)
      print('Discriminater Loss is '+' '+str(d_loss))
      print('Generato Loss is '+' '+str(g_loss))


OriginalShapeOne=48
OriginalShapeTwo=48

shapeOne=labelTwo.shape[1]//8 #Size changing according to the different batch
shapeTwo=labelTwo.shape[2]//8
#print(shapeOne,shapeTwo)

secondShapeOne=labelTwo.shape[1]//4
secondShapeTwo=labelTwo.shape[2]//4

ThirdShapeOne=labelTwo.shape[1]//2
ThirdShapeTwo=labelTwo.shape[2]//2

random_latent_vectors = tf.random.normal(shape=(1, 100))
result=generator.predict(random_latent_vectors)
result=np.array(result)
cv2.imshow('rt',result)
cv2.waitKey(0)