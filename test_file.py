import config
import logging
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from utils import (imageData,getBatchDict)
from generator import GeneratorTwo
from discriminator import Discriminator
from customlayers import (ResnetBlock,
                          CustomResizing,
                          CustomResizingTwo,
                          CustomResizingThree,ResidualLayerScal)
# data,countDict,uniq=imageData(config.PATH)
# batchData=getBatchDict(data,uniq,countDict)

# batchSix=np.array(batchData['batch_6'])
# batchSeven=np.array(batchData['batch_7'])
# dat=np.array([batchSix,batchSeven])


# tf.config.run_functions_eagerly(True)


# random_latent_vectors = tf.random.normal(shape=(1, 100))
# generatorTwo=GeneratorTwo().getGeneratorModel()
# print(generatorTwo.summary())
# result=generatorTwo(random_latent_vectors)
# print(result.shape)


data,countDict,uniq=imageData(config.PATH)
batchData=getBatchDict(data,uniq,countDict)
batchTwo=np.array(batchData['batch_2'])
# batchSix=np.array(batchData['batch_6'])
# batchSeven=np.array(batchData['batch_7'])
dat=np.array([batchTwo])
print(batchTwo.shape)
# print(batchSix[0].shape)

# for epoch in range(1):
#     print("\nStart epoch", epoch)
#     for datt in dat:
#         config.batch_size=datt.shape[0] #Dynamic batch size
#         config.shapeOne=datt.shape[1]//8 #Size changing according to the different batch
#         config.shapeTwo=datt.shape[2]//8

#         config.secondShapeOne=datt.shape[1]//4
#         config.secondShapeTwo=datt.shape[2]//4

#         config.ThirdShapeOne=datt.shape[1]//2
#         config.ThirdShapeTwo=datt.shape[2]//2

#         random_latent_vectors = tf.random.normal(shape=(config.batch_size, 100))
#         result=generatorTwo(random_latent_vectors)
#         print(result.shape)
generator=GeneratorTwo().getGeneratorModel()
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
    global batch_size
    # try:
    # Sample random points in the latent space
    random_latent_vectors = tf.random.normal(shape=(config.batch_size, 100))#Dynamic batch size
    print(random_latent_vectors.shape)
    # Decode them to fake images
    generated_images = generator(random_latent_vectors)
    # Combine them with real images
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
    # except Exception as error:
    #     logging.error(
    #         "Error while traing the gan model")
    #     raise Exception(
    #         "Error occurred while training"
    #     ) from error
epochs=50
for epoch in range(epochs):
    print("\nStart epoch", epoch)
    for index,datt in enumerate(dat):
        config.batch_size=datt.shape[0] #Dynamic batch size

        config.shapeOne=datt.shape[1]//8 #Size changing according to the different batch
        config.shapeTwo=datt.shape[2]//8

        config.secondShapeOne=datt.shape[1]//4
        config.secondShapeTwo=datt.shape[2]//4

        config.ThirdShapeOne=datt.shape[1]//2
        config.ThirdShapeTwo=datt.shape[2]//2
        print('Batch_'+str(index))
        tensor = tf.convert_to_tensor(datt,dtype=tf.float32)
        d_loss, g_loss, generated_images = train_step(tensor)#Training
        print('Discriminater Loss is '+' '+str(d_loss))
        print('Generato Loss is '+' '+str(g_loss))
        print(generated_images.shape)
        
img = keras.utils.array_to_img(generated_images[0] * 255.0, scale=False)
img.save( "C:\\veeresh\\veer.png")
        





