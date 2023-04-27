import os
import cv2
import logging
import config
import tensorflow as tf
import keras
import config
from generator import GeneratorTwo
from discriminator import Discriminator

def imageData(path):
    """This function loads the images and 
    count the unique batch size and image size

    Args:
        path (_type_):It is the path to the image folder

    Raises:
        Exception:None type image found

    Returns:
        _type_:Returns image stored in an array
        with the count of different image size after 
        applying the resizing technique fom the paper
    """    
    imageArray=[]
    uniqueValueCountDict={}
    try:
        for img in os.listdir(path):
            image=cv2.imread(path+'\\'+img)
            imageArray.append(image)
            val=image.shape[0]
            if val in uniqueValueCountDict:
                uniqueValueCountDict[val]=uniqueValueCountDict[val]+1
            else:
                uniqueValueCountDict[val]=1    
        return imageArray,uniqueValueCountDict,list(uniqueValueCountDict.keys())
    except Exception as error:
        logging.error(
            "Error while extracting image data and unique Count")
        raise Exception(
            "Error occurred while extracting image data."
        ) from error
        
def getBatchDict(data,uniqueValueList,countDict):
    """_summary_

    Args:
        data (_type_): Images stored in a list

        uniqueValueList (_type_):Dictonary containing count of 
        unique image size
        Example:{128: 188,104: 215}

        countDict (_type_): list containg unique image size

    Raises:
        Exception: index out of range

    Returns:
        _type_: Returns a dictonairy with key as batch number
        and vaues as images
        Example:{'batch_size_1':[imageOne,imageTwo],
                'batch_size_2':[imageOne,imageTwo]}
    """    
    batchDict={}
    try:
        for index,value in enumerate(uniqueValueList):
            valueCount=0
            imageList=[]
            for img in data:
                if img.shape[0]==value:
                    valueCount=valueCount+1
                    imageList.append(img)
                if valueCount==countDict[value]:
                    break
            batchDict['batch_'+str(index+1)]=imageList
        return batchDict
    except Exception as error:
        logging.error(
            "Error while extracting data in batches")
        raise Exception(
            "Error occurred while extracting data......"
        ) from error

latent_dim=100
generator=GeneratorTwo().getGeneratorModel()
untrainedGenerator=GeneratorTwo().getGeneratorModel()
discriminator=Discriminator().getDiscriminatorModel()
d_optimizer = keras.optimizers.Adam(learning_rate=0.03)
g_optimizer = keras.optimizers.Adam(learning_rate=0.04)
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
    try:
        # Sample random points in the latent space
        random_latent_vectors = tf.random.normal(shape=(config.batch_size, latent_dim))#Dynamic batch size
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
        d_optimizer.apply_gradients(zip(grads, discriminator.trainable_weights))

        # Sample random points in the latent space
        random_latent_vectors = tf.random.normal(shape=(config.batch_size, latent_dim))
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
    except Exception as error:
        logging.error(
            "Error while traing the gan model")
        raise Exception(
            "Error occurred while training"
        ) from error