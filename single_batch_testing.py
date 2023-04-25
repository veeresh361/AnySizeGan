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

data,countDict,uniq=imageData(config.PATH)
batchData=getBatchDict(data,uniq,countDict)
#batchOne=np.array(batchData['batch_1'])
batchTwo=np.array(batchData['batch_2'])
# batchSix=np.array(batchData['batch_6'])
# batchSeven=np.array(batchData['batch_7'])
# dat=np.array([batchSix,batchSeven])
print(batchTwo.shape)

config.batch_size=batchTwo.shape[0] #Dynamic batch size

config.shapeOne=batchTwo.shape[1]//8 #Size changing according to the different batch
config.shapeTwo=batchTwo.shape[2]//8

config.secondShapeOne=batchTwo.shape[1]//4
config.secondShapeTwo=batchTwo.shape[2]//4

config.ThirdShapeOne=batchTwo.shape[1]//2
config.ThirdShapeTwo=batchTwo.shape[2]//2     
tensor = tf.convert_to_tensor(batchTwo,dtype=tf.float32)
random_latent_vectors = tf.random.normal(shape=(config.batch_size, 100))
print(random_latent_vectors.shape)
generatorTwo=GeneratorTwo().getGeneratorModel()
result=generatorTwo(random_latent_vectors)
print(result.shape)