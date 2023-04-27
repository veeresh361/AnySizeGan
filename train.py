import config
import numpy as np
import tensorflow as tf
import keras
from utils import (train_step,imageData,getBatchDict)

data,countDict,uniq=imageData(config.PATH)
batchData=getBatchDict(data,uniq,countDict)

tf.config.run_functions_eagerly(True)

epochs=10
for epoch in range(epochs):
    #print("\nStart epoch", epoch)
    for key in batchData:
      datt=batchData[key]
      datt=np.array(datt)
      #print(datt.shape)
      shapeOne=datt.shape[1]//8 #Size changing according to the different batch
      shapeTwo=datt.shape[2]//8
      #print(shapeOne,shapeTwo)

      secondShapeOne=datt.shape[1]//4
      secondShapeTwo=datt.shape[2]//4

      ThirdShapeOne=datt.shape[1]//2
      ThirdShapeTwo=datt.shape[2]//2
      #generator.summary()
      for i in range(0,datt.shape[0],32):#Taking 32 images from each image size
        batchImages=datt[i:i+32,:,:,:]
        batch_size=batchImages.shape[0]
        batchImages=tf.cast(batchImages, tf.float32)
        d_loss, g_loss, generated_images = train_step(batchImages)
      print('Discriminater Loss is '+' '+str(d_loss))
      print('Generato Loss is '+' '+str(g_loss))