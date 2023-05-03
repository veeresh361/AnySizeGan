import config
import numpy as np
import tensorflow as tf
import keras
from utils import (train_step,imageData,getBatchDict)

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

epochs=1
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