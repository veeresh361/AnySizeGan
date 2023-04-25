import config
import config
import numpy as np
import tensorflow as tf
import keras
from utils import (train_step,imageData,getBatchDict)

data,countDict,uniq=imageData(config.PATH)
batchData=getBatchDict(data,uniq,countDict)

batchSix=np.array(batchData['batch_6'])
batchSeven=np.array(batchData['batch_7'])
batchTwo=np.array(batchData['batch_2'])
batchThree=np.array(batchData['batch_3'])
dat=np.array([batchSix,batchSeven])
#dat=np.array([batchTwo,batchThree,batchSix,batchSeven])


tf.config.run_functions_eagerly(True)

epochs = 20

# for epoch in range(epochs):
#     print("\nStart epoch", epoch)
#     for datt in dat:
#         config.batch_size=datt.shape[0] #Dynamic batch size
#         config.shapeOne=datt.shape[1]//8 #Size changing according to the different batch
#         config.shapeTwo=datt.shape[2]//8
#         print(config.shapeOne,config.shapeTwo)
#         tensor = tf.convert_to_tensor(datt,dtype=tf.float32)
#         d_loss, g_loss, generated_images = train_step(tensor)#Training
#         break

# print(d_loss)





for epoch in range(epochs):
    print("\nStart epoch", epoch)
    for index,datt in enumerate(dat):
        config.batch_size=datt.shape[0] #Dynamic batch size

        config.shapeOne=datt.shape[1]//8 #Size changing according to the different batch
        config.shapeTwo=datt.shape[2]//8

        config.secondShapeOne=datt.shape[1]//4
        config.secondShapeTwo=datt.shape[2]//4

        config.ThirdShapeOne=datt.shape[1]//2
        config.ThirdShapeTwo=datt.shape[1]//2
        print('Batch_'+str(index))
        tensor = tf.convert_to_tensor(datt,dtype=tf.float32)
        d_loss, g_loss, generated_images = train_step(tensor)#Training
        print('Discriminater Loss is '+' '+str(d_loss))
        print('Generato Loss is '+' '+str(g_loss))
img = keras.utils.array_to_img(generated_images[0] * 255.0, scale=False)
img.save( "C:\\veeresh\\veer.png")