import logging
from config import (shapeOne,shapeTwo)
import keras
from keras import layers
from customlayers import (ResnetBlock,
                          CustomResizing,
                          CustomResizingTwo,
                          CustomResizingThree,ResidualLayerScal)


# class Generator:
#     """This class is the generator for any size gan"""
#     def __init__(self,):
#         self.name='Generator'
#         self.latent_dim=100
        
#     def getGeneratorModel(self):
#         """This function creats a generator model using keras

#         Returns:
#             _type_: keras model object
#         """      
#         try:  
#             generator = keras.Sequential(
#             [
#                 keras.Input(shape=(self.latent_dim,)),
#                 layers.Dense(4 * 4 * 256),# This dimention is fixed in the paper
#                 layers.LeakyReLU(alpha=0.2),
#                 layers.Reshape((4, 4, 256)),
#                 CustomResizing(),#first reshape
#                 ResnetBlock(),
#                 layers.Conv2DTranspose(256, (4,4), strides=(2,2), padding='same'),#Second reshape
#                 ResnetBlock(),
#                 layers.Conv2DTranspose(256, (4,4), strides=(2,2), padding='same'),#Third Reshape
#                 ResnetBlock(),
#                 layers.Conv2DTranspose(3, (4,4), strides=(2,2), padding='same'),#Generated Fake Images

#             ],
#             name="generator")
#             return generator
#         except Exception as error:
#             logging.error(
#                 "Error while building generator Model")
#             raise Exception(
#                 "Error occurred while building model."
#             ) from error
        


class GeneratorTwo:
    """This class is the generator for any size gan"""
    def __init__(self,):
        self.name='Generator'
        self.latent_dim=100
        
    def getGeneratorModel(self):
        """This function creats a generator model using keras

        Returns:
            _type_: keras model object
        """      
        # try:  
        generator = keras.Sequential(
        [
            keras.Input(shape=(self.latent_dim,)),
            layers.Dense(4 * 4 * 256),# This dimention is fixed in the paper
            layers.LeakyReLU(alpha=0.2),
            layers.Reshape((4, 4, 256)),
            CustomResizing(),#first reshape
            ResidualLayerScal(),#first res block
            CustomResizingTwo(),#Second reshape
            ResidualLayerScal(),#second res block
            CustomResizingThree(),#Third Reshape
            ResidualLayerScal(),#third res block
            layers.Conv2DTranspose(3, (4,4), strides=(2,2), padding='same'),#Generated Fake Images

        ],
        name="generatorTwo")
        return generator
        # except Exception as error:
        #     logging.error(
        #         "Error while building generator Model")
        #     raise Exception(
        #         "Error occurred while building model."
        #     ) from error
        