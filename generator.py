import logging
from config import (shapeOne,shapeTwo)
import keras
from keras import layers
from customlayers import (
                          CustomResizing,
                          CustomResizingTwo,
                          CustomResizingThree,ResidualLayerScal,ResidualLayerScalTwo,
                          ResidualLayerScalThree)
        
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
        try:  
            generator = keras.Sequential(
            [
                keras.Input(shape=(self.latent_dim,)),
                layers.Dense(4 * 4 * 512),# This dimention is fixed in the paper
                layers.LeakyReLU(alpha=0.2),
                layers.Reshape((4, 4, 512)),
                CustomResizing(),#first reshape
                ResidualLayerScal(),#first res block
                CustomResizingTwo(),#Second reshape
                ResidualLayerScalTwo(),#second res block
                CustomResizingThree(),#Third Reshape
                ResidualLayerScalThree(),#third res block
                CustomResizingThree(),
                layers.Conv2DTranspose(3, (3,3), strides=(2,2), padding='same'),#Generated Fake Images

            ],
            name="generatorTwo")
            return generator
        except Exception as error:
            logging.error(
                "Error while building generator Model")
            raise Exception(
                "Error occurred while building model."
            ) from error
        