import logging
import keras
from keras import layers
from customlayers import CustomInputLayer


class Discriminator:
    """This class creates discriminator model using keras"""

    def __init__(
        self,
    ):
        self.name = "Discriminater"

    def getDiscriminatorModel(self):
        """This function creates discriminator model

        Returns:
            _type_:keras model object
        """
        try:
            discriminator = keras.Sequential(
                [
                    CustomInputLayer(),
                    layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same"),
                    layers.LeakyReLU(alpha=0.2),
                    layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same"),
                    layers.LeakyReLU(alpha=0.2),
                    layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same"),
                    layers.LeakyReLU(alpha=0.2),
                    layers.GlobalMaxPooling2D(),
                    #layers.Dense(128),
                    layers.Dense(1),
                ],
                name="discriminator",
            )
            return discriminator

        except Exception as error:
            logging.error("Error while building Discriminator Model")
            raise Exception("Error occurred while building model.") from error
