import config
import tensorflow as tf
from keras import layers
from keras.layers import Layer
from keras.layers import Add, Activation
from keras.initializers import glorot_uniform


class CustomResizing(Layer):
    """This class creates a custom resizing layer"""

    global shapeOne
    global shapeTwo

    def __init__(self, **kwargs):
        super(CustomResizing, self).__init__(**kwargs)

    def build(
        self,
        input_shapes,
    ):
        pass

    def call(self, x, mask=None):
        newArray = tf.image.resize(x, [config.shapeOne, config.shapeTwo])#Dynamic resizing
        print(newArray.shape)
        return newArray

    def get_config(self):
        base_config = super(CustomResizing, self).get_config()
        return dict(list(base_config.items()))

    def compute_output_shape(self, input_shape):
        return (input_shape[0],) + (input_shape[1] * input_shape[2], input_shape[3])


class ResnetBlock(Layer):
    """This class is a custom resnet block"""

    def __init__(self, **kwargs):
        super(ResnetBlock, self).__init__(**kwargs)

    def build(
        self,
        input_shapes,
    ):
        pass

    def call(self, x, mask=None):
        X_shortcut = x

        X = layers.Conv2D(
            filters=256,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="valid",
            kernel_initializer=glorot_uniform(seed=0),
        )(x)
        X = layers.BatchNormalization(axis=3)(X)
        X = layers.Activation("relu")(X)

        X = layers.Conv2D(
            filters=256,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="valid",
            kernel_initializer=glorot_uniform(seed=0),
        )(x)
        X = layers.BatchNormalization(axis=3)(X)
        X = layers.Activation("relu")(X)

        X = layers.Conv2D(
            filters=256,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="valid",
            kernel_initializer=glorot_uniform(seed=0),
        )(x)
        X = layers.BatchNormalization(axis=3)(X)
        X = layers.Activation("relu")(X)

        X = Add()([X, X_shortcut])
        X = Activation("relu")(X)
        return X

    def get_config(self):
        base_config = super(ResnetBlock, self).get_config()
        return dict(list(base_config.items()))

    def compute_output_shape(self, input_shape):
        return (input_shape[0],) + (input_shape[1] * input_shape[2], input_shape[3])


class CustomInputLayer(Layer):
    """This class is a custom input layer for discriminator"""

    def __init__(self, **kwargs):
        super(CustomInputLayer, self).__init__(**kwargs)

    def call(self, x):
        return x
    


class CustomResizingTwo(Layer):
    """This class creates a custom resizing layer"""

    global secondShapeOne
    global secondShapeTwo

    def __init__(self, **kwargs):
        super(CustomResizingTwo, self).__init__(**kwargs)

    def build(
        self,
        input_shapes,
    ):
        pass

    def call(self, x, mask=None):
        newArray = tf.image.resize(x, [config.secondShapeOne, config.secondShapeTwo])#Dynamic resizing
        print(newArray.shape)
        return newArray

    def get_config(self):
        base_config = super(CustomResizingTwo, self).get_config()
        return dict(list(base_config.items()))

    def compute_output_shape(self, input_shape):
        return (input_shape[0],) + (input_shape[1] * input_shape[2], input_shape[3])
    
    
class CustomResizingThree(Layer):
    """This class creates a custom resizing layer"""

    global ThirdShapeOne
    global ThirdShapeTwo

    def __init__(self, **kwargs):
        super(CustomResizingThree, self).__init__(**kwargs)

    def build(
        self,
        input_shapes,
    ):
        pass

    def call(self, x, mask=None):
        newArray = tf.image.resize(x, [config.ThirdShapeOne, config.ThirdShapeTwo])#Dynamic resizing
        print(newArray.shape)
        return newArray

    def get_config(self):
        base_config = super(CustomResizingThree, self).get_config()
        return dict(list(base_config.items()))

    def compute_output_shape(self, input_shape):
        return (input_shape[0],) + (input_shape[1] * input_shape[2], input_shape[3])

    
class ResidualLayerScal(Layer):
    
    def __init__(self, f=None, s=None, fillter_size_top=None,
               fillter_size_mid=None, fillter_size_bot=None):
        
        super(ResidualLayerScal, self).__init__()
        self.conv_top_1 = layers.Conv2D(256, (1, 1),
                                                 strides=(1, 1), padding='valid')
        # Make the hyperparameters different 
        self.conv_top_2 =layers.Conv2D(256, (1, 1),strides=(1, 1), padding='valid')

        self.batch_norm_top_1 = layers.BatchNormalization(axis=3)
        self.batch_norm_top_2 = layers.BatchNormalization(axis=3)

        self.activation_relu = layers.Activation('relu')
        self.add_op = layers.Add()

    def call(self, input_x, training=False):
        x_shortcut = input_x

        ##PATH 1
        x_path_1 = input_x
        # First CONV block of path 1
        x_path_1 = self.conv_top_1(x_path_1)
        x_path_1 = self.batch_norm_top_1(x_path_1, training=training)
        x_path_1 = self.activation_relu(x_path_1)

        # Second CONV block of path 1
        x_path_1 = self.conv_top_2(x_path_1)
        x_path_1 = self.batch_norm_top_2(x_path_1)
        x_path_1 = self.activation_relu(x_path_1)
        
        x = self.add_op([x_path_1,x_shortcut])
        ##PATH 2
        return x