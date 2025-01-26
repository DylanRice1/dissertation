from imports import *

IMG_SIZE = (400,400)

# Defining the base model
efficientNet_model = tf.keras.applications.resnet50.ResNet50(include_top = False,
                                                     weights = 'imagenet',
                                                     input_shape = (IMG_SIZE[0], IMG_SIZE[1], 3))

inception_v3_model = tf.keras.applications.inception_v3.InceptionV3(include_top = False,
                                                                    weights = 'imagenet',
                                                                    input_shape = (IMG_SIZE[0], IMG_SIZE[1], 3))

mobileNet_model = tf.keras.applications.resnet50.ResNet50(include_top = False,
                                                     weights = 'imagenet',
                                                     input_shape = (IMG_SIZE[0], IMG_SIZE[1], 3))

vgg16_model = tf.keras.applications.vgg16.VGG16(include_top = False,
                                                weights = 'imagenet',
                                                input_shape = (IMG_SIZE[0], IMG_SIZE[1], 3))

base_model = tf.keras.applications.resnet50.ResNet50(include_top = False,
                                                     weights = 'imagenet',
                                                     input_shape = (IMG_SIZE[0], IMG_SIZE[1], 3))