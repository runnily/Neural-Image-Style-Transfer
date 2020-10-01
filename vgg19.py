"""
This file is to see the layers within the model vgg19
"""
from tensorflow.keras.applications.vgg19 import VGG19
# load model
model = VGG19()
# summarize the model
model.summary()