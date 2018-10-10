import keras.applications.vgg19 as vgg19

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

model = vgg19.VGG19()
model.summary()


