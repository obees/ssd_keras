import tensorflow as tf
from tensorflow.keras.models import load_model
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
###from tensorflow.keras import backend as K

model_path = './saved_models/rgbd_inference.h5'
# We need to create an SSDLoss object in order to pass that to the model loader.
ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

###K.clear_session() # Clear previous models from memory.

model = load_model(model_path, custom_objects={'AnchorBoxes': AnchorBoxes,
                                               'compute_loss': ssd_loss.compute_loss,
                                               'DecodeDetections': DecodeDetections})

model.load_weights('./saved_models/rgbd_weights.h5')

images_dir = '../datasets/rgbd/'
image_path = "1591535321_185807526_rgb.jpeg"
# image=tf.io.read_file("../datasets/rgbd/1591535321_185807526_rgb.jpeg")
image=tf.io.read_file("images/1591535321_185807526_rgb.jpeg")
image = tf.io.decode_jpeg(image)
image.shape
image_list = tf.expand_dims(image, axis=0)
image_list.shape
y_pred2 = model.predict(image_list)
print(y_pred2)