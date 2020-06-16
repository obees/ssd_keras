from IPython import get_ipython

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TerminateOnNaN, CSVLogger
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model

from math import ceil
import numpy as np
from matplotlib import pyplot as plt

from models.keras_ssd7 import build_model
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast

from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast

from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms
from data_generator.data_augmentation_chain_variable_input_size import DataAugmentationVariableInputSize
from data_generator.data_augmentation_chain_constant_input_size import DataAugmentationConstantInputSize
from data_generator.data_augmentation_chain_original_ssd import SSDDataAugmentation



img_height = 300 # Height of the input images
img_width = 480 # Width of the input images
img_channels = 3 # Number of color channels of the input images
intensity_mean = 127.5 # Set this to your preference (maybe `None`). The current settings transform the input pixel values to the interval `[-1,1]`.
intensity_range = 127.5 # Set this to your preference (maybe `None`). The current settings transform the input pixel values to the interval `[-1,1]`.
n_classes = 5 # Number of positive classes
scales = [0.08, 0.16, 0.32, 0.64, 0.96] # An explicit list of anchor box scaling factors. If this is passed, it will override `min_scale` and `max_scale`.
aspect_ratios = [0.5, 1.0, 2.0] # The list of aspect ratios for the anchor boxes
two_boxes_for_ar1 = True # Whether or not you want to generate two anchor boxes for aspect ratio 1
steps = None # In case you'd like to set the step sizes for the anchor box grids manually; not recommended
offsets = None # In case you'd like to set the offsets for the anchor box grids manually; not recommended
clip_boxes = False # Whether or not to clip the anchor boxes to lie entirely within the image boundaries
variances = [1.0, 1.0, 1.0, 1.0] # The list of variances by which the encoded target coordinates are scaled
normalize_coords = False # Whether or not the model is supposed to use coordinates relative to the image size
coords='corners'

# Build Model
# K.clear_session() # Clear previous models from memory.

# model = build_model(image_size=(img_height, img_width, img_channels),
#                     n_classes=n_classes,
#                     mode='training',
#                     l2_regularization=0.0005,
#                     scales=scales,
#                     aspect_ratios_global=aspect_ratios,
#                     aspect_ratios_per_layer=None,
#                     two_boxes_for_ar1=two_boxes_for_ar1,
#                     steps=steps,
#                     offsets=offsets,
#                     clip_boxes=clip_boxes,
#                     variances=variances,
#                     normalize_coords=normalize_coords,
#                     subtract_mean=intensity_mean,
#                     divide_by_stddev=intensity_range)

# adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

# ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

# model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

# Load Model

# model_path = 'ssd7_epoch-15_loss-2.5398_val_loss-2.7014.h5'
# model_path = 'ssd7_epoch-12_loss-2.4442_val_loss-2.5686.h5'
model_path = 'ssd7_epoch-16_loss-2.3541_val_loss-2.5166.h5'

# We need to create an SSDLoss object in order to pass that to the model loader.
ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

K.clear_session() # Clear previous models from memory.

model = load_model(model_path, custom_objects={'AnchorBoxes': AnchorBoxes,
                                               'compute_loss': ssd_loss.compute_loss})

model.run_eagerly = True
# 

# train_dataset = DataGenerator(load_images_into_memory=True, hdf5_dataset_path=None)
val_dataset = DataGenerator(load_images_into_memory=True, hdf5_dataset_path=None)

# Images
images_dir = '../datasets/udacity_driving_datasets/'

# Ground truth
# train_labels_filename = '../datasets/udacity_driving_datasets/labels_train2.csv'
val_labels_filename   = '../datasets/udacity_driving_datasets/labels_val.csv'

# train_dataset.parse_csv(images_dir=images_dir,
#                         labels_filename=train_labels_filename,
#                         input_format=['image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'], # This is the order of the first six columns in the CSV file that contains the labels for your dataset. If your labels are in XML format, maybe the XML parser will be helpful, check the documentation.
#                         include_classes='all')

val_dataset.parse_csv(images_dir=images_dir,
                      labels_filename=val_labels_filename,
                      input_format=['image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'],
                      include_classes='all')

# train_dataset_size = train_dataset.get_dataset_size()
val_dataset_size   = val_dataset.get_dataset_size()

# print("Number of images in the training dataset:\t{:>6}".format(train_dataset_size))
print("Number of images in the validation dataset:\t{:>6}".format(val_dataset_size))

# 3: Set the batch size.

batch_size = 32

# 4: Define the image processing chain.

# data_augmentation_chain = DataAugmentationConstantInputSize(random_brightness=(-48, 48, 0.5),
#                                                             random_contrast=(0.5, 1.8, 0.5),
#                                                             random_saturation=(0.5, 1.8, 0.5),
#                                                             random_hue=(18, 0.5),
#                                                             random_flip=0.5,
#                                                             random_translate=((0.03,0.5), (0.03,0.5), 0.5),
#                                                             random_scale=(0.5, 2.0, 0.5),
#                                                             n_trials_max=3,
#                                                             clip_boxes=clip_boxes,
#                                                             overlap_criterion='area',
#                                                             bounds_box_filter=(0.3, 1.0),
#                                                             bounds_validator=(0.5, 1.0),
#                                                             n_boxes_min=1,
#                                                             background=(0,0,0))

# 5: Instantiate an encoder that can encode ground truth labels into the format needed by the SSD loss function.

# The encoder constructor needs the spatial dimensions of the model's predictor layers to create the anchor boxes.
predictor_sizes = [model.get_layer('classes4').output_shape[1:3],
                   model.get_layer('classes5').output_shape[1:3],
                   model.get_layer('classes6').output_shape[1:3],
                   model.get_layer('classes7').output_shape[1:3]]

ssd_input_encoder = SSDInputEncoder(img_height=img_height,
                                    img_width=img_width,
                                    n_classes=n_classes,
                                    predictor_sizes=predictor_sizes,
                                    scales=scales,
                                    aspect_ratios_global=aspect_ratios,
                                    two_boxes_for_ar1=two_boxes_for_ar1,
                                    steps=steps,
                                    offsets=offsets,
                                    clip_boxes=clip_boxes,
                                    variances=variances,
                                    matching_type='bipartite',
                                    pos_iou_threshold=0.5,
                                    neg_iou_limit=0.3,
                                    normalize_coords=normalize_coords,
                                    coords=coords)

# ssd_input_encoder = SSDInputEncoder(img_height=img_height,
#                                     img_width=img_width,
#                                     n_classes=n_classes,
#                                     predictor_sizes=predictor_sizes,
#                                     scales=scales,
#                                     aspect_ratios_global=aspect_ratios,
#                                     two_boxes_for_ar1=two_boxes_for_ar1,
#                                     steps=steps,
#                                     offsets=offsets,
#                                     clip_boxes=clip_boxes,
#                                     variances=variances,
#                                     matching_type='multi',
#                                     pos_iou_threshold=0.5,
#                                     neg_iou_limit=0.3,
#                                     normalize_coords=normalize_coords)


# 6: Create the generator handles that will be passed to Keras' `fit_generator()` function.

# ob_generator = train_dataset.generate(batch_size=batch_size,
#                                          shuffle=True,
#                                          transformations=[data_augmentation_chain],
#                                          label_encoder=ssd_input_encoder,
#                                          returns={'processed_images',
#                                                   'encoded_labels',
#                                                   'matched_anchors',
#                                                   'processed_labels',
#                                                   'filenames',
#                                                   'image_ids',
#                                                   'evaluation-neutral',
#                                                   'inverse_transform',
#                                                   'original_images',
#                                                   'original_labels'
#                                                   },
#                                          keep_images_without_gt=False)

# train_generator = train_dataset.generate(batch_size=batch_size,
#                                          shuffle=True,
#                                          transformations=[],
#                                          label_encoder=ssd_input_encoder,
#                                          returns={'processed_images',
#                                                   'encoded_labels'},
#                                          keep_images_without_gt=False)

# # trainX, trainy = train_dataset.generate_oneshot_encoded(label_encoder=ssd_input_encoder)

# val_generator = val_dataset.generate(batch_size=batch_size,
#                                      shuffle=False,
#                                      transformations=[],
#                                      label_encoder=ssd_input_encoder,
#                                      returns={'processed_images',
#                                               'encoded_labels'},
#                                      keep_images_without_gt=False)

# ob_processed_images, ob_encoded_labels, ob_matched_anchors, ob_processed_labels, ob_filenames, ob_image_ids, ob_evaluation_neutral, ob_inverse_transform,ob_original_images, ob_original_labels = next(ob_generator)

#model_checkpoint = ModelCheckpoint(filepath='ssd7_epoch-{epoch:02d}_loss-{loss:.4f}_val_loss-{val_loss:.4f}.h5',
# model_checkpoint = ModelCheckpoint(filepath='ssd7_epoch-{epoch:02d}_val_loss-{val_loss:.4f}.h5',
#                                    monitor='val_loss',
#                                    verbose=1,
#                                    save_best_only=True,
#                                    save_weights_only=False,
#                                    mode='auto',
                                   
#                                    period=1)

# csv_logger = CSVLogger(filename='ssd7_training_log.csv',
#                        separator=',',
#                        append=True)

# early_stopping = EarlyStopping(monitor='val_loss',
#                                min_delta=0.0,
#                                patience=10,
#                                verbose=1)

# reduce_learning_rate = ReduceLROnPlateau(monitor='val_loss',
#                                          factor=0.2,
#                                          patience=8,
#                                          verbose=1,
#                                          epsilon=0.001,
#                                          cooldown=0,
#                                          min_lr=0.00001)

# callbacks = [model_checkpoint,
#              csv_logger,
#              early_stopping,
#              reduce_learning_rate]

# initial_epoch   = 0
# final_epoch     = 20
# steps_per_epoch = 1000

# history = model.fit(x=train_generator,
#                               steps_per_epoch=steps_per_epoch,
#                               epochs=final_epoch,
#                               callbacks=callbacks,
#                               initial_epoch=initial_epoch)

# history = model.fit(x=trainX, y=trainy,
#                               steps_per_epoch=steps_per_epoch,
#                               epochs=final_epoch,
#                               initial_epoch=initial_epoch)

# plt.figure(figsize=(20,12))
# plt.plot(history.history['loss'], label='loss')
# plt.plot(history.history['val_loss'], label='val_loss')
# plt.legend(loc='upper right', prop={'size': 24})


# Predict

predict_generator = val_dataset.generate(batch_size=1,
                                         shuffle=False,
                                         transformations=[],
                                         label_encoder=ssd_input_encoder,
                                         returns={'processed_images',
                                                  'processed_labels',
                                                  'filenames',
                                                  'encoded_labels',
                                                  'matched_anchors',
                                                  'image_ids',
                                                  'original_images',
                                                  'original_labels'},
                                         keep_images_without_gt=False)


processed_images, encoded_labels,matched_anchors, processed_labels, batch_filenames,  image_ids,  original_images, original_labels = next(predict_generator)



i = 0 # Which batch item to look at

while not batch_filenames[i]== "../datasets/udacity_driving_datasets/1478899063275530566.jpg":
    processed_images, encoded_labels,matched_anchors, processed_labels, batch_filenames,  image_ids,  original_images, original_labels = next(predict_generator)


print("Image:", batch_filenames[i])
print()
print("Ground truth boxes:\n")
print(processed_labels[i])

y_pred = model.predict(processed_images)

y_pred_decoded = decode_detections(y_pred,
                                   confidence_thresh=0.3,
                                   iou_threshold=0.30,
                                   top_k=200,
                                   normalize_coords=normalize_coords,
                                   img_height=img_height,
                                   img_width=img_width,
                                   input_coords='corners')
                                    # confidence_thresh=0.01,
                                    # iou_threshold=0.45,
                                    # top_k=200,
                                    # input_coords='centroids',
                                    # normalize_coords=True,
                                    # img_height=None,
                                    # img_width=None,
                                    # border_pixels='half'

np.set_printoptions(precision=2, suppress=True, linewidth=90)
print("Predicted boxes:\n")
print('   class   conf xmin   ymin   xmax   ymax')
print(y_pred_decoded[i])

plt.figure(figsize=(20,12))
plt.imshow(processed_images[i].numpy())

current_axis = plt.gca()

colors = plt.cm.hsv(np.linspace(0, 1, n_classes+1)).tolist() # Set the colors for the bounding boxes
classes = ['background', 'car', 'truck', 'pedestrian', 'bicyclist', 'light'] # Just so we can print class names onto the image instead of IDs

# Draw the ground truth boxes in green (omit the label for more clarity)
for box in processed_labels[i]:
    xmin = box[1]
    ymin = box[2]
    xmax = box[3]
    ymax = box[4]
    label = '{}'.format(classes[int(box[0])])
    current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color='green', fill=False, linewidth=2))  
    #current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':'green', 'alpha':1.0})

# Draw the predicted boxes in blue
for box in y_pred_decoded[i]:
    xmin = box[-4]
    ymin = box[-3]
    xmax = box[-2]
    ymax = box[-1]
    color = colors[int(box[0])]
    label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
    current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color=color, fill=False, linewidth=2))  
    current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':color, 'alpha':1.0})
plt.show()
imag = plt.imread(batch_filenames[i])
plt.imshow(imag)
plt.show()
print("oh")