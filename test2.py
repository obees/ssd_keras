from data_generator.object_detection_2d_data_generator import DataGenerator

train_dataset = DataGenerator(load_images_into_memory=True, hdf5_dataset_path=None)

train_images_dirs = ['../datasets/rgbd/']
train_image_set_filenames = ['../datasets/rgbd/train.txt']
train_annotations_dirs = ['../datasets/rgbd/labels/']
train_classes=['background','gate']

val_images_dirs = ['../datasets/rgbd/']
val_image_set_filenames = ['../datasets/rgbd/val.txt']
val_annotations_dirs = ['../datasets/rgbd/labels/']
val_classes=['background','gate']

train_dataset.parse_labelimg_xml(train_images_dirs,
                                 train_image_set_filenames,
                                 train_annotations_dirs,
                                 train_classes)