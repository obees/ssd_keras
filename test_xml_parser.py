from data_generator.object_detection_2d_data_generator import DataGenerator

val_dataset = DataGenerator(load_images_into_memory=True, hdf5_dataset_path=None)

images_dir = '../datasets/rgbd/'


val_dataset.parse_labelimg_xml(images_dir=images_dir,
                      labels_filename=val_labels_filename,
                      input_format=['image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'],
                      include_classes='all')