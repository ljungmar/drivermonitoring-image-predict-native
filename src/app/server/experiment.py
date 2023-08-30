import h5py
from pathlib import Path
import os
import json
from PIL import Image
import numpy as np
import argparse
import tensorflow as tf
import datetime
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
import efficientnet.keras as efn
import argparse
#pip3 install -U efficientnet

# %load_ext tensorboard




from tensorflow.keras.preprocessing.image import load_img, img_to_array, smart_resize

def load_and_preprocess_image(image_path, rows, cols):
    # Load the image using PIL
    img = Image.open(image_path)
    
    print("Original Image Shape:", img.size)
    
    # Resize the image to at least 75x75 pixels
    min_size = 75
    img = img.resize((max(min_size, rows), max(min_size, cols)))

    # Crop the image to the specified size if necessary
    img = img.crop((0, 0, rows, cols))

    # Convert the image to a NumPy array
    img_array = np.array(img)

    # Preprocess the image according to the requirements of the chosen model
    preprocessed_img = preprocess_input(img_array)

    # Return the preprocessed image
    print("Processed Image Shape:", preprocessed_img.shape)

    return preprocessed_img

models_dir = "saved_models"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)


def read_hdf5(hdf5_dir=Path('./'), subset='train', rows=256, cols=256):
    """ Reads image from HDF5.
        Parameters:
        ---------------
        hdf5_dir   path where the h5 file is
        rows       number of pixel rows (in the image)
        cols        number of pixel cols (in the image)
        Returns:
        ----------
        images      images array, (N, rows, cols, 3) to be read
        labels      associated meta data, int label (N, 1)
    """
    images, labels = [], []
    # Open the HDF5 file
    file = h5py.File(hdf5_dir / f"driver_distraction_{rows}x{cols}_{subset}.h5", "r+")
    #images = np.array(file["/images"]).astype(float)
    #labels = np.array(file["/meta"]).astype(int)
    return file #images, labels


def batch_image_generator(images, labels, batch_size=32, data_augmentation=None):
    y_train = to_categorical(labels, num_classes=2)
    sample_count = len(labels)
    while True:
        for index in range(sample_count//batch_size):
            start = index * batch_size
            end = (index + 1) * batch_size
            x = images[start: end]
            y = y_train[start: end]
            if data_augmentation is not None:
                generator = data_augmentation.flow(x, y, batch_size=batch_size)
                x, y = next(generator)
            yield x, y


def create_vgg16_V2(rows, cols):
    """
    Architecture and adaptation of the VGG16 for our project
    """
    print('VGG16 Version 2 will be trained on the dataset')
    nb_classes = 2
    # Remove fully connected layer and replace
    vgg16_model = VGG16(input_shape = (rows, cols, 3), weights="imagenet", include_top=False)
    for layer in vgg16_model.layers:
        layer.trainable = True
    x = vgg16_model.output
    x = Flatten(name="flatten")(vgg16_model.output)
    #x = Dense(4096, activation='relu')(x)
    #x = Dense(2048, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(nb_classes, activation='softmax')(x)  # Output layer with 2 neurons

    model = Model(vgg16_model.input, predictions)
    return model


def create_vgg16(rows, cols):
    """
    Architecture and adaptation of the VGG16 for our project
    """
    nb_classes = 2
    # Remove fully connected layer and replace
    vgg16_model = VGG16(input_shape = (rows, cols, 3), weights="imagenet", include_top=False)
    for layer in vgg16_model.layers:
        layer.trainable = False
    x = GlobalAveragePooling2D()(vgg16_model.output) # 512 outputs
    x = Dense(2048, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    #x = Dense(64, activation='relu')(x)
    predictions = Dense(nb_classes, activation='softmax')(x)  # add dense layer with 10 neurons and activation softmax
    model = Model(vgg16_model.input, predictions)
    return model


def create_vgg16(rows, cols):
    """
    Architecture and adaptation of the VGG16 for our project
    """
    nb_classes = 2
    # Remove fully connected layer and replace
    vgg16_model = VGG16(input_shape = (rows, cols, 3), weights="imagenet", include_top=False)
    for layer in vgg16_model.layers:
        layer.trainable = False
    x = GlobalAveragePooling2D()(vgg16_model.output) # 512 outputs
    x = Dense(2048, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    #x = Dense(64, activation='relu')(x)
    predictions = Dense(nb_classes, activation='softmax')(x)  # add dense layer with 10 neurons and activation softmax
    model = Model(vgg16_model.input, predictions)
    return model


def create_vgg16_ft(rows, cols):
    """
    Architecture and adaptation of the VGG16 for our project
    """
    print('VGG16 Version Finetuning will be trained on the dataset')
    nb_classes = 2
    # Remove fully connected layer and replace
    vgg16_model = VGG16(input_shape = (rows, cols, 3), weights="imagenet", include_top=False)
    for layer in vgg16_model.layers:
        layer.trainable = True
    x = GlobalAveragePooling2D()(vgg16_model.output) # 512 outputs
    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    #x = Dense(64, activation='relu')(x)
    predictions = Dense(nb_classes, activation='softmax')(x)  # add dense layer with 10 neurons and activation softmax
    model = Model(vgg16_model.input, predictions)
    return model


def create_inception(rows, cols, nb_classes):
    """
    Architecture and adaptation of the VGG16 for our project
    """
    nb_classes = 2
    # Remove fully connected layer and replace
    inception_model = InceptionV3(input_shape = (rows, cols, 3), include_top=False, weights='imagenet')
    for layer in inception_model.layers:
        layer.trainable = False
    x = Flatten()(inception_model.output)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    predictions = Dense(nb_classes, activation='softmax')(x)  # Output layer with 2 neurons
    model = Model(inception_model.input, predictions)
    return model


def create_efficientnet(rows, cols):
    """
    Architecture and adaptation of the VGG16 for our project
    """
    nb_classes = 2
    # Remove fully connected layer and replace
    base_model = efn.EfficientNetB0(input_shape=(rows, cols, 3), include_top=False, weights='imagenet')
    for layer in base_model.layers:
        layer.trainable = False
    x = Flatten()(base_model.output)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    predictions = Dense(nb_classes, activation='softmax')(x)  # add dense layer with 10 neurons and activation softmax
    model = Model(base_model.input, predictions)
    return model


def create_resnet(rows, cols):
    """
    Architecture and adaptation of the VGG16 for our project
    """
    nb_classes = 2
    # Remove fully connected layer and replace
    resnet_model = ResNet50(input_shape = (rows, cols, 3),weights="imagenet", include_top=False)
    for layer in resnet_model.layers:
        layer.trainable = False
    x = Flatten()(resnet_model.output)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    predictions = Dense(nb_classes, activation='softmax')(x)  # add dense layer with 10 neurons and activation softmax
    model = Model(resnet_model.input, predictions)
    return model

def create_mobilenet_v2(rows, cols):
    base_model = MobileNetV2(input_shape=(rows, cols, 3), include_top=False, weights='imagenet')
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_model(name, rows, cols):
    nb_classes = 2  # Number of output classes (number of classes in your dataset)
    
    if 'inc' in name.lower():
        return create_inception(rows, cols, nb_classes)  # Use InceptionV3 with 2 classes
    if 'vgg2' in name.lower():
        return create_vgg16_V2(rows, cols)  # Update the function parameters as needed
    if 'vggft' in name.lower():
        return create_vgg16_ft(rows, cols)  # Update the function parameters as needed
    if 'vgg' in name.lower():
        return create_vgg16(rows, cols)  # Update the function parameters as needed
    if 'inc' in name.lower():
        return create_inception(rows, cols)  # Update the function parameters as needed
    if 'res' in name.lower():
        return create_resnet(rows, cols)  # Update the function parameters as needed
    if 'eff' in name.lower():
        return create_efficientnet(rows, cols)  # Update the function parameters as needed
    if 'mob' in name.lower():
        return create_mobilenet_v2(rows, cols)  # Update the function parameters as needed
    raise AttributeError('The model ' + str(name) + ' is not available. Choose from: VGG, inceptionNet, ResNet, EfficientNet, or MobileNetV2.')

def training_with_dataaugmentation(file_train, model, nb_epoch, batch_size, rows, cols, train_datagen=None):
  
    if train_datagen is None:
      train_datagen = ImageDataGenerator(
          zoom_range=[0.1, 5.0],
          brightness_range=[0.1, 6.5],
          rotation_range=70,
          horizontal_flip=True,
          height_shift_range=0.4,
          width_shift_range=0.5,
          rescale = 1.0/255,
          fill_mode='nearest',
          shear_range = 0.2,
      )
    val_datagen = ImageDataGenerator(rescale = 1.0/255, validation_split=0.2)

    images = file_train["/images"]
    labels = np.array(file_train["/meta"]).astype(int)
    pivot = images.shape[0] - round(images.shape[0]*0.2)

    training_generator = batch_image_generator(images[:pivot], labels[:pivot], batch_size, data_augmentation=train_datagen)
    validation_generator = batch_image_generator(images[pivot:], labels[pivot:], batch_size, data_augmentation=val_datagen)

    ds_train = tf.data.Dataset.from_generator(
        lambda: batch_image_generator(images[:pivot], labels[:pivot], batch_size,train_datagen),
        output_types=(tf.float32, tf.float32),
        output_shapes=([batch_size, rows, cols, 3], [batch_size, 2])
    )

    ds_vali = tf.data.Dataset.from_generator(
        lambda: batch_image_generator(images[:pivot], labels[:pivot], batch_size, train_datagen),
        output_types=(tf.float32, tf.float32),
        output_shapes=([batch_size, rows, cols, 3], [batch_size, 2])
    )

    nb_train_samples = len(labels[:pivot])
    nb_validation_samples = len(labels[pivot:])
    log_dir = "/content/drive/MyDrive/logs/fit_ft/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint_path = "/content/drive/MyDrive/logs/cpft-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=1,
        save_weights_only=True, save_freq='epoch', period=5)
#    model.save_weights(checkpoint_path.format(epoch=0))

    history = model.fit(ds_train,
                         steps_per_epoch=nb_train_samples // batch_size,
                         epochs=nb_epoch,
                         verbose=1,
                         validation_data=ds_vali,
                         validation_steps=nb_validation_samples // batch_size,
                         initial_epoch=0,
                         callbacks=[cp_callback])
    return model, history

def main(image_path, datapath, modelname, rows, cols, batch_size, nb_epoch):
    hdf5_dir = Path(datapath)
    print('Reading the dataset')

    file_train = read_hdf5(hdf5_dir, 'train', rows, cols)

    model = create_model(modelname, rows, cols)
    model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Load the input image and preprocess it
    input_image = load_and_preprocess_image(image_path, rows, cols)

    class_labels = model.predict(np.expand_dims(input_image, axis=0))

    # Predict the class label for the input image
    class_names = ['focused', 'distracted']

    # Ensure the predicted class labels match the filtered class names
    num_classes = len(class_names)
    if class_labels.shape[1] != num_classes:
        print(f"Error: The model's output classes ({class_labels.shape[1]}) don't match the filtered class names ({num_classes}).")
        exit(1)
    # Get the class name with the highest probability
    class_index = np.argmax(class_labels[0])
    class_name = class_names[class_index]

    # Prepare the result as a JSON object
    result = {
        "image_path": image_path,
        "prediction": class_name
    }

    # Print the result as JSON string to stdout
    print(json.dumps(result))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Driver Monitoring Prediction')
    parser.add_argument('-p', '--image_path', required=True, help='Path to the uploaded image')
    parser.add_argument("-d", "--datapath", help="path of the dataset directory containing h5 files", required=True)
    parser.add_argument("-r", "--rows", help="row size, default 256", type=int, default=256)
    parser.add_argument("-c", "--cols", help="column size, default 256", type=int, default=256)
    parser.add_argument("-b", "--batch_size", help="batch size, default 64", type=int, default=64)
    parser.add_argument("-e", "--nb_epoch", help="number of epoch, default 10", type=int, default=10)
    parser.add_argument("-m", "--model", help="model name, choose between vgg, inception, resnet or efficientnet, default vgg",
                        choices=["VGG", "InceptionNet", "ResNet", "EfficientNet"], type=str, default='vgg')
    args = parser.parse_args()
    print(args)
    print('trained models will be found in saved_models/')
    main(args.image_path, args.datapath, args.model, args.rows, args.cols, args.batch_size, args.nb_epoch)

# python3 experiment.py -d DATAPATH/statefarm/ -r 256 -c 256 -b 64 -e 10 -m vgg




# python experiment.py -d ./OUTPUT_PATH -r 256 -c 256 -b 256 -e 10 -m VGG