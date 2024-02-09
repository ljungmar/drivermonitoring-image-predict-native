from convert_data import paths_train, labels_test
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras import backend

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
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization
import time

tf.config.run_functions_eagerly(True)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


def load_and_preprocess_image(image_path, rows, cols, min_size=75):
    # Load the image using PIL
    img = Image.open(image_path)

    print("Original Image Shape:", img.size)

    # Resize the image to at least 75x75 pixels
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

    if 'images' in file and 'meta' in file:
        images = np.array(file["images"]).astype(float)
        labels = np.array(file["meta"]).astype(int)

    return images, labels

def batch_image_generator(images, labels, batch_size=32, data_augmentation=None):
    y_train = tf.cast(labels, dtype=tf.int32)  # Ensure labels are integers
    y_train = to_categorical(y_train, num_classes=2)
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

def training_with_dataaugmentation(images, labels, model, nb_epoch, batch_size, rows, cols, train_datagen=None):
    if train_datagen is None:
        train_datagen = ImageDataGenerator(
            zoom_range=0.2,
            brightness_range=[0.5, 1.5],
            rotation_range=40,
            horizontal_flip=True,
            height_shift_range=0.2,
            width_shift_range=0.2,
            rescale=1.0/255,
            fill_mode='nearest',
            shear_range=0.2,
        )
    val_datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)

    pivot = int(len(images) * 0.8)  # Assuming 80% training and 20% validation split

    images_train, images_val, labels_train, labels_val = train_test_split(
        images, labels, test_size=0.2, random_state=42)

    ds_train = tf.data.Dataset.from_generator(
        lambda: batch_image_generator(images_train, labels_train, batch_size, train_datagen),
        output_types=(tf.float32, tf.float32),
        output_shapes=([batch_size, rows, cols, 3], [batch_size, 2])
    )

    ds_vali = tf.data.Dataset.from_generator(
        lambda: batch_image_generator(images_val, labels_val, batch_size, val_datagen),
        output_types=(tf.float32, tf.float32),
        output_shapes=([batch_size, rows, cols, 3], [batch_size, 2])
    )

    nb_train_samples = len(labels_train)
    nb_validation_samples = len(labels_val)
    log_dir = "/content/drive/MyDrive/logs/fit_ft/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint_path = "/content/drive/MyDrive/logs/cpft-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=1,
        save_weights_only=True,
        save_freq='epoch',
        period=5
    )

    history = model.fit(
        ds_train,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=nb_epoch,
        verbose=1,
        validation_data=ds_vali,
        validation_steps=nb_validation_samples // batch_size,
        initial_epoch=0,
        callbacks=[cp_callback]
    )

    # Print out the accuracy as floats
    print("Training Accuracy:", history.history['accuracy'][-1])
    print("Validation Accuracy:", history.history['val_accuracy'][-1])


    return model, history

def create_cnn(rows, cols):
    """
    Simple CNN architecture for your project
    """
    nb_classes = 2

    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(rows, cols, 3)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(nb_classes, activation='softmax'))

    return model

def create_model(name, rows, cols):
    nb_classes = 2

    if 'cnn' in name.lower():  # Add this line for the new CNN model
        return create_cnn(rows, cols)
    else:
        raise AttributeError(f'The model {name} is not available. Only CNN is available.')

def evaluate_model(model, ds_test, test_labels, batch_size, images_test):


    if ds_test is None or len(ds_test) == 0:
        print("Error: Test dataset is empty.")
        return

    test_labels_categorical = to_categorical(test_labels, num_classes=2)

    # Print batch shapes for debugging
    for x, y in ds_test.take(1):
        print("Batch Shapes - X:", x.shape, "Y:", y.shape)

    #test_loss, test_accuracy = model.evaluate(ds_test, steps=len(test_labels) // batch_size)
    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(ds_test)
    print("Test Loss:", test_loss)
    print("Test Accuracy:", test_accuracy)




    #print("Test Loss:", test_loss)
    #print("Test Accuracy:", test_accuracy)

def main(image_path, datapath, modelname, rows, cols, batch_size, nb_epoch):
    start_time = time.time()
    hdf5_dir = Path(datapath)
    print('Reading the dataset')

    images_train, labels_train = read_hdf5(hdf5_dir, 'train', rows, cols)
    images_test, labels_test = read_hdf5(hdf5_dir, 'test', rows, cols)

    model = create_model(modelname, rows, cols)
    print("model created")
    model.summary()
    print("summary completed")
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'], run_eagerly=True)
    print("Input shape:", images_train.shape)
    print("Labels shape:", labels_train.shape)

    # Call the training function
    trained_model, history = training_with_dataaugmentation(images_train, labels_train, model, nb_epoch, batch_size, rows, cols)

    # Evaluate the model on the test set
    # ds_test = tf.data.Dataset.from_generator(
    #     lambda: batch_image_generator(images_test, labels_test, batch_size),
    #     output_types=(tf.float32, tf.float32),
    #     output_shapes=([batch_size, rows, cols, 3], [batch_size, 2])
    # )

    # Check the shape of the test data before calling predict
    # for test_data, _ in ds_test.take(1):
    #     print("Shape of the original test data:", test_data.shape)

    # Move the axis from index 3 to index 1
    # test_data_moved = np.moveaxis(test_data, 3, 1)

    # Print the shape of the modified test data
    # print("Shape of the modified test data:", test_data_moved.shape)

    # Call the evaluate_model function with the modified test data
    # evaluate_model(trained_model, ds_test, labels_test, batch_size, images_test)

    # Load the input image and preprocess it
    #input_image = load_and_preprocess_image(image_path, rows, cols)

    #class_labels = trained_model.predict(ds_test, steps=len(labels_test) // batch_size)
     # Load the input image and preprocess it
    input_image = load_and_preprocess_image(image_path, rows, cols)
    class_labels = trained_model.predict(np.expand_dims(input_image, axis=0))

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

    end_time = time.time()  # Record the end time
    elapsed_time = end_time - start_time
    minutes, seconds = divmod(elapsed_time, 60)

    print(f"Elapsed time: {int(minutes)} minutes and {seconds:.2f} seconds")

    # Print the result as a JSON string to stdout
    json_result = json.dumps(result)
    print(json_result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Driver Monitoring Prediction')
    parser.add_argument('-p', '--image_path', required=True, help='Path to the uploaded image')
    parser.add_argument("-d", "--datapath", help="path of the dataset directory containing h5 files", required=True)
    parser.add_argument("-r", "--rows", help="row size, default 256", type=int, default=256)
    parser.add_argument("-c", "--cols", help="column size, default 256", type=int, default=256)
    parser.add_argument("-b", "--batch_size", help="batch size, default 64", type=int, default=64)
    parser.add_argument("-e", "--nb_epoch", help="number of epoch, default 10", type=int, default=10)
    parser.add_argument("-m", "--model", help="model name, choose cnn", choices=["cnn"], type=str, default='cnn')
    args = parser.parse_args()
    print(args)
    print('trained models will be found in saved_models/')
    tf.config.run_functions_eagerly(True)
    main(args.image_path, args.datapath, args.model, args.rows, args.cols, args.batch_size, args.nb_epoch)
