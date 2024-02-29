import traceback
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
from keras.layers import Dropout
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


def load_and_preprocess_image(image, output_path, rows, cols, min_size=75):
    if isinstance(image, str):
        # Load the image using PIL
        img = Image.open(image)

        print("Original Image Shape:", img.size)

        # Resize the image to at least 75x75 pixels
        img = img.resize((max(min_size, rows), max(min_size, cols)))

        # Crop the image to the specified size if necessary
        img = img.crop((0, 0, rows, cols))

        # Save the preprocessed image
        img.save(output_path)

        # Convert the image to a NumPy array
        img_array = np.array(img)
    elif isinstance(image, np.ndarray):
        img_array = image
    else:
        raise ValueError("Unsupported image type. Only file paths or NumPy arrays are supported.")

    # Preprocess the image according to the requirements of the chosen model
    preprocessed_img = preprocess_input(img_array)

    # Return the preprocessed image
    print("Original Image Shape:", img_array.shape)
    print("Processed Image Shape:", preprocessed_img.shape)

    return preprocessed_img



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

def batch_image_generator(images, labels, batch_size, rows, cols, data_augmentation=None, epochs=1, saved_directory=None):
    sample_count = len(labels)
    num_classes = 2

    train = True

    if train:
        epoch_range = sample_count // batch_size
    else:
        epoch_range = (sample_count + batch_size - 1) // batch_size  # Handle the last batch

    for epoch in range(epochs):
        for index in range(epoch_range):
            start = index * batch_size
            end = min((index + 1) * batch_size, sample_count)  # Adjust end to handle the last batch
            x = images[start: end]
            y = labels[start: end]

            if saved_directory:
                # Load images from the saved directory
                loaded_images = []
                for image_index in range(start, end):
                    image_path = os.path.join(saved_directory, f"image_{image_index}_preprocessed.jpg")
                    img = Image.open(image_path)
                    img_array = np.array(img)
                    preprocessed_img = preprocess_input(img_array)
                    loaded_images.append(preprocessed_img)
                x = np.array(loaded_images)

            if train and data_augmentation is not None:
                generator = data_augmentation.flow(x, y, batch_size=batch_size)

                for i in range(len(generator)):
                    x_augmented, y_augmented = next(generator)

                    # Check the shape of y_augmented
                    print("THEEEEEEEEEEEEEEEEEEEEEEEEE SHAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAPE:", y_augmented.shape)

                    # Check the dimensionality of y_augmented
                    if len(y_augmented.shape) == 1:
                        # If y_augmented is 1D, assume it's not one-hot encoded and convert it
                        y_augmented_one_hot = to_categorical(y_augmented, num_classes=num_classes)
                    else:
                        # If y_augmented is already one-hot encoded, use it directly
                        y_augmented_one_hot = y_augmented

                    # Ensure the shape is correct
                    assert x_augmented.shape[1:] == (rows, cols, 3), f"Expected 1 image with shape {(rows, cols, 3)}, but got {x_augmented.shape}"

                    yield x_augmented, y_augmented_one_hot
            else:
                # Convert labels to one-hot encoding if not already
                y_one_hot = to_categorical(y, num_classes=num_classes)

                # Ensure the shape is correct
                assert x.shape == (end - start, rows, cols, 3), f"Expected {end - start} images with shape {(rows, cols, 3)}, but got {x.shape}"
                assert y_one_hot.shape == (end - start, num_classes), f"Expected {end - start} labels with shape ({num_classes},), but got {y_one_hot.shape}"

                yield x, y_one_hot


def training_with_dataaugmentation(images_train, labels_train, images_test, labels_test, model, nb_epoch, batch_size, rows, cols, train_datagen=None):
    if train_datagen is None:
        train_datagen = ImageDataGenerator(
            zoom_range=0.2,
            brightness_range=[0.5, 1.5],
            rotation_range=40,
            horizontal_flip=True,
            height_shift_range=0.2,
            width_shift_range=0.2,
            rescale=1.0 / 255,
            fill_mode='nearest',
            shear_range=0.2,
        )
    test_datagen = None
    if test_datagen is None:
        test_datagen = ImageDataGenerator(
            zoom_range=0.1,
            brightness_range=[0.8, 1.2],
            rotation_range=20,
            horizontal_flip=True,
            height_shift_range=0.1,
            width_shift_range=0.1,
            rescale=1.0/255,
            fill_mode='nearest',
            shear_range=0.1,
        )

    pivot_train = int(len(images_train) * 0.8)  # Assuming 80% training and 20% validation split

    images_train, images_val = images_train[:pivot_train], images_train[pivot_train:]
    labels_train, labels_val = labels_train[:pivot_train], labels_train[pivot_train:]

    # Print dataset sizes and steps per epoch
    print("Training Dataset Size:", len(labels_train))
    print("Validation Dataset Size:", len(labels_val))
    print("Test Dataset Size:", len(labels_test))

    print("Steps per Epoch (Training):", len(labels_train) // batch_size)
    print("Steps per Epoch (Validation):", len(labels_val) // batch_size)

    # Convert labels to categorical
    y_train = to_categorical(labels_train, num_classes=2)
    y_val = to_categorical(labels_val, num_classes=2)
    y_test = to_categorical(labels_test, num_classes=2)

    ds_train = tf.data.Dataset.from_generator(
        lambda: batch_image_generator(images_train, y_train, batch_size, rows, cols, data_augmentation=train_datagen),
        output_types=(tf.float32, tf.float32),
        output_shapes=([batch_size, rows, cols, 3], [batch_size, 2])
    )
    ds_train = ds_train.repeat()

    ds_vali = tf.data.Dataset.from_generator(
        lambda: batch_image_generator(images_val, y_val, batch_size, rows, cols, data_augmentation=train_datagen),
        output_types=(tf.float32, tf.float32),
        output_shapes=([batch_size, rows, cols, 3], [batch_size, 2])
    )

    print("images_test sizeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee:", images_test.size)
    print("y_test: ", y_test.size)

    ds_test = tf.data.Dataset.from_generator(
        lambda: batch_image_generator(images_test, y_test, batch_size, rows, cols, data_augmentation=test_datagen),
        output_types=(tf.float32, tf.float32),
        output_shapes=([batch_size, rows, cols, 3], [batch_size, 2])
    )
    ds_test = ds_test.repeat()
    """ ds_test = tf.data.Dataset.from_tensor_slices((images_test, y_test))
    ds_test = ds_test.map(lambda x, y: (tf.debugging.check_numerics(x, "Input contains NaN"), y))
    ds_test = ds_test.batch(batch_size) """

    nb_train_samples = len(labels_train)
    nb_validation_samples = len(labels_val)
    nb_test_samples = len(labels_test)
    print("nb_train_sample:", nb_train_samples)
    print("nb_test_sample heeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeere:", nb_test_samples)

    log_dir = "/content/drive/MyDrive/logs/fit_ft/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint_path = "/content/drive/MyDrive/logs/cpft-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=1,
        save_weights_only=True,
        save_freq='epoch',
        validation_freq=5  # Use validation_freq instead of period
    )

    if batch_size <= 0:
        print("Error: batch_size should be a positive non-zero value.")
        return model, None

    try:
        history = model.fit(
            ds_train,
            steps_per_epoch=nb_train_samples // batch_size,
            epochs=nb_epoch,
            verbose=1,
            validation_data=ds_vali,
            validation_steps=nb_validation_samples // batch_size,
            callbacks=[cp_callback]
        )
    except Exception as e:
        print(f"An error occurred during training: {e}")
        return model, None

    try:
        test_loss, test_accuracy = model.evaluate(ds_test, steps=nb_test_samples // batch_size)
        print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')
    except Exception as e:
        print(f'Error during evaluation: {e}')
        test_loss = 0
        test_accuracy = 0

    # Print out the accuracy and loss as floats
    print("Training Accuracy:", history.history['accuracy'][-1])
    print("Training Loss:", history.history['loss'][-1])
    print("Validation Accuracy:", history.history['val_accuracy'][-1])
    print("Test Accuracy:", test_accuracy)
    print("Test Loss:", test_loss)

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
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

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

    # Print information about the loaded datasets
    print("Labels train shape:", labels_train.shape)
    print("Labels test shape:", labels_test.shape)

    model = create_model(modelname, rows, cols)
    print("model created")
    model.summary()
    print("summary completed")


    custom_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=custom_optimizer, loss='categorical_crossentropy', metrics=['accuracy'], run_eagerly=True)




    print("Input shape:", images_train.shape)
    print("Labels shape:", labels_train.shape)
    print("Labels test shape:", labels_test.shape)

    # Save the preprocessed test images
    output_dir = Path("path/to/your/preprocessed/test/")
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, image in enumerate(images_test):
        output_path = output_dir / f"image_{i}_preprocessed.jpg"
        load_and_preprocess_image(image, output_path, rows, cols)

    # Call the training function
    trained_model, history = training_with_dataaugmentation(images_train, labels_train, images_test, labels_test, model, nb_epoch, batch_size, rows, cols)
    trained_model.add(Dropout(0.5))

    try:
        trained_model.save('./saved_models/trained_model_64x64.h5')
    except Exception as e:
        print("Something somewhere. Went Wrong:")
        print(traceback.format.exec())

    # Load the input image and preprocess it
    input_image = load_and_preprocess_image(args.image_path, "output_image.jpg", args.rows, args.cols)
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
