import h5py
import cv2
from pathlib import Path
import glob
import os
import numpy as np
import argparse
import random
from sklearn.model_selection import train_test_split

paths_train = ""
labels_test = 0

def dir_path(string):
    try:
        os.path.isdir(string)
        return string
    except OSError:
        raise (string, 'not a directory')

def load_image(path, rows=256, cols=256):
    img = cv2.imread(path)
    img = cv2.resize(img, (rows, cols))
    return img

def load_data(datapath, subset='train', rows=256, cols=256, test_size=0.25, random_state=42):
    focused_paths = glob.glob(os.path.join(datapath, subset, "c0", "*.jpg"))
    distracted_paths = (
        glob.glob(os.path.join(datapath, subset, "c1", "*.jpg"))
        + glob.glob(os.path.join(datapath, subset, "c2", "*.jpg"))
        + glob.glob(os.path.join(datapath, subset, "c7", "*.jpg"))
    )

    focused_labels = [0] * len(focused_paths)
    distracted_labels = [1] * len(distracted_paths)

    paths = focused_paths + distracted_paths
    labels = focused_labels + distracted_labels

     # Use train_test_split to split the data into training and testing sets
    paths_train, paths_test, labels_train, labels_test = train_test_split(paths, labels, test_size=test_size, random_state=random_state)
    paths_train = paths_train * 2
    labels_train = labels_train * 2

    paths_test = paths_test * 2
    labels_test = labels_test * 2

    print("Number of labels in train set:", len(labels_train))
    print("Number of labels in test set:", len(labels_test))

    if subset == 'train':
        paths = paths_train
        labels = labels_train
    elif subset == 'test':
        paths = paths_test
        labels = labels_test
    else:
        raise ValueError("Invalid subset value. Use 'train' or 'test'.")

    c = list(zip(paths, labels))
    random.shuffle(c)
    paths, labels = zip(*c)

    images = [load_image(x, rows, cols) for x in paths]
    return images, labels

def store_hdf5(images, labels, out_dir=Path('./'), subset='train', rows=256, cols=256):
    file = h5py.File(out_dir / f"driver_distraction_{rows}x{cols}_{subset}.h5", "w")
    dataset = file.create_dataset(
        "images", np.shape(images), np.uint8, data=images, compression="gzip", chunks=True
    )
    meta_set = file.create_dataset(
        "meta", np.shape(labels), np.uint8, data=labels
    )
    file.close()

def main(datapath, output, subset, rows, cols):
    hdf5_dir = Path(output)
    hdf5_dir.mkdir(parents=True, exist_ok=True)
    print('Reading the dataset')
    images, labels = load_data(datapath, subset, rows, cols, 0.2, 42)
    print('Writing the dataset')
    store_hdf5(images, labels, hdf5_dir, subset, rows, cols)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Converting the whole dataset into one h5 file.')
    parser.add_argument("-d", "--datapath", help="path of the dataset", type=dir_path, required=True)
    parser.add_argument("-o", "--destpath", help="path where to write the output file", required=True)
    parser.add_argument("-r", "--rows", help="row size, default 256", type=int, default=256)
    parser.add_argument("-c", "--cols", help="column size, default 256", type=int, default=256)
    parser.add_argument("-s", "--set", help="data subset could be train or test", type=str, default='train')
    args = parser.parse_args()
    print(args)
    main(args.datapath, args.destpath, args.set, args.rows, args.cols)
