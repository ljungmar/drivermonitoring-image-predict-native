import h5py
import cv2
from pathlib import Path
import glob
import os
import numpy as np
import argparse
import random

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

def load_data(datapath, subset='train', rows=256, cols=256):
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
    images, labels = load_data(datapath, subset, rows, cols)
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
