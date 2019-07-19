import os

import requests
import numpy as np
from torch.utils.data import Dataset


class KMNIST(Dataset):
    """
    Dataset class for use with pytorch for the Kuzushiji-MNIST dataset as given in
    Deep Learning for Classical Japanese Literature. Tarin Clanuwat et al. arXiv:1812.01718

    Kuzushiji-MNIST contains 70,000 28x28 grayscale images spanning 10 classes (one from each column of hiragana),
    and is perfectly balanced like the original MNIST dataset (6k/1k train/test for each class).
    """

    def __init__(self, data_dir='.', train: bool = True, transform=None):
        """
        :param data_dir: Directory of the data
        :param train: Use training or test set
        :param transform: pytorch transforms for data augmentation
        """

        self.__urls = [
            'http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-train-imgs.npz',
            'http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-train-labels.npz',
            'http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-test-imgs.npz',
            'http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-test-labels.npz',
        ]

        t_str = 'train' if train else 'test'
        imgs_fn = 'kmnist-{}-imgs.npz'.format(t_str)
        labels_fn = 'kmnist-{}-labels.npz'.format(t_str)

        if not os.path.exists(data_dir):
            os.mkdir(os.path.abspath(data_dir))
        if not os.path.exists(os.path.abspath(os.path.join(data_dir, imgs_fn))):
            self.__download(os.path.abspath(data_dir))

        imgs_fn = os.path.abspath(os.path.join(data_dir, imgs_fn))
        labels_fn = os.path.abspath(os.path.join(data_dir, labels_fn))

        self.images = np.load(imgs_fn)['arr_0']
        self.labels = np.load(labels_fn)['arr_0']
        self.n_classes = len(np.unique(self.labels))
        self.class_labels, self.class_frequency = np.unique(self.labels, return_counts=True)
        self.class_frequency = self.class_frequency / np.sum(self.class_frequency)
        self.data_dir = data_dir
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1  # only gray scale
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = np.expand_dims(self.images[idx], axis=-1)
        if self.transform:
            image = self.transform(image)

        label = np.int(self.labels[idx])
        return image, label

    def __download(self, data_dir):
        print('Datadir', data_dir)
        for url in self.__urls:
            fn = os.path.basename(url)
            req = requests.get(url, stream=True)
            print('Downloading {}'.format(fn))
            with open(os.path.join(data_dir, fn), 'wb') as fh:
                for chunck in req.iter_content(chunk_size=1024):
                    if chunck:
                        fh.write(chunck)
            print('done')
        print('All files downloaded')


class K49(Dataset):
    """
    Dataset class for use with pytorch for the Kuzushiji-49 dataset as given in
    Deep Learning for Classical Japanese Literature. Tarin Clanuwat et al. arXiv:1812.01718

    Kuzushiji-49 contains 270,912 images spanning 49 classes, and is an extension of the Kuzushiji-MNIST dataset.
    """

    def __init__(self, data_dir='.', train: bool = True, transform=None):
        """
        :param data_dir: Directory of the data
        :param train: Use training or test set
        :param transform: pytorch transforms for data augmentation
        """

        self.__urls = [
            'http://codh.rois.ac.jp/kmnist/dataset/k49/k49-train-imgs.npz',
            'http://codh.rois.ac.jp/kmnist/dataset/k49/k49-train-labels.npz',
            'http://codh.rois.ac.jp/kmnist/dataset/k49/k49-test-imgs.npz',
            'http://codh.rois.ac.jp/kmnist/dataset/k49/k49-test-labels.npz',
        ]

        t_str = 'train' if train else 'test'
        imgs_fn = 'k49-{}-imgs.npz'.format(t_str)
        labels_fn = 'k49-{}-labels.npz'.format(t_str)

        if not os.path.exists(data_dir):
            os.mkdir(os.path.abspath(data_dir))
        if not os.path.exists(os.path.abspath(os.path.join(data_dir, imgs_fn))):
            self.__download(os.path.abspath(data_dir))

        imgs_fn = os.path.abspath(os.path.join(data_dir, imgs_fn))
        labels_fn = os.path.abspath(os.path.join(data_dir, labels_fn))

        self.images = np.load(imgs_fn)['arr_0']
        self.labels = np.load(labels_fn)['arr_0']
        self.n_classes = len(np.unique(self.labels))
        self.class_labels, self.class_frequency = np.unique(self.labels, return_counts=True)
        self.class_frequency = self.class_frequency / np.sum(self.class_frequency)
        self.data_dir = data_dir
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1  # only gray scale
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = np.expand_dims(self.images[idx], axis=-1)
        if self.transform:
            image = self.transform(image)

        label = np.int(self.labels[idx])
        return image, label

    def __download(self, data_dir):
        print('Datadir', data_dir)
        for url in self.__urls:
            fn = os.path.basename(url)
            req = requests.get(url, stream=True)
            print('Downloading {}'.format(fn))
            with open(os.path.join(data_dir, fn), 'wb') as fh:
                for chunck in req.iter_content(chunk_size=1024):
                    if chunck:
                        fh.write(chunck)
            print('done')
        print('All files downloaded')
