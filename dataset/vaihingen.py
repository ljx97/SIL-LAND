import os
import random
import torch.utils.data as data
from torch import distributed, zeros_like, unique
import torchvision as tv
import numpy as np
from .utils import Subset, filter_images, group_images

from PIL import Image


classes = {
    'Clutter',
    'Building',
    'Trees',
    'Impervious_surfaces',
    'Low_Vegetation',
    'Cars'
}
# label_to_color = {
#             0: [0, 0, 255],
#             1:[0, 255, 0]
#             2: [255, 255, 255],
#             3: [0, 255, 255],
#             4: [255, 255, 0],
#             5: [0, 0, 0],
#     }


class VaihingenSegmentation(data.Dataset):
        def __init__(self, root, train=True, transform=None):

            # root = os.path.expanduser(root)
            # print('-------', root)
            root = '/workspace/SDR/data/'
            # root = root

            base_dir = "Vaihingen"
            vaihingen_root = os.path.join(root, base_dir)
            if train:
                split = 'training512'
            else:
                split = 'test_split_no_boundary'
            annotation_folder = os.path.join(vaihingen_root, 'labels',split)
            image_folder = os.path.join(vaihingen_root,'images',split)

            self.images = []
            fnames = sorted(os.listdir(image_folder))
            self.images = [(os.path.join(image_folder, x), os.path.join(annotation_folder, x[:-3] + "png")) for x in
                           fnames]

            self.transform = transform

        def __getitem__(self, index):
            """
            Args:
                index (int): Index
            Returns:
                tuple: (image, target) where target is the image segmentation.
            """
            img = Image.open(self.images[index][0]).convert('RGB')  # 第二维是0表示图片
            target = Image.open(self.images[index][1])  # 第二维是1 表示分割的真值

            if self.transform is not None:
                img, target = self.transform(img, target)

            return img, target

        def __len__(self):
            return len(self.images)

class VaihingenSegmentationIncremental(data.Dataset):
    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 labels=None,
                 labels_old=None,
                 idxs_path=None,
                 masking=True,
                 where_to_sim='GPU_windows',
                 overlap=True,
                 rank=0):

        full_data = VaihingenSegmentation(root, train)
        self.rank = rank
        self.labels = []
        self.labels_old = []
        self.where_to_sim = where_to_sim
        if labels is not None:
            # store the labels
            labels_old = labels_old if labels_old is not None else []

            self.__strip_zero(labels)
            self.__strip_zero(labels_old)

            assert not any(l in labels_old for l in labels), "labels and labels_old must be disjoint sets"
            self.labels = [0] + labels
            self.labels_old = [0] + labels_old
            # self.labels = labels
            # self.labels_old = labels_old

            self.order = [0] + labels_old + labels
            # self.order = labels_old + labels
            # take index of images with at least one class in labels and all classes in labels+labels_old+[255]
            if idxs_path is not None and os.path.exists(idxs_path):
                idxs = np.load(idxs_path).tolist()
            else:
                idxs = filter_images(full_data, labels, labels_old, overlap=overlap)
                if self.where_to_sim == 'GPU_windows' or self.where_to_sim == 'CPU_windows':
                    if idxs_path is not None and self.rank == 0:
                        np.save(idxs_path, np.array(idxs, dtype=int))
                else:
                    if idxs_path is not None and distributed.get_rank() == 0:
                        np.save(idxs_path, np.array(idxs, dtype=int))

            if train:
                masking_value = 0
            else:
                masking_value = 255
            # masking_value = 255
            self.inverted_order = {label: self.order.index(label) for label in self.order}

            self.inverted_order[255] = masking_value
            print('self.inverted_order:', self.inverted_order)
            reorder_transform = tv.transforms.Lambda(
                    lambda t: t.apply_(lambda x: self.inverted_order[x] if x in self.inverted_order else masking_value))

            if masking:

                if self.where_to_sim == 'GPU_windows' or self.where_to_sim == 'CPU_windows':
                    target_transform = self.tmp_funct3
                else:
                    tmp_labels = self.labels + [255]
                    target_transform = tv.transforms.Lambda(
                        lambda t: t.apply_(lambda x: self.inverted_order[x] if x in tmp_labels else masking_value))
            else:
                target_transform = reorder_transform

            # make the subset of the dataset
            self.dataset = Subset(full_data, idxs, transform, target_transform)
        else:
            self.dataset = full_data
    def tmp_funct1(self, x):
        tmp = zeros_like(x)
        for value in unique(x):
            if value in self.inverted_order:
                new_value = self.inverted_order[value.item()]
            else:
                new_value = self.inverted_order[255]  # i.e. masking value
            tmp[x == value] = new_value
        return tmp

    def tmp_funct3(self, x):
        tmp = zeros_like(x)
        for value in unique(x):
            if value in self.labels + [255]:
                new_value = self.inverted_order[value.item()]
            else:
                new_value = self.inverted_order[255]  # i.e. masking value
            tmp[x == value] = new_value
        return tmp

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """

        return self.dataset[index]

    @staticmethod
    def __strip_zero(labels):
        while 0 in labels:
            labels.remove(0)

    def __len__(self):
        return len(self.dataset)