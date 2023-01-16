import evaluate_detection.transforms as T
# partly taken from https://github.com/pytorch/vision/blob/master/torchvision/datasets/voc.py
import functools
import torch

import os
import tarfile
import collections

from torchvision.datasets import VisionDataset
import xml.etree.ElementTree as ET
from PIL import Image
from torchvision.datasets.utils import download_url



CLASS_NAMES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
)

DATASET_YEAR_DICT = {
    '2012': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar',
        'filename': 'VOCtrainval_11-May-2012.tar',
        'md5': '6cd6e144f989b92b3379bac3b3de84fd',
        'base_dir': os.path.join('VOCdevkit', 'VOC2012')
    },
    '2011': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2011/VOCtrainval_25-May-2011.tar',
        'filename': 'VOCtrainval_25-May-2011.tar',
        'md5': '6c3384ef61512963050cb5d687e5bf1e',
        'base_dir': os.path.join('TrainVal', 'VOCdevkit', 'VOC2011')
    },
    '2010': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar',
        'filename': 'VOCtrainval_03-May-2010.tar',
        'md5': 'da459979d0c395079b5c75ee67908abb',
        'base_dir': os.path.join('VOCdevkit', 'VOC2010')
    },
    '2009': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2009/VOCtrainval_11-May-2009.tar',
        'filename': 'VOCtrainval_11-May-2009.tar',
        'md5': '59065e4b188729180974ef6572f6a212',
        'base_dir': os.path.join('VOCdevkit', 'VOC2009')
    },
    '2008': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2008/VOCtrainval_14-Jul-2008.tar',
        'filename': 'VOCtrainval_11-May-2012.tar',
        'md5': '2629fa636546599198acfcfbfcf1904a',
        'base_dir': os.path.join('VOCdevkit', 'VOC2008')
    },
    '2007': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar',
        'filename': 'VOCtrainval_06-Nov-2007.tar',
        'md5': 'c52e279531787c972589f7e41ab4ae64',
        'base_dir': os.path.join('VOCdevkit', 'VOC2007')
    },
    '2007-test': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar',
        'filename': 'VOCtest_06-Nov-2007.tar',
        'md5': 'b6e924de25625d8de591ea690078ad9f',
        'base_dir': os.path.join('VOCdevkit', 'VOC2007')
    }
}

def make_transforms(image_set, imgs_size=224, padding=1):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    rec_size = imgs_size // 2 - padding
    scales = [(rec_size, rec_size)]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomResize(scales),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize(scales),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')
class VOCDetection(VisionDataset):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Detection Dataset.
    Args:
        root (string): Root directory of the VOC Dataset.
        year (string, optional): The dataset year, supports years 2007 to 2012.
        image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
            (default: alphabetic indexing of VOC's 20 classes).
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, required): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(self,
                 root,
                 years='2012',
                 image_sets='train',
                 transform=None,
                 target_transform=None,
                 transforms=None,
                 no_cats=False,
                 keep_single_objs_only=1,
                 filter_by_mask_size=1):
        super(VOCDetection, self).__init__(root, transforms, transform, target_transform)
        self.images = []
        self.annotations = []
        self.imgids = []
        self.imgid2annotations = {}
        self.image_set = []

        self.CLASS_NAMES = CLASS_NAMES
        self.MAX_NUM_OBJECTS = 64
        self.no_cats = no_cats
        base_dir = os.path.dirname(os.path.abspath(__file__))
        # loading random per class support, randomly chosen from pascal 2012 train partition
        self.support_set = torch.load(os.path.join(base_dir, '2012_support_set.pth'))
        # load pascal 2012 val samples that have single object and occupy less than 20% of the image.
        self.val_flattened_set = torch.load(os.path.join(base_dir, '2012_val_flattened_set.pth'))


        for year, image_set in zip(years, image_sets):

            if year == "2007" and image_set == "test":
                year = "2007-test"
            valid_sets = ["train", "trainval", "val"]
            if year == "2007-test":
                valid_sets.append("test")

            base_dir = DATASET_YEAR_DICT[year]['base_dir']
            voc_root = os.path.join(self.root, base_dir)
            image_dir = os.path.join(voc_root, 'JPEGImages')
            annotation_dir = os.path.join(voc_root, 'Annotations')

            if not os.path.isdir(voc_root):
                raise RuntimeError('Dataset not found or corrupted.' +
                                   ' You can use download=True to download it')
            file_names = self.extract_fns(image_set, voc_root)
            self.image_set.extend(file_names)

            self.images.extend([os.path.join(image_dir, x + ".jpg") for x in file_names])
            self.annotations.extend([os.path.join(annotation_dir, x + ".xml") for x in file_names])

            self.imgids.extend(self.convert_image_id(x, to_integer=True) for x in file_names)
            self.imgid2annotations.update(dict(zip(self.imgids, self.annotations)))

        if keep_single_objs_only:
            single_indices = []
            for index in range(len(self.imgids)):
                target, instances = self.load_instances(self.imgids[index])
                if len(instances) == 1:
                    single_indices.append(index)
            self.images = [self.images[i] for i in range(len(self.images)) if i in single_indices]
            self.annotations = [self.annotations[i] for i in range(len(self.annotations)) if i in single_indices]
            self.imgids = [self.imgids[i] for i in range(len(self.imgids)) if i in single_indices]

        if filter_by_mask_size:
            valid_mask_size_indices = []
            for index in range(len(self.imgids)):
                target, instances = self.load_instances(self.imgids[index])
                s = target['annotation']['size']
                image_area = int(s['width'])*int(s['height'])
                instance_area = instances[0]['area']
                frac = instance_area / image_area
                if frac < 0.2:
                    valid_mask_size_indices.append(index)
            self.images = [self.images[i] for i in range(len(self.images)) if i in valid_mask_size_indices]
            self.annotations = [self.annotations[i] for i in range(len(self.annotations)) if i in valid_mask_size_indices]
            self.imgids = [self.imgids[i] for i in range(len(self.imgids)) if i in valid_mask_size_indices]



        assert (len(self.images) == len(self.annotations) == len(self.imgids))

    @staticmethod
    def convert_image_id(img_id, to_integer=False, to_string=False, prefix='2021'):
        if to_integer:
            return int(prefix + img_id.replace('_', ''))
        if to_string:
            x = str(img_id)
            assert x.startswith(prefix)
            x = x[len(prefix):]
            if len(x) == 6:
                return x
            return x[:4] + '_' + x[4:]

    @functools.lru_cache(maxsize=None)
    def load_instances(self, img_id):
        tree = ET.parse(self.imgid2annotations[img_id])
        target = self.parse_voc_xml(tree.getroot())

        image_id = target['annotation']['filename']
        instances = []
        for obj in target['annotation']['object']:
            cls = obj["name"]
            # We include "difficult" samples in training.
            # Based on limited experiments, they don't hurt accuracy.
            difficult = int(obj["difficult"])
            # if difficult == 1:
            # continue
            bbox = obj["bndbox"]
            bbox = [float(bbox[x]) for x in ["xmin", "ymin", "xmax", "ymax"]]
            # Original annotations are integers in the range [1, W or H]
            # Assuming they mean 1-based pixel indices (inclusive),
            # a box with annotation (xmin=1, xmax=W) covers the whole image.
            # In coordinate space this is represented by (xmin=0, xmax=W)
            bbox[0] -= 1.0
            bbox[1] -= 1.0
            instance = dict(
                category_id=1 if self.no_cats else CLASS_NAMES.index(cls),
                bbox=bbox,
                area=(bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
                difficult=difficult,
                image_id=img_id
            )
            instances.append(instance)

        assert len(instances) <= self.MAX_NUM_OBJECTS
        return target, instances

    def extract_fns(self, image_set, voc_root):
        splits_dir = os.path.join(voc_root, 'ImageSets/Main')
        split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')
        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]
        return file_names

    def __getitem__(self, idx):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a dictionary of the XML tree.
        """
        index, label = self.val_flattened_set[idx]
        img = Image.open(self.images[index]).convert('RGB')
        target, instances = self.load_instances(self.imgids[index])
        # keep instance with a same label
        w, h = map(target['annotation']['size'].get, ['width', 'height'])
        target = dict(
            image_id=torch.tensor([self.imgids[index]], dtype=torch.int64),
            labels=torch.tensor([i['category_id'] for i in instances], dtype=torch.int64),
            area=torch.tensor([i['area'] for i in instances], dtype=torch.float32),
            boxes=torch.as_tensor([i['bbox'] for i in instances], dtype=torch.float32),
            orig_size=torch.as_tensor([int(h), int(w)]),
            size=torch.as_tensor([int(h), int(w)]),
            iscrowd=torch.zeros(len(instances), dtype=torch.uint8)
        )

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgids)

    def parse_voc_xml(self, node):
        voc_dict = {}
        children = list(node)
        if children:
            def_dic = collections.defaultdict(list)
            for dc in map(self.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            if node.tag == 'annotation':
                def_dic['object'] = [def_dic['object']]
            voc_dict = {
                node.tag:
                    {ind: v[0] if len(v) == 1 else v
                     for ind, v in def_dic.items()}
            }
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict


def download_extract(url, root, filename, md5):
    download_url(url, root, filename, md5)
    with tarfile.open(os.path.join(root, filename), "r") as tar:
        tar.extractall(path=root)


