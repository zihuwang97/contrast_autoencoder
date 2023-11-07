import os
import random
import json
from PIL import Image
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import random_split, ConcatDataset, Subset
from torch.utils.data import Dataset

from transforms import MultiView, RandomResizedCrop, ColorJitter, GaussianBlur, RandomRotation
from torchvision import transforms as T
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torchvision.datasets import STL10, CIFAR10, CIFAR100, ImageFolder, OxfordIIITPet, Caltech101, Flowers102, StanfordCars, INaturalist

import kornia.augmentation as K
import pandas as pd


class Cub2011(Dataset):
    base_folder = 'CUB_200_2011/images'
    url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, train=True, transform=None, loader=default_loader, download=True):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = default_loader
        self.train = train

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)
        return img, target

class ImageList(torch.utils.data.Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.samples)

class ImageNet100(ImageFolder):
    def __init__(self, root, split, transform):
        with open('data_slice/imagenet100.txt') as f:
            classes = [line.strip() for line in f]
            class_to_idx = { cls: idx for idx, cls in enumerate(classes) }

        super().__init__(os.path.join(root, split), transform=transform)
        samples = []
        for path, label in self.samples:
            cls = self.classes[label]
            if cls not in class_to_idx:
                continue
            label = class_to_idx[cls]
            samples.append((path, label))

        self.samples = samples
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.targets = [s[1] for s in samples]

class Pets(ImageList):
    def __init__(self, root, split, transform=None):
        with open(os.path.join(root, 'annotations', f'{split}.txt')) as f:
            annotations = [line.split() for line in f]

        samples = []
        for sample in annotations:
            path = os.path.join(root, 'images', sample[0] + '.jpg')
            label = int(sample[1])-1
            samples.append((path, label))

        super().__init__(samples, transform)

class Food101(ImageList):
    def __init__(self, root, split, transform=None):
        with open(os.path.join(root, 'meta', 'classes.txt')) as f:
            classes = [line.strip() for line in f]
        with open(os.path.join(root, 'meta', f'{split}.json')) as f:
            annotations = json.load(f)

        samples = []
        for i, cls in enumerate(classes):
            for path in annotations[cls]:
                samples.append((os.path.join(root, 'images', f'{path}.jpg'), i))

        super().__init__(samples, transform)

class DTD(ImageList):
    def __init__(self, root, split, transform=None):
        with open(os.path.join(root, 'labels', f'{split}1.txt')) as f:
            paths = [line.strip() for line in f]

        classes = sorted(os.listdir(os.path.join(root, 'images')))
        samples = [(os.path.join(root, 'images', path), classes.index(path.split('/')[0])) for path in paths]
        super().__init__(samples, transform)

class SUN397(ImageList):
    def __init__(self, root, split, transform=None):
        with open(os.path.join(root, 'ClassName.txt')) as f:
            classes = [line.strip() for line in f]

        with open(os.path.join(root, f'{split}_01.txt')) as f:
            samples = []
            for line in f:
                path = line.strip()
                for y, cls in enumerate(classes):
                    if path.startswith(cls+'/'):
                        samples.append((os.path.join(root, 'SUN397', path[1:]), y))
                        break
        super().__init__(samples, transform)

def load_pretrain_datasets(dataset='cifar10',
                           datadir='/data',
                           color_aug='default'):

    if dataset == 'imagenet100':
        mean = torch.tensor([0.485, 0.456, 0.406])
        std  = torch.tensor([0.229, 0.224, 0.225])
        
        train_transform = MultiView(T.Compose([T.RandomResizedCrop(224, scale=(0.2, 1.0)),
                                                T.RandomHorizontalFlip(),
                                                T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                                                T.RandomGrayscale(p=0.2),
                                                T.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                                                T.ToTensor(),
                                                T.Normalize(mean, std)]))
        test_transform = T.Compose([T.Resize(224),
                                    T.CenterCrop(224),
                                    T.ToTensor(),
                                    T.Normalize(mean, std)])

        trainset = ImageNet100(datadir, split='train', transform=train_transform)
        valset   = ImageNet100(datadir, split='train', transform=test_transform)
        testset  = ImageNet100(datadir, split='val', transform=test_transform)

    elif dataset == 'stl10':
        mean = torch.tensor([0.43, 0.42, 0.39])
        std  = torch.tensor([0.27, 0.26, 0.27])

        if color_aug == 'default':
            s = 1
        elif color_aug == 'strong':
            s = 2.
        elif color_aug == 'weak':
            s = 0.5

        train_transform = MultiView(T.Compose([T.RandomResizedCrop(96, scale=(0.2, 1.0)),
                                                T.RandomHorizontalFlip(),
                                                T.RandomApply([T.ColorJitter(0.4*s, 0.4*s, 0.4*s, 0.1*s)], p=0.8),
                                                T.RandomGrayscale(p=0.2*s),
                                                T.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                                                # GaussianBlur(9, (0.1, 2.0)),
                                                T.ToTensor(),
                                                T.Normalize(mean, std)]))
        test_transform = T.Compose([T.Resize(96),
                                    T.CenterCrop(96),
                                    T.ToTensor(),
                                    T.Normalize(mean, std)])
        
        trainset = STL10(datadir, split='train+unlabeled', transform=train_transform)
        valset   = STL10(datadir, split='train',           transform=test_transform)
        testset  = STL10(datadir, split='test',            transform=test_transform)

    elif dataset == 'stl10_rot':
        mean = torch.tensor([0.43, 0.42, 0.39])
        std  = torch.tensor([0.27, 0.26, 0.27])
        train_transform = MultiView(RandomResizedCrop(96, scale=(0.2, 1.0)))
        test_transform = T.Compose([T.Resize(96),
                                    T.CenterCrop(96),
                                    T.ToTensor(),
                                    T.Normalize(mean, std)])
        t1 = nn.Sequential(K.RandomHorizontalFlip(),
                           ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
                           K.RandomGrayscale(p=0.2),
                           GaussianBlur(9, (0.1, 2.0)),
                           RandomRotation(p=0.5),
                           K.Normalize(mean, std))
        t2 = nn.Sequential(K.RandomHorizontalFlip(),
                           ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
                           K.RandomGrayscale(p=0.2),
                           GaussianBlur(9, (0.1, 2.0)),
                           RandomRotation(p=0.5),
                           K.Normalize(mean, std))

        trainset = STL10(datadir, split='train+unlabeled', transform=train_transform)
        valset   = STL10(datadir, split='train',           transform=test_transform)
        testset  = STL10(datadir, split='test',            transform=test_transform)

    elif dataset == 'stl10_sol':
        mean = torch.tensor([0.43, 0.42, 0.39])
        std  = torch.tensor([0.27, 0.26, 0.27])
        train_transform = MultiView(RandomResizedCrop(96, scale=(0.2, 1.0)))

        test_transform = T.Compose([T.Resize(96),
                                    T.CenterCrop(96),
                                    T.ToTensor(),
                                    T.Normalize(mean, std)])
        t1 = nn.Sequential(K.RandomHorizontalFlip(),
                           ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
                           K.RandomSolarize(0.5, 0.0, p=0.5),
                           K.RandomGrayscale(p=0.2),
                           GaussianBlur(9, (0.1, 2.0)),
                           K.Normalize(mean, std))
        t2 = nn.Sequential(K.RandomHorizontalFlip(),
                           ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
                           K.RandomSolarize(0.5, 0.0, p=0.5),
                           K.RandomGrayscale(p=0.2),
                           GaussianBlur(9, (0.1, 2.0)),
                           K.Normalize(mean, std))

        trainset = STL10(datadir, split='train+unlabeled', transform=train_transform)
        valset   = STL10(datadir, split='train',           transform=test_transform)
        testset  = STL10(datadir, split='test',            transform=test_transform)

    else:
        raise Exception(f'Unknown dataset {dataset}')

    return dict(train=trainset,
                val=valset,
                test=testset,)

def load_datasets(dataset='cifar10',
                  datadir='/data',
                  pretrain_data='stl10'):

    if pretrain_data == 'imagenet100':
        mean = torch.tensor([0.485, 0.456, 0.406])
        std  = torch.tensor([0.229, 0.224, 0.225])
        transform = T.Compose([T.Resize(224, interpolation=Image.BICUBIC),
                               T.CenterCrop(224),
                               T.ToTensor(),
                               T.Normalize(mean, std)])

    elif pretrain_data == 'stl10':
        mean = torch.tensor([0.43, 0.42, 0.39])
        std  = torch.tensor([0.27, 0.26, 0.27])
        transform = T.Compose([T.Resize(96, interpolation=Image.BICUBIC),
                               T.CenterCrop(96),
                               T.ToTensor(),
                               T.Normalize(mean, std)])

    generator = lambda seed: torch.Generator().manual_seed(seed)
    if dataset == 'imagenet100':
        trainval = ImageNet100(datadir, split='train', transform=transform)
        train, val = random_split(trainval, [0.9, 0.1], generator=generator(41))
        test     = ImageNet100(datadir, split='val', transform=transform)
        num_classes = 100

    elif dataset == 'food101':
        trainval   = Food101(root=datadir, split='train', transform=transform)
        train, val = random_split(trainval, [68175, 7575], generator=generator(42))
        test       = Food101(root=datadir, split='test',  transform=transform)
        num_classes = 101

    elif dataset == 'cifar10':
        trainval   = CIFAR10(root=datadir, train=True,  transform=transform)
        train, val = random_split(trainval, [45000, 5000], generator=generator(43))
        test       = CIFAR10(root=datadir, train=False, transform=transform)
        num_classes = 10

    elif dataset == 'cifar100':
        trainval   = CIFAR100(root=datadir, train=True,  transform=transform)
        train, val = random_split(trainval, [45000, 5000], generator=generator(44))
        test       = CIFAR100(root=datadir, train=False, transform=transform)
        num_classes = 100

    elif dataset == 'sun397':
        trn_indices, val_indices = torch.load('data_slice/sun397.pth')
        trainval = SUN397(root=datadir, split='Training', transform=transform)
        train    = Subset(trainval, trn_indices)
        val      = Subset(trainval, val_indices)
        test     = SUN397(root=datadir, split='Testing',  transform=transform)
        num_classes = 397

    elif dataset == 'dtd':
        train    = DTD(root=datadir, split='train', transform=transform)
        val      = DTD(root=datadir, split='val',   transform=transform)
        trainval = ConcatDataset([train, val])
        test     = DTD(root=datadir, split='test',  transform=transform)
        num_classes = 47

    elif dataset == 'pets':
        trainval = OxfordIIITPet(root=datadir, split='trainval', transform=transform)
        # trainval   = Pets(root=datadir, split='trainval', transform=transform)
        train, val = random_split(trainval, [2940, 740], generator=generator(49))
        test       = OxfordIIITPet(root=datadir, split='test', transform=transform)
        num_classes = 37

    elif dataset == 'caltech101':
        transform.transforms.insert(0, T.Lambda(lambda img: img.convert('RGB')))
        D = Caltech101(datadir, transform=transform)
        trn_indices, val_indices, tst_indices = torch.load('data_slice/caltech101.pth')
        train    = Subset(D, trn_indices)
        val      = Subset(D, val_indices)
        trainval = ConcatDataset([train, val])
        test     = Subset(D, tst_indices)
        num_classes = 101

    elif dataset == 'flowers':
        train    = Flowers102(root=datadir, split='train',
                              transform=transform)
        val      = Flowers102(root=datadir, split='val',
                              transform=transform)
        trainval = ConcatDataset([train, val])
        test     = Flowers102(root=datadir, split='test',
                              transform=transform)
        num_classes = 102
    
    elif dataset == 'cars':
        trainval = StanfordCars(root=datadir, split='train', transform=transform)
        train, val = random_split(trainval, [0.9, 0.1], generator=generator(50))
        test     = StanfordCars(root=datadir, split='test', transform=transform)
        num_classes = 196

    elif dataset in ['flowers-5shot', 'flowers-10shot']:
        if dataset == 'flowers-5shot':
            n = 5
        else:
            n = 10
        train    = ImageFolder(os.path.join(datadir, 'trn'), transform=transform)
        val      = ImageFolder(os.path.join(datadir, 'val'), transform=transform)
        trainval = ImageFolder(os.path.join(datadir, 'trn'), transform=transform)
        trainval.samples += val.samples
        trainval.targets += val.targets
        indices = defaultdict(list)
        for i, y in enumerate(trainval.targets):
            indices[y].append(i)
        indices = sum([random.sample(indices[y], n) for y in indices.keys()], [])
        trainval = Subset(trainval, indices)
        test     = ImageFolder(os.path.join(datadir, 'tst'), transform=transform)
        num_classes = 102

    elif dataset == 'stl10':
        trainval   = STL10(root=datadir, split='train', transform=transform)
        test       = STL10(root=datadir, split='test',  transform=transform)
        train, val = random_split(trainval, [4500, 500], generator=generator(50))
        num_classes = 10

    elif dataset == 'mit67':
        trainval = ImageFolder(os.path.join(datadir, 'train'), transform=transform)
        test     = ImageFolder(os.path.join(datadir, 'test'),  transform=transform)
        train, val = random_split(trainval, [4690, 670], generator=generator(51))
        num_classes = 67

    elif dataset == 'cub200':
        trn_indices, val_indices = torch.load('data_slice/cub200.pth')
        trainval = Cub2011(datadir, train=True, transform=transform, loader=default_loader, download=False)
        train    = Subset(trainval, trn_indices)
        val      = Subset(trainval, val_indices)
        test     = Cub2011(datadir, train=False, transform=transform, loader=default_loader, download=False)
        num_classes = 200

    elif dataset == 'dog':
        trn_indices, val_indices = torch.load('data_slice/dog.pth')
        trainval = ImageFolder(os.path.join(datadir, 'train'), transform=transform)
        train    = Subset(trainval, trn_indices)
        val      = Subset(trainval, val_indices)
        test     = ImageFolder(os.path.join(datadir, 'test'),  transform=transform)
        num_classes = 120

    return dict(trainval=trainval,
                train=train,
                val=val,
                test=test,
                num_classes=num_classes)


def load_fewshot_datasets(dataset='cifar10',
                          datadir='/data',
                          pretrain_data='stl10'):

    if pretrain_data == 'imagenet100':
        mean = torch.tensor([0.485, 0.456, 0.406])
        std  = torch.tensor([0.229, 0.224, 0.225])
        transform = T.Compose([T.Resize(224, interpolation=Image.BICUBIC),
                               T.CenterCrop(224),
                               T.ToTensor(),
                               T.Normalize(mean, std)])

    elif pretrain_data == 'stl10':
        mean = torch.tensor([0.43, 0.42, 0.39])
        std  = torch.tensor([0.27, 0.26, 0.27])
        transform = T.Compose([T.Resize(96, interpolation=Image.BICUBIC),
                               T.CenterCrop(96),
                               T.ToTensor(),
                               T.Normalize(mean, std)])

    if dataset == 'cub200':
        train = ImageFolder(os.path.join(datadir, 'train'), transform=transform)
        test  = ImageFolder(os.path.join(datadir, 'test'),  transform=transform)
        test.samples = train.samples + test.samples
        test.targets = train.targets + test.targets

    elif dataset == 'fc100':
        train = ImageFolder(os.path.join(datadir, 'train'), transform=transform)
        test  = ImageFolder(os.path.join(datadir, 'test'),  transform=transform)

    elif dataset == 'plant_disease':
        train = ImageFolder(os.path.join(datadir, 'train'), transform=transform)
        test  = ImageFolder(os.path.join(datadir, 'test'),  transform=transform)
        test.samples = train.samples + test.samples
        test.targets = train.targets + test.targets

    return dict(test=test)