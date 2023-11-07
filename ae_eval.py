import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms as T
from torchvision.datasets import STL10
from torchvision.utils import save_image
from argparse import ArgumentParser

from models import load_backbone, load_decoder
from datasets import ImageNet100
from transforms import GaussianBlur

parser = ArgumentParser()
parser.add_argument('--ckpt', type=str, required=True)
parser.add_argument('--dataset', type=str, default='imagenet100')
parser.add_argument('--datadir', type=str, default='/data')
parser.add_argument('--model', type=str, default='resnet50')
parser.add_argument('--device', type=str, default='cuda:7')
parser.add_argument('--latent-dim', type=int, default='512')
args = parser.parse_args()
if args.dataset=='imagenet100':
        args.input_height = 224
elif args.dataset=='stl10':
    args.input_height = 96

args.num_backbone_features = 512 if args.model=='resnet18' else 2048


# Load the pre-trained AutoEncoder
ckpt = torch.load(args.ckpt)
backbone = load_backbone(args).to(args.device)
decoder = load_decoder(args).to(args.device)
backbone.load_state_dict(ckpt['backbone'])
decoder.load_state_dict(ckpt['decoder'])


# Load data
if args.dataset=='imagenet100':
    mean = torch.tensor([0.485, 0.456, 0.406])
    std  = torch.tensor([0.229, 0.224, 0.225])
    test_transform = T.Compose([T.Resize(224),
                                T.CenterCrop(224),
                                T.ToTensor(),
                                T.Normalize(mean, std)])
    # test_transform = T.Compose([T.RandomResizedCrop(224, scale=(0.2, 1.0)),
    #                             T.RandomHorizontalFlip(),
    #                             T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    #                             T.RandomGrayscale(p=0.2),
    #                             T.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
    #                             T.ToTensor(),
    #                             T.Normalize(mean, std)])

    dataset = ImageNet100(args.datadir, split='train', transform=test_transform)
elif args.dataset=='stl10':
    mean = torch.tensor([0.43, 0.42, 0.39])
    std  = torch.tensor([0.27, 0.26, 0.27])
    test_transform = T.Compose([T.Resize(96),
                                T.CenterCrop(96),
                                T.ToTensor(),
                                T.Normalize(mean, std)])
    dataset = STL10(args.datadir, split='train+unlabeled', transform=test_transform)

dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=0, pin_memory=True)


# Calculate reconstruction loss
data, _ = next(iter(dataloader))
data = data.to(args.device)
data_rec = decoder(backbone(data))
reconstruction_loss = torch.nn.functional.mse_loss(data_rec, data, reduction="mean")
print('====> mean reconstruction loss : {}'.format(reconstruction_loss))


# Save the raw and the reconstructed image
save_image(data[0], 'raw_img.png')
save_image(data_rec[0], 'rec_img.png')

