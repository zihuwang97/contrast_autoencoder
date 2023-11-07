import argparse
import os
import cv2
import numpy as np
import torch
from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

import torchvision
from models import load_backbone


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--device', type=str, default='cuda:0', help='GPU to be used')
    parser.add_argument('--pretrained', type=str, required=True)
    parser.add_argument(
        '--image-path',
        type=str,
        default=None,
        help='Input image path')
    parser.add_argument(
        '--output-image-path',
        type=str,
        default=None,
        help='Output image path')
    parser.add_argument(
        '--model',
        type=str,
        default='resnet50',
        help='model name')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen_smooth',
        action='store_true',
        help='Reduce noise by taking the first principle componenet'
        'of cam_weights*activations')
    parser.add_argument('--method', type=str, default='gradcam',
                        choices=['gradcam', 'gradcam++',
                                 'scorecam', 'xgradcam',
                                 'ablationcam', 'eigencam',
                                 'eigengradcam', 'layercam', 'fullgrad'],
                        help='Can be gradcam/gradcam++/scorecam/xgradcam'
                             '/ablationcam/eigencam/eigengradcam/layercam')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args


if __name__ == '__main__':
    """ python cam.py -image-path <path_to_image>
    Example usage of loading an image, and computing:
        1. CAM
        2. Guided Back Propagation
        3. Combining both
    """

    args = get_args()
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad}
    
    # Choose the target layer you want to compute the visualization for.
    # Usually this will be the last convolutional layer in the model.
    # Some common choices can be:
    # Resnet18 and 50: model.layer4
    # VGG, densenet161: model.features[-1]
    # mnasnet1_0: model.layers[-1]
    # You can print the model to help chose the layer
    # You can pass a list with several target layers,
    # in that case the CAMs will be computed per layer and then aggregated.
    # You can also try selecting all layers of a certain type, with e.g:
    # from pytorch_grad_cam.utils.find_layers import find_layer_types_recursive
    # find_layer_types_recursive(model, [torch.nn.ReLU])
    # target_layers = [model.layer4]
    # import pdb; pdb.set_trace()
    # rgb_img = cv2.imread(args.image_path, 1)[:, :, ::-1]
    
    ## LOAD MODEL
    model = torchvision.models.__dict__[args.model](num_classes=1000)
    checkpoint = torch.load(args.pretrained, map_location="cpu")
    state_dict = checkpoint['state_dict']
    for k in list(state_dict.keys()):
        # retain only encoder_q up to before the embedding layer
        if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
            # remove prefix
            state_dict[k[len("module.encoder_q."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]

    model.load_state_dict(state_dict, strict=False)
    model.fc = torch.nn.Identity()

    ## SPECIFY TARGET LAYERS
    target_layers = [model.layer4[-1]]
    reshape_transform = None

    model.eval()
    img_path = args.image_path
    if args.image_path:
        img_path = args.image_path
    else:
        import requests
        image_url = 'http://146.48.86.29/edge-mac/imgs/n02123045/ILSVRC2012_val_00023779.JPEG'
        img_path = image_url.split('/')[-1]
        if os.path.exists(img_path):
            img_data = requests.get(image_url).content
            with open(img_path, 'wb') as handler:
                handler.write(img_data)

    if args.output_image_path:
        save_name = args.output_image_path
    else:
        img_type = img_path.split('.')[-1]
        it_len = len(img_type)
        save_name = img_path.split('/')[-1][:-(it_len + 1)]
        save_name = save_name + '_' + args.model + '.' + img_type

    img = cv2.imread(img_path, 1)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    if args.model == 'resize':
        cv2.imwrite(save_name, img)
    else:
        rgb_img = img[:, :, ::-1]
        rgb_img = np.float32(rgb_img) / 255
        input_tensor = preprocess_image(rgb_img,
                                        mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])


        # We have to specify the target we want to generate
        # the Class Activation Maps for.
        # If targets is None, the highest scoring category (for every member in the batch) will be used.
        # You can target specific categories by
        # targets = [e.g ClassifierOutputTarget(281)]
        targets = None

        # Using the with statement ensures the context is freed, and you can
        # recreate different CAM objects in a loop.
        cam_algorithm = methods[args.method]
        with cam_algorithm(model=model,
                        target_layers=target_layers,
                        use_cuda=args.use_cuda,
                        reshape_transform=reshape_transform, 
                        ) as cam:

            # AblationCAM and ScoreCAM have batched implementations.
            # You can override the internal batch size for faster computation.
            cam.batch_size = 32
            grayscale_cam = cam(input_tensor=input_tensor,
                                targets=targets,
                                aug_smooth=args.aug_smooth,
                                eigen_smooth=args.eigen_smooth)
            
            import pandas as pd
            t_np = torch.tensor(grayscale_cam[0]).numpy() #convert to Numpy array
            df = pd.DataFrame(t_np) #convert to a dataframe
            df.to_csv("testfile",index=False)
            print(torch.tensor(grayscale_cam).max(), torch.tensor(grayscale_cam).min())
            # Here grayscale_cam has only one image in the batch
            grayscale_cam = grayscale_cam[0, :]

            cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

            # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
            cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

        # gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)
        # gb = gb_model(input_tensor, target_category=None)

        # cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
        # cam_gb = deprocess_image(cam_mask * gb)
        # gb = deprocess_image(gb)

        # cv2.imwrite(f'{args.method}_cam_poolformer_s24.jpg', cam_image)
        
        cv2.imwrite(save_name, cam_image)
        # cv2.imwrite(f'{args.method}_gb_poolformer_s24.jpg', gb)
        # cv2.imwrite(f'{args.method}_cam_gb_poolformer_s24.jpg', cam_gb)