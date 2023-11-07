import os
from argparse import ArgumentParser
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

import ignite
from ignite.engine import Events
import ignite.distributed as idist

from datasets import load_pretrain_datasets
from models import load_backbone, load_mlp, load_ss_predictor, load_decoder
import trainers
from utils import Logger

from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad


def simsiam(args):
    out_dim = 2048
    device = idist.device()

    build_model  = partial(idist.auto_model, sync_bn=True)
    backbone     = build_model(load_backbone(args))
    projector    = build_model(load_mlp(args.num_backbone_features,
                                        out_dim,
                                        out_dim,
                                        num_layers=2+int(args.dataset.startswith('imagenet')),
                                        last_bn=True))
    predictor    = build_model(load_mlp(out_dim,
                                        out_dim // 4,
                                        out_dim,
                                        num_layers=2,
                                        last_bn=False))

    SGD = partial(optim.SGD, lr=args.lr, weight_decay=args.wd, momentum=args.momentum)
    build_optim = lambda x: idist.auto_optim(SGD(x))
    optimizers = [build_optim(list(backbone.parameters())+list(projector.parameters())),
                  build_optim(list(predictor.parameters()))]
    schedulers = [optim.lr_scheduler.CosineAnnealingLR(optimizers[0], args.max_epochs)]

    trainer = trainers.simsiam(backbone=backbone,
                               projector=projector,
                               predictor=predictor,
                               optimizers=optimizers,
                               device=device,)

    return dict(backbone=backbone,
                projector=projector,
                predictor=predictor,
                optimizers=optimizers,
                schedulers=schedulers,
                trainer=trainer)


def moco(args):
    out_dim = 128
    device = idist.device()


    build_model  = partial(idist.auto_model, sync_bn=True)
    backbone     = build_model(load_backbone(args))
    projector    = build_model(load_mlp(args.num_backbone_features,
                                        args.num_backbone_features,
                                        out_dim,
                                        num_layers=2,
                                        last_bn=False))

    SGD = partial(optim.SGD, lr=args.lr, weight_decay=args.wd, momentum=args.momentum)
    build_optim = lambda x: idist.auto_optim(SGD(x))
    optimizers = [build_optim(list(backbone.parameters())+list(projector.parameters()))]
    schedulers = [optim.lr_scheduler.CosineAnnealingLR(optimizers[0], args.max_epochs)]

    trainer = trainers.moco(
            backbone=backbone,
            projector=projector,
            optimizers=optimizers,
            device=device,)

    return dict(backbone=backbone,
                projector=projector,
                optimizers=optimizers,
                schedulers=schedulers,
                trainer=trainer)

def simclr(args):
    out_dim = 128
    device = idist.device()

    cam_methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad}
    
    # MODEL
    build_model  = partial(idist.auto_model, sync_bn=True)
    backbone     = build_model(load_backbone(args))
    projector    = build_model(load_mlp(args.num_backbone_features,
                                        args.num_backbone_features,
                                        out_dim,
                                        num_layers=2,
                                        last_bn=False))
    decoder      = build_model(load_decoder(args))     # BUILD DECODER

    # CAM 
    cam = cam_methods[args.use_cam](model=backbone,
                                    target_layers=backbone.module.layer4[-1],
                                    use_cuda=True,
                                    reshape_transform=None)

    # OPTIMIZER & SCHEDULER
    SGD = partial(optim.SGD, lr=args.lr, weight_decay=args.wd, momentum=args.momentum)
    ADAM = partial(optim.Adam, lr=1e-4)
    build_optim  = lambda x: idist.auto_optim(SGD(x))
    build_adam = lambda x: idist.auto_optim(ADAM(x))
    optimizers   = [build_optim(list(backbone.parameters())+list(projector.parameters())), 
                    build_adam(list(decoder.parameters()))]
    schedulers   = [optim.lr_scheduler.CosineAnnealingLR(optimizers[0], args.max_epochs),
                    optim.lr_scheduler.CosineAnnealingLR(optimizers[1], args.max_epochs)]

    # TRAINER
    trainer = trainers.simclr(backbone=backbone,
                              projector=projector,
                              decoder=decoder,
                              optimizers=optimizers,
                              device=device,
                              cam=cam,)

    return dict(backbone=backbone,
                projector=projector,
                decoder=decoder,
                optimizers=optimizers,
                schedulers=schedulers,
                trainer=trainer)


def byol(args, t1, t2):
    out_dim = 256
    h_dim = 4096
    device = idist.device()

    ss_objective = SSObjective(
        crop  = args.ss_crop,
        color = args.ss_color,
        flip  = args.ss_flip,
        blur  = args.ss_blur,
        rot   = args.ss_rot,
        only  = args.ss_only,
    )

    build_model  = partial(idist.auto_model, sync_bn=True)
    backbone     = build_model(load_backbone(args))
    projector    = build_model(load_mlp(args.num_backbone_features,
                                        h_dim,
                                        out_dim,
                                        num_layers=2,
                                        last_bn=False))
    predictor    = build_model(load_mlp(out_dim,
                                        h_dim,
                                        out_dim,
                                        num_layers=2,
                                        last_bn=False))
    ss_predictor = load_ss_predictor(args.num_backbone_features, ss_objective)
    ss_predictor = { k: build_model(v) for k, v in ss_predictor.items() }
    ss_params = sum([list(v.parameters()) for v in ss_predictor.values()], [])

    SGD = partial(optim.SGD, lr=args.lr, weight_decay=args.wd, momentum=args.momentum)
    build_optim = lambda x: idist.auto_optim(SGD(x))
    optimizers = [build_optim(list(backbone.parameters())+list(projector.parameters())+ss_params+list(predictor.parameters()))]
    schedulers = [optim.lr_scheduler.CosineAnnealingLR(optimizers[0], args.max_epochs)]

    trainer = trainers.byol(backbone=backbone,
                            projector=projector,
                            predictor=predictor,
                            ss_predictor=ss_predictor,
                            t1=t1, t2=t2,
                            optimizers=optimizers,
                            device=device,
                            ss_objective=ss_objective)

    return dict(backbone=backbone,
                projector=projector,
                predictor=predictor,
                ss_predictor=ss_predictor,
                optimizers=optimizers,
                schedulers=schedulers,
                trainer=trainer)


def swav(args, t1, t2):
    out_dim = 128
    h_dim = 2048
    device = idist.device()

    ss_objective = SSObjective(
        crop  = args.ss_crop,
        color = args.ss_color,
        flip  = args.ss_flip,
        blur  = args.ss_blur,
        rot   = args.ss_rot,
        only  = args.ss_only,
    )

    build_model  = partial(idist.auto_model, sync_bn=True)
    backbone     = build_model(load_backbone(args))
    projector    = build_model(load_mlp(args.num_backbone_features,
                                        h_dim,
                                        out_dim,
                                        num_layers=2,
                                        last_bn=False))
    prototypes   = build_model(nn.Linear(out_dim, 100, bias=False))
    ss_predictor = load_ss_predictor(args.num_backbone_features, ss_objective)
    ss_predictor = { k: build_model(v) for k, v in ss_predictor.items() }
    ss_params = sum([list(v.parameters()) for v in ss_predictor.values()], [])

    SGD = partial(optim.SGD, lr=args.lr, weight_decay=args.wd, momentum=args.momentum)
    build_optim = lambda x: idist.auto_optim(SGD(x))
    optimizers = [build_optim(list(backbone.parameters())+list(projector.parameters())+ss_params+list(prototypes.parameters()))]
    schedulers = [optim.lr_scheduler.CosineAnnealingLR(optimizers[0], args.max_epochs)]

    trainer = trainers.swav(backbone=backbone,
                            projector=projector,
                            prototypes=prototypes,
                            ss_predictor=ss_predictor,
                            t1=t1, t2=t2,
                            optimizers=optimizers,
                            device=device,
                            ss_objective=ss_objective)

    return dict(backbone=backbone,
                projector=projector,
                prototypes=prototypes,
                ss_predictor=ss_predictor,
                optimizers=optimizers,
                schedulers=schedulers,
                trainer=trainer)


def main(local_rank, args):
    cudnn.benchmark = True
    device = idist.device()
    logger = Logger(args.logdir, args.resume)

    # DATASETS
    datasets = load_pretrain_datasets(dataset=args.dataset,
                                      datadir=args.datadir,
                                      color_aug=args.color_aug)
    build_dataloader = partial(idist.auto_dataloader,
                               batch_size=args.batch_size,
                               num_workers=args.num_workers,
                               shuffle=True,
                               pin_memory=True)
    trainloader = build_dataloader(datasets['train'], drop_last=True)
    valloader   = build_dataloader(datasets['val']  , drop_last=False)
    testloader  = build_dataloader(datasets['test'],  drop_last=False)

    # MODELS
    if args.framework == 'simsiam':
        models = simsiam(args)
    elif args.framework == 'simclr':
        models = simclr(args)
    elif args.framework == 'moco':
        models = moco(args)
    else:
        print("======> your method is not supported")
        exit()

    trainer   = models['trainer']
    evaluator = trainers.nn_evaluator(backbone=models['backbone'],
                                      trainloader=valloader,
                                      testloader=testloader,
                                      device=device)

    if args.distributed:
        @trainer.on(Events.EPOCH_STARTED)
        def set_epoch(engine):
            for loader in [trainloader, valloader, testloader]:
                loader.sampler.set_epoch(engine.state.epoch)

    @trainer.on(Events.ITERATION_STARTED)
    def log_lr(engine):
        lrs = {}
        for i, optimizer in enumerate(models['optimizers']):
            for j, pg in enumerate(optimizer.param_groups):
                lrs[f'lr/{i}-{j}'] = pg['lr']
        logger.log(engine, engine.state.iteration, print_msg=False, **lrs)

    @trainer.on(Events.ITERATION_COMPLETED)
    def log(engine):
        loss = engine.state.output.pop('loss')
    
        logger.log(engine, engine.state.iteration,
                   print_msg=engine.state.iteration % args.print_freq == 0,
                   loss=loss)

        logger.log(engine, engine.state.iteration,
                   print_msg=False,
                   **engine.state.output)

    @trainer.on(Events.EPOCH_COMPLETED(every=args.eval_freq))
    def evaluate(engine):
        acc = evaluator()
        logger.log(engine, engine.state.epoch, acc=acc)

    @trainer.on(Events.EPOCH_COMPLETED)
    def update_lr(engine):
        for scheduler in models['schedulers']:
            scheduler.step()

    @trainer.on(Events.EPOCH_COMPLETED(every=args.ckpt_freq))
    def save_ckpt(engine):
        logger.save(engine, **models)

    if args.resume is not None:
        @trainer.on(Events.STARTED)
        def load_state(engine):
            ckpt = torch.load(os.path.join(args.logdir, f'ckpt-{args.resume}.pth'), map_location='cpu')
            for k, v in models.items():
                if isinstance(v, nn.parallel.DistributedDataParallel):
                    v = v.module

                if hasattr(v, 'state_dict'):
                    v.load_state_dict(ckpt[k])

                if type(v) is list and hasattr(v[0], 'state_dict'):
                    for i, x in enumerate(v):
                        x.load_state_dict(ckpt[k][i])

    trainer.run(trainloader, max_epochs=args.max_epochs)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--logdir', type=str, required=True)
    parser.add_argument('--resume', type=int, default=None)
    parser.add_argument('--dataset', type=str, default='stl10')
    parser.add_argument('--datadir', type=str, default='/data')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--max-epochs', type=int, default=200)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--latent-dim', type=int, default=512)

    parser.add_argument('--framework', type=str, default='simclr')
    parser.add_argument('--use-cam', type=str, default=None)

    parser.add_argument('--base-lr', type=float, default=0.03)
    parser.add_argument('--wd', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9)

    parser.add_argument('--print-freq', type=int, default=200)
    parser.add_argument('--ckpt-freq', type=int, default=10)
    parser.add_argument('--eval-freq', type=int, default=1)

    parser.add_argument('--color-aug', type=str, default='default')
    parser.add_argument('--dist-url', type=str, default='tcp://localhost:10002')


    args = parser.parse_args()
    args.lr = args.base_lr * args.batch_size / 256
    if args.dataset=='imagenet100':
        args.input_height = 224
    elif args.dataset=='stl10':
        args.input_height = 96
    if not args.distributed:
        with idist.Parallel() as parallel:
            parallel.run(main, args)
    else:
        with idist.Parallel('nccl', nproc_per_node=torch.cuda.device_count()) as parallel:#, init_method=args.dist_url) as parallel:
            parallel.run(main, args)