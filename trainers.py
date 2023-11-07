import math
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from ignite.engine import Engine
import ignite.distributed as idist



def simsiam(backbone,
            projector,
            predictor,
            optimizers,
            device,
            ):

    def training_step(engine, batch):
        backbone.train()
        projector.train()
        predictor.train()

        for o in optimizers:
            o.zero_grad()

        (x1, x2, x3), _ = batch
        y1, y2, y3 = backbone(x1), backbone(x2), backbone(x3)

        bsz = y3.size(0)
        y3_normed = F.normalize(y3, dim=-1)
        mean3 = torch.mean(y3_normed, dim=0)
        y3_var = y3_normed - mean3.unsqueeze(0)
        cov_mat3 = (y3_var.T @ y3_var)/bsz
        diag3 = torch.diag(cov_mat3)
        loss_decov = (torch.norm(cov_mat3) - torch.norm(diag3))/2

        z1 = projector(y1)
        z2 = projector(y2)
        p1 = predictor(z1)
        p2 = predictor(z2)
        loss1 = F.cosine_similarity(p1, z2.detach(), dim=-1).mean().mul(-1)
        loss2 = F.cosine_similarity(p2, z1.detach(), dim=-1).mean().mul(-1)
        loss = (loss1+loss2).mul(0.5)

        loss = loss + loss_decov
        outputs = dict(loss=loss)
        
        loss.backward()

        for o in optimizers:
            o.step()

        return outputs

    return Engine(training_step)


def moco(backbone,
         projector,
         optimizers,
         device,
         momentum=0.999,
         K=65536,
         T=0.2,
         ):

    target_backbone  = deepcopy(backbone)
    target_projector = deepcopy(projector)
    for p in list(target_backbone.parameters())+list(target_projector.parameters()):
        p.requires_grad = False

    queue = F.normalize(torch.randn(K, 128).to(device)).detach()
    queue.requires_grad = False
    queue.ptr = 0

    def training_step(engine, batch):
        backbone.train()
        projector.train()
        target_backbone.train()
        target_projector.train()

        for o in optimizers:
            o.zero_grad()

        (x1, x2), _ = batch
        y1 = backbone(x1)
        z1 = F.normalize(projector(y1))
        with torch.no_grad():
            y2 = target_backbone(x2)
            z2 = F.normalize(target_projector(y2))

        l_pos = torch.einsum('nc,nc->n', [z1, z2]).unsqueeze(-1)
        l_neg = torch.einsum('nc,kc->nk', [z1, queue.clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=1).div(T)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)
        loss = F.cross_entropy(logits, labels)
        outputs = dict(loss=loss)

        loss.backward()

        for o in optimizers:
            o.step()

        # momentum network update
        for online, target in [(backbone, target_backbone), (projector, target_projector)]:
            for p1, p2 in zip(online.parameters(), target.parameters()):
                p2.data.mul_(momentum).add_(p1.data, alpha=1-momentum)

        # queue update
        keys = idist.utils.all_gather(z1)
        queue[queue.ptr:queue.ptr+keys.shape[0]] = keys
        queue.ptr = (queue.ptr+keys.shape[0]) % K

        return outputs

    engine = Engine(training_step)
    return engine


def simclr(backbone,
           projector,
           decoder,
           optimizers,
           device,
           cam,
           T=0.2,
           ):

    def training_step(engine, batch):
        backbone.train()
        projector.train()

        for o in optimizers:
            o.zero_grad()

        (x1, x2), _ = batch
        y1 = backbone(x1)
        y2 = backbone(x2)

        z1 = F.normalize(projector(y1))
        z2 = F.normalize(projector(y2))

        z = torch.cat([z1, z2], 0)
        scores = torch.einsum('ik, jk -> ij', z, z).div(T)
        n = z1.shape[0]
        labels = torch.tensor(list(range(n, 2*n)) + list(range(0, n)), device=scores.device)
        masks = torch.zeros_like(scores, dtype=torch.bool)
        for i in range(2*n):
            masks[i, i] = True
        scores = scores.masked_fill(masks, float('-inf'))
        loss = F.cross_entropy(scores, labels)

        # RECONSTRUCTION LOSS
        y = torch.cat([y1, y2], 0)
        x = torch.cat([x1, x2], 0).to(backbone.device)

        # TODO: Obtain CAM mask
        grayscale_cam = cam(input_tensor=x,
                            targets=None,
                            aug_smooth=True,
                            eigen_smooth=True)
        cam_mask = torch.where(

        x_hat = decoder(y)
        l_rec = F.mse_loss(x_hat, x, reduction="mean")

        loss = 0.05*l_rec + loss

        outputs = dict(loss=loss)

        loss.backward()

        for o in optimizers:
            o.step()

        return outputs

    engine = Engine(training_step)
    return engine


def byol(backbone,
         projector,
         predictor,
         ss_predictor,
         t1,
         t2,
         optimizers,
         device,
         ss_objective,
         momentum=0.996,
         ):

    target_backbone  = deepcopy(backbone)
    target_projector = deepcopy(projector)
    for p in list(target_backbone.parameters())+list(target_projector.parameters()):
        p.requires_grad = False

    def training_step(engine, batch):
        backbone.train()
        projector.train()
        predictor.train()

        for o in optimizers:
            o.zero_grad()

        x1, x2, d1, d2 = prepare_training_batch(batch, t1, t2, device)
        y1, y2 = backbone(x1), backbone(x2)
        z1, z2 = projector(y1), projector(y2)
        p1, p2 = predictor(z1), predictor(z2)
        with torch.no_grad():
            tgt1 = target_projector(target_backbone(x1))
            tgt2 = target_projector(target_backbone(x2))

        loss1 = F.cosine_similarity(p1, tgt2.detach(), dim=-1).mean().mul(-1)
        loss2 = F.cosine_similarity(p2, tgt1.detach(), dim=-1).mean().mul(-1)
        loss = (loss1+loss2).mul(2)

        outputs = dict(loss=loss)
        outputs['z1'] = z1
        outputs['z2'] = z2

        ss_losses = ss_objective(ss_predictor, y1, y2, d1, d2)
        (loss+ss_losses['total']).backward()
        for k, v in ss_losses.items():
            outputs[f'ss/{k}'] = v

        for o in optimizers:
            o.step()

        # momentum network update
        m = 1 - (1-momentum)*(math.cos(math.pi*(engine.state.epoch-1)/engine.state.max_epochs)+1)/2
        for online, target in [(backbone, target_backbone), (projector, target_projector)]:
            for p1, p2 in zip(online.parameters(), target.parameters()):
                p2.data.mul_(m).add_(p1.data, alpha=1-m)

        return outputs

    return Engine(training_step)


def distributed_sinkhorn(Q, nmb_iters):
    with torch.no_grad():
        Q = shoot_infs(Q)
        sum_Q = torch.sum(Q)
        # idist.utils.all_reduce(sum_Q)
        Q /= sum_Q
        r = torch.ones(Q.shape[0]).cuda(non_blocking=True) / Q.shape[0]
        # c = torch.ones(Q.shape[1]).cuda(non_blocking=True) / (args.world_size * Q.shape[1])
        c = torch.ones(Q.shape[1]).cuda(non_blocking=True) / (1 * Q.shape[1])
        for it in range(nmb_iters):
            u = torch.sum(Q, dim=1)
            # idist.utils.all_reduce(u)
            u = r / u
            u = shoot_infs(u)
            Q *= u.unsqueeze(1)
            Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)
        return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()


def shoot_infs(inp_tensor):
    """Replaces inf by maximum of tensor"""
    mask_inf = torch.isinf(inp_tensor)
    ind_inf = torch.nonzero(mask_inf)
    if len(ind_inf) > 0:
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = 0
            elif len(ind) == 1:
                inp_tensor[ind[0]] = 0
        m = torch.max(inp_tensor)
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = m
            elif len(ind) == 1:
                inp_tensor[ind[0]] = m
    return inp_tensor


def swav(backbone,
         projector,
         prototypes,
         ss_predictor,
         t1,
         t2,
         optimizers,
         device,
         ss_objective,
         epsilon=0.05,
         n_iters=3,
         temperature=0.1,
         freeze_n_iters=410,
         ):

    def training_step(engine, batch):
        backbone.train()
        projector.train()
        prototypes.train()

        for o in optimizers:
            o.zero_grad()

        with torch.no_grad():
            w = prototypes.weight.data.clone()
            w = F.normalize(w, dim=1, p=2)
            prototypes.weight.copy_(w)

        x1, x2, d1, d2 = prepare_training_batch(batch, t1, t2, device)
        y1, y2 = backbone(x1), backbone(x2)
        z1, z2 = projector(y1), projector(y2)
        z1 = F.normalize(z1, dim=1, p=2)
        z2 = F.normalize(z2, dim=1, p=2)
        p1, p2 = prototypes(z1), prototypes(z2)

        q1 = distributed_sinkhorn(torch.exp(p1 / epsilon).t(), n_iters)
        q2 = distributed_sinkhorn(torch.exp(p2 / epsilon).t(), n_iters)

        p1 = F.softmax(p1 / temperature, dim=1)
        p2 = F.softmax(p2 / temperature, dim=1)

        loss1 = -torch.mean(torch.sum(q1 * torch.log(p2), dim=1))
        loss2 = -torch.mean(torch.sum(q2 * torch.log(p1), dim=1))
        loss = loss1+loss2

        outputs = dict(loss=loss)
        outputs['z1'] = z1
        outputs['z2'] = z2

        ss_losses = ss_objective(ss_predictor, y1, y2, d1, d2)
        (loss+ss_losses['total']).backward()
        for k, v in ss_losses.items():
            outputs[f'ss/{k}'] = v

        if engine.state.iteration < freeze_n_iters:
            for p in prototypes.parameters():
                p.grad = None

        for o in optimizers:
            o.step()

        return outputs

    return Engine(training_step)


def collect_features(backbone,
                     dataloader,
                     device,
                     normalize=True,
                     dst=None,
                     verbose=False):

    if dst is None:
        dst = device

    backbone.eval()
    with torch.no_grad():
        features = []
        labels   = []
        for i, (x, y) in enumerate(dataloader):
            if x.ndim == 5:
                _, n, c, h, w = x.shape
                x = x.view(-1, c, h, w)
                y = y.view(-1, 1).repeat(1, n).view(-1)
            z = backbone(x.to(device))
            if normalize:
                z = F.normalize(z, dim=-1)
            features.append(z.to(dst).detach())
            labels.append(y.to(dst).detach())
            if verbose and (i+1) % 10 == 0:
                print(i+1)
        features = idist.utils.all_gather(torch.cat(features, 0).detach())
        labels   = idist.utils.all_gather(torch.cat(labels, 0).detach())

    return features, labels


def nn_evaluator(backbone,
                 trainloader,
                 testloader,
                 device):

    def evaluator():
        backbone.eval()
        with torch.no_grad():
            features, labels = collect_features(backbone, trainloader, device)
            corrects, total = 0, 0
            for x, y in testloader:
                z = F.normalize(backbone(x.to(device)), dim=-1)
                scores = torch.einsum('ik, jk -> ij', z, features)
                preds = labels[scores.argmax(1)]

                corrects += (preds.cpu() == y).long().sum().item()
                total += y.shape[0]
            corrects = idist.utils.all_reduce(corrects)
            total = idist.utils.all_reduce(total)

        return corrects / total

    return evaluator