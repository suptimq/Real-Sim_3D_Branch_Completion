import torch
import torch.nn as nn
import os
import json
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter
from utils.metrics import Metrics
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2, DensityAwareChamferDistance
from losses.losses import CPCLoss, RepulsionLoss, AdversarialLoss, compute_loss_pre, compute_loss, compute_loss_supervision

import torch.optim as optim

import wandb
import numpy as np
from collections import OrderedDict


def get_alpha(epoch, varying_constant, varying_constant_epochs):

    alpha = varying_constant[-1]
    for i, ep in enumerate(varying_constant_epochs):
        if epoch < ep:
            alpha = varying_constant[i]

    return alpha


def colorized_pc(pc, color=1):

    assert color >= 1 and color <= 14, f'Invalid Color Value {color}'
    assert pc.shape[-1] == 3, f'Invalid Point Cloud Shape {pc.shape}'

    colored_pc = np.ones((pc.shape[0], 1)) * color
    colored_pc = np.concatenate((pc, colored_pc), axis=1)

    return colored_pc


def run_net(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    # build dataset
    (train_sampler, train_dataloader), (_, test_dataloader) = builder.dataset_builder(args, config.dataset.train), \
                                                            builder.dataset_builder(args, config.dataset.val)
    # build model
    base_model = builder.model_builder(config)
    discriminator = builder.model_builder(config.model.discriminator)
    adversarial_loss = AdversarialLoss()
    if args.use_gpu:
        base_model.to(args.local_rank)
        discriminator.to(args.local_rank)
    
    # parameter setting
    start_epoch = 0
    best_metrics = None
    metrics = None

    # resume ckpts
    if args.resume:
        start_epoch, best_metrics = builder.resume_model(base_model, discriminator, args, logger = logger)
        best_metrics = Metrics(config.loss.consider_metric, best_metrics)
    elif args.start_ckpts is not None:
        builder.load_model(base_model, discriminator, args.start_ckpts, logger = logger)

    # print model info
    print_log('Trainable_parameters:', logger = logger)
    print_log('=' * 25, logger = logger)
    for name, param in base_model.named_parameters():
        if param.requires_grad:
            print_log(name, logger=logger)
    print_log('=' * 25, logger = logger)
    
    print_log('Untrainable_parameters:', logger = logger)
    print_log('=' * 25, logger = logger)
    for name, param in base_model.named_parameters():
        if not param.requires_grad:
            print_log(name, logger=logger)
    print_log('=' * 25, logger = logger)

    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            discriminator = torch.nn.SyncBatchNorm.convert_sync_batchnorm(discriminator)
            print_log('Using Synchronized BatchNorm ...', logger = logger)
        base_model = nn.parallel.DistributedDataParallel(base_model, device_ids=[args.local_rank % torch.cuda.device_count()], find_unused_parameters=True)
        discriminator = nn.parallel.DistributedDataParallel(discriminator, device_ids=[args.local_rank % torch.cuda.device_count()], find_unused_parameters=True)
        print_log('Using Distributed Data parallel ...' , logger = logger)
    else:
        print_log('Using Data parallel ...' , logger = logger)
        base_model = nn.DataParallel(base_model).cuda()
        discriminator = nn.DataParallel(discriminator).cuda()
    # optimizer & scheduler
    optimizer = builder.build_optimizer(base_model, config)
    optimizer_D = optim.AdamW(discriminator.parameters(), **config.optimizer.kwargs)
    
    # Criterion in evaluation
    eval_criterion = OrderedDict({
        ('L1', ChamferDistanceL1()), 
        ('L2', ChamferDistanceL2()),
        ('DCD', DensityAwareChamferDistance()),
        ('Repulsion', RepulsionLoss())
    })

    if args.resume:
        builder.resume_optimizer(optimizer, optimizer_D, args, logger = logger)
    scheduler = builder.build_scheduler(base_model, optimizer, config, last_epoch=start_epoch-1)
    scheduler_D = builder.build_scheduler(discriminator, optimizer_D, config, last_epoch=start_epoch-1)

    # trainval
    # training
    base_model.zero_grad()
    discriminator.zero_grad()

    major_loss_list = [f'Sparse{config.loss.consider_metric}', f'Dense{config.loss.consider_metric}']

    if 'cpc' in config.loss.additional_metrics.keys():
        eval_criterion['CPC'] = CPCLoss()

    if isinstance(config.model.discriminator, dict):
        eval_criterion['Adversarial'] = AdversarialLoss()

    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # TODO come up with a better way to arrange loss list
        extra_loss_list = []
        if 'repulsion' in config.loss.additional_metrics.keys():
            extra_loss_list.append('RepulsionLoss')
        if 'cpc' in config.loss.additional_metrics.keys() and epoch > config.loss.additional_metrics['cpc']['turn_on_ep']:
            extra_loss_list.append('P2PVarLoss')
            extra_loss_list.append('P2SMeanLoss')

        extra_loss_list.append('SkelLoss')

        if isinstance(config.model.discriminator, dict):
            extra_loss_list.append('GeneratorLoss')
            extra_loss_list.append('DiscriminatorLoss')

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(major_loss_list + extra_loss_list)

        num_iter = 0

        base_model.train()  # set model to training mode
        discriminator.train()
        n_batches = len(train_dataloader)
        for idx, (taxonomy_ids, model_ids, data) in enumerate(train_dataloader):
            data_time.update(time.time() - batch_start_time)
            npoints = config.dataset.train._base_.N_POINTS
            dataset_name = config.dataset.train._base_.NAME

            centers = None
            center_radii = None
            center_dirs = None

            if dataset_name == 'PCN' or dataset_name == 'Completion3D' or dataset_name == 'Projected_ShapeNet':
                partial = data[0].cuda()
                gt = data[1].cuda()
                if config.dataset.train._base_.CARS:
                    if idx == 0:
                        print_log('padding while KITTI training', logger=logger)
                    partial = misc.random_dropping(partial, epoch) # specially for KITTI finetune

            elif dataset_name == 'ShapeNet':
                gt = data.cuda()
                partial, _ = misc.seprate_point_cloud(gt, npoints, [int(npoints * 1/4) , int(npoints * 3/4)], fixed_points = None)
                partial = partial.cuda()
            elif dataset_name == 'PCNSkel':
                partial = data['partial'].cuda()
                gt = data['gt'].cuda()
                centers = data['centers'].cuda()
                center_radii = data['center_radii'].cuda()
                center_dirs = data['center_directions'].cuda()
                if len(center_radii.shape) == 2:
                    center_radii = center_radii.unsqueeze(-1)
                assert len(centers.shape) == len(center_radii.shape) == len(center_dirs.shape) == 3, "Shape Error"
                assert centers is not None and center_radii is not None and center_dirs is not None, "Invalid Segment Data"
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

            num_iter += 1
           
            ret = base_model(partial)

            coarse_points = ret[0]
            dense_points = ret[3] if config.NAME == 'AdaPoinTr' else ret[1]
            skel_xyz = ret[4] if len(ret) > 4 else None

            loss_dict = base_model.module.get_loss(
                ret, gt, centers=centers, center_radii=center_radii, center_dirs=center_dirs, epoch=epoch)
            loss_dict['generator_loss'] = adversarial_loss(discriminator(dense_points.detach()), 1.0)

            _loss = 0
            for loss_name, loss_val in loss_dict.items():
                if config.loss.consider_metric == 'DCD' and loss_name == 'fine_loss':
                    loss_weight = get_alpha(epoch, config.loss.varying_constant, config.loss.varying_constant_epochs)
                elif loss_name == 'skel_loss':
                    loss_weight = config.loss.additional_metrics['skel']['alpha']
                else:
                    loss_weight = 1

                _loss += loss_val * loss_weight

            _loss.backward()

            # forward
            if num_iter == config.step_per_update:
                torch.nn.utils.clip_grad_norm_(base_model.parameters(), getattr(config, 'grad_norm_clip', 10), norm_type=2, error_if_nonfinite=True)
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

            # -------------------------
            # Train Discriminator
            # -------------------------
            real_loss = adversarial_loss(discriminator(gt), 1.0)
            fake_loss = adversarial_loss(discriminator(dense_points.detach()), 0.0)
            d_loss = real_loss + fake_loss

            loss_dict['disciminator_loss'] = d_loss

            # raise ddp gradient error when using 1 GPU
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), getattr(config, 'grad_norm_clip', 10), norm_type=2, error_if_nonfinite=True)
            optimizer_D.step()
            discriminator.zero_grad()

            if args.distributed:
                reduce_loss_list = []
                for loss_name, loss_val in loss_dict.items():
                    loss_multiplier = 1000 if loss_name in ['coarse_loss', 'fine_loss'] else 1
                    reduce_loss_list.append(dist_utils.reduce_tensor(loss_val, args).item() * loss_multiplier)
                losses.update(reduce_loss_list)
            else:
                for loss_name, loss_val in loss_dict.items():
                    loss_multiplier = 1000 if loss_name in ['coarse_loss', 'fine_loss'] else 1
                    loss_dict[loss_name] = loss_val * loss_multiplier
                losses.update(list(loss_dict.values()))

            if args.distributed:
                torch.cuda.synchronize()

            n_itr = epoch * n_batches + idx

            if config.wandb.enable and args.local_rank == 0:
                # wandb.log({'Loss/Train/Batch/Sparse': sparse_loss.item() * loss_multiplier}, step=epoch, commit=False)
                # wandb.log({'Loss/Train/Batch/Dense': dense_loss.item() * loss_multiplier}, step=epoch, commit=False)
                pass
            elif train_writer is not None:
                # train_writer.add_scalar('Loss/Batch/Sparse', sparse_loss.item() * loss_multiplier, n_itr)
                # train_writer.add_scalar('Loss/Batch/Dense', dense_loss.item() * loss_multiplier, n_itr)
                pass

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            if epoch % (args.train_vis_freq//10) == 0 and idx % args.train_vis_freq == 0:
                input_pc = partial[0].squeeze().detach().cpu().numpy()
                sparse_pc = coarse_points[0].squeeze().detach().cpu().numpy()
                dense_pc = dense_points[0].squeeze().detach().cpu().numpy()
                gt_pc = gt[0].squeeze().detach().cpu().numpy()

                # Colorize point cloud by adding labels ([1,14])
                # GT - 1, Coarse - 2, Dense - 3, Center - 4, Input - 14
                colored_input_pc = colorized_pc(input_pc, color=14)
                colored_gt_pc = colorized_pc(gt_pc, color=1)
                colored_sparse_pc = colorized_pc(sparse_pc, color=2)
                colored_dense_pc = colorized_pc(dense_pc, color=3)

                # GT and prediction
                gt_coarse = np.concatenate((colored_sparse_pc, colored_gt_pc), axis=0)
                gt_dense = np.concatenate((colored_dense_pc, colored_gt_pc), axis=0)

                if skel_xyz is not None:
                    skel_pc = skel_xyz[0].squeeze().detach().cpu().numpy()
                    colored_skel_pc = colorized_pc(skel_pc, color=5)

                # Add centers
                if centers is not None:
                    center_pc = centers[0].squeeze().detach().cpu().numpy()                
                    colored_center_pc = colorized_pc(center_pc, color=4)
                    input_pc = np.concatenate((colored_center_pc, colored_input_pc), axis=0)
                    gt_pc = np.concatenate((colored_center_pc, colored_gt_pc), axis=0)
                    sparse_pc = np.concatenate((colored_center_pc, colored_sparse_pc), axis=0)
                    dense_pc = np.concatenate((colored_center_pc, colored_dense_pc), axis=0)

                    if skel_xyz is not None:
                        skel_pc = np.concatenate((colored_center_pc, colored_skel_pc), axis=0)

                if config.wandb.enable and args.local_rank == 0:
                    wandb.log({f'Visualization/Train/Model_{idx:03d}/Input': [wandb.Object3D(input_pc)]}, commit=False)
                    wandb.log({f'Visualization/Train/Model_{idx:03d}/GT': [wandb.Object3D(gt_pc)]}, commit=False)
                    wandb.log({f'Visualization/Train/Model_{idx:03d}/Sparse': [wandb.Object3D(sparse_pc)]}, commit=False)
                    wandb.log({f'Visualization/Train/Model_{idx:03d}/Dense': [wandb.Object3D(dense_pc)]}, commit=False)
                    
                    wandb.log({f'Visualization/Train/Model_{idx:03d}/GT-Sparse': [wandb.Object3D(gt_coarse)]}, commit=False)
                    wandb.log({f'Visualization/Train/Model_{idx:03d}/GTDense': [wandb.Object3D(gt_dense)]}, commit=False)

                    if skel_xyz is not None:
                        wandb.log({f'Visualization/Train/Model_{idx:03d}/GT-Skelton': [wandb.Object3D(skel_pc)]}, commit=False)


            if idx % 100 == 0:
                print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %s G_lr = %.6f D_lr = %.6f' %
                            (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                            ['%.4f' % l for l in losses.val()], optimizer.param_groups[0]['lr'], optimizer_D.param_groups[0]['lr']), logger = logger)

            if config.scheduler.type == 'GradualWarmup':
                if n_itr < config.scheduler.kwargs_2.total_epoch:
                    scheduler.step()
                    scheduler_D.step()

        if isinstance(scheduler, list):
            for item in scheduler:
                item.step()
        else:
            scheduler.step()

        if isinstance(scheduler_D, list):
            for item in scheduler_D:
                item.step()
        else:
            scheduler_D.step()

        epoch_end_time = time.time()

        if isinstance(scheduler, list):
            scheduler_name = scheduler[0].__class__.__name__
            last_lr = scheduler[0].get_last_lr()[-1]           # the first value is for no-decayed params
        else:
            scheduler_name = scheduler.__class__.__name__
            last_lr = scheduler.get_last_lr()[0]

        if isinstance(scheduler_D, list):
            scheduler_name_D = scheduler_D[0].__class__.__name__
            last_lr_D = scheduler_D[0].get_last_lr()[-1]           # the first value is for no-decayed params
        else:
            scheduler_name_D = scheduler_D.__class__.__name__
            last_lr_D = scheduler_D.get_last_lr()[0]

        if config.wandb.enable and args.local_rank == 0:
            for i, loss_name in enumerate(losses.items):
                wandb.log({f'Loss/Train/Epoch/{loss_name}': losses.avg(i)}, step=epoch, commit=False)
            wandb.log({f'Misc/Epoch/G_{scheduler_name}/LR': last_lr}, step=epoch, commit=False)
            wandb.log({f'Misc/Epoch/D_{scheduler_name_D}/LR': last_lr_D}, step=epoch, commit=False if epoch % args.val_freq == 0 else True)
        elif train_writer is not None:
            # train_writer.add_scalar(f'Loss/Epoch/{config.loss.consider_metric}', losses.avg(0), epoch)
            # train_writer.add_scalar(f'Loss/Epoch/{config.loss.consider_metric}', losses.avg(1), epoch)
            # train_writer.add_scalar(f'Misc/Epoch/{scheduler_name}', last_lr, epoch)
            pass

        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s' %
            (epoch,  epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()]), logger = logger)

        if epoch % args.val_freq == 0:
            # Validate the current model
            metrics = validate(base_model, discriminator, test_dataloader, epoch, eval_criterion, val_writer, args, config, logger=logger)

            # Save ckeckpoints
            if  metrics.better_than(best_metrics):
                best_metrics = metrics
                builder.save_checkpoint(base_model, optimizer, discriminator, optimizer_D, epoch, metrics, best_metrics, 'ckpt-best', args, logger = logger)
        builder.save_checkpoint(base_model, optimizer, discriminator, optimizer_D, epoch, metrics, best_metrics, 'ckpt-last', args, logger = logger)      
        if (config.max_epoch - epoch) < 2:
            builder.save_checkpoint(base_model, optimizer, discriminator, optimizer_D, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}', args, logger = logger)     
    if train_writer is not None and val_writer is not None:
        train_writer.close()
        val_writer.close()

def validate(base_model, discriminator, test_dataloader, epoch, eval_criterion, val_writer, args, config, logger = None):
    print_log(f"[VALIDATION] Start validating epoch {epoch}", logger = logger)
    base_model.eval()  # set model to eval mode
    discriminator.eval()

    npoints = config.dataset.val._base_.N_POINTS
    dataset_name = config.dataset.val._base_.NAME

    category_losses = []
    for loss_name in eval_criterion.keys():
        if loss_name == 'CPC':
            category_losses.append('SparseP2PVarLoss')
            category_losses.append('SparseP2SMeanLoss')
            category_losses.append('DenseP2PVarLoss')
            category_losses.append('DenseP2SMeanLoss')
        elif loss_name == 'Adversarial':
            category_losses.append('GeneratorLoss')
            category_losses.append('DiscriminatorLoss')
        else:
            category_losses.append(f'SparseLoss{loss_name}')
            category_losses.append(f'DenseLoss{loss_name}')

    category_losses.append(f'PreSkelLoss')
    category_losses.append(f'SkelLoss')
    if dataset_name == 'PCNSkel':
        category_losses.append(f'SupervisedSkelLoss')

    test_losses = AverageMeter(category_losses)
    test_metrics = AverageMeter(Metrics.names())
    category_metrics = dict()
    n_samples = len(test_dataloader) # bs is 1

    interval =  n_samples // 5

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            taxonomy_id = taxonomy_ids[0] if isinstance(taxonomy_ids[0], str) else taxonomy_ids[0].item()
            model_id = model_ids[0]

            centers = None
            center_radii = None
            center_dirs = None

            if dataset_name == 'PCN' or dataset_name == 'Completion3D' or dataset_name == 'Projected_ShapeNet':
                partial = data[0].cuda()
                gt = data[1].cuda()
            elif dataset_name == 'ShapeNet':
                gt = data.cuda()
                partial, _ = misc.seprate_point_cloud(gt, npoints, [int(npoints * 1/4) , int(npoints * 3/4)], fixed_points = None)
                partial = partial.cuda()
            elif dataset_name == 'PCNSkel':
                partial = data['partial'].cuda()
                gt = data['gt'].cuda()
                centers = data['centers'].cuda()
                center_radii = data['center_radii'].cuda()
                center_dirs = data['center_directions'].cuda()

                if len(center_radii.shape) == 2:
                    center_radii = center_radii.unsqueeze(-1)
                assert len(centers.shape) == len(center_radii.shape) == len(center_dirs.shape) == 3, "Shape Error"
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

            ret = base_model(partial)
            coarse_points = ret[0]
            dense_points = ret[1]
            skel_xyz = ret[2] if len(ret) > 2 else None
            skel_r = ret[3] if len(ret) > 2 else None

            fake = discriminator(dense_points)   # bs num_seed
            real = discriminator(gt)  # bs num_seed

            loss_dict = {}
            for loss_name, loss_func in eval_criterion.items():
                if loss_name == 'CPC':
                    if centers is not None and center_radii is not None and center_dirs is not None:
                        loss_dict['SparseP2PVarLoss'], loss_dict['SparseP2SMeanLoss'] = loss_func(coarse_points, centers, center_radii)
                        loss_dict['DenseP2PVarLoss'], loss_dict['DenseP2SMeanLoss'] = loss_func(dense_points, centers, center_radii)
                    else:
                        loss_dict['SparseP2PVarLoss'] = loss_dict['DenseP2PVarLoss'] = torch.tensor(0., device=coarse_points.device)
                        loss_dict['SparseP2SMeanLoss'] = loss_dict['DenseP2SMeanLoss'] = torch.tensor(0., device=coarse_points.device)
                elif loss_name == 'Repulsion':
                    loss_dict[f'SparseLoss{loss_name}'] = loss_func(coarse_points)
                    loss_dict[f'DenseLoss{loss_name}'] = loss_func(dense_points)
                elif loss_name == 'Adversarial':
                    loss_dict['GeneratorLoss'] = loss_func(fake, 1.0) * 0.01
                    loss_dict['DiscriminatorLoss'] = loss_func(real, 1.0) + loss_func(fake, 0.0)
                else:
                    loss_dict[f'SparseLoss{loss_name}'] = loss_func(coarse_points, gt)
                    loss_dict[f'DenseLoss{loss_name}'] = loss_func(dense_points, gt)

            loss_dict['PreSkelLoss'] = compute_loss_pre(gt, skel_xyz)
            loss_dict['SkelLoss'] = compute_loss(gt, skel_xyz, skel_r, w1=0.3, w2=0.4)
            if dataset_name == 'PCNSkel':
                loss_skel, loss_skelr = compute_loss_supervision(centers, center_radii, skel_xyz, skel_r)
                loss_dict['SupervisedSkelLoss'] = loss_skel + loss_skelr

            if args.distributed:            
                reduce_loss_list = []
                for loss_name, loss_val in loss_dict.items():
                    # loss_multiplier = 1000 if 'L1' in loss_name or 'L2' in loss_name else 1
                    loss_multiplier = 1
                    reduce_loss_list.append(dist_utils.reduce_tensor(loss_val, args).item() * loss_multiplier)
                test_losses.update(reduce_loss_list)
            else:
                for loss_name, loss_val in loss_dict.items():
                    # loss_multiplier = 1000 if 'L1' in loss_name or 'L2' in loss_name else 1
                    loss_dict[loss_name] = loss_val * loss_multiplier
                test_losses.update(list(loss_dict.values()))

            # dense_points_all = dist_utils.gather_tensor(dense_points, args)
            # gt_all = dist_utils.gather_tensor(gt, args)

            # _metrics = Metrics.get(dense_points_all, gt_all)
            _metrics = Metrics.get(dense_points, gt)
            if args.distributed:
                _metrics = [dist_utils.reduce_tensor(_metric, args).item() for _metric in _metrics]
            else:
                _metrics = [_metric.item() for _metric in _metrics]

            for _taxonomy_id in taxonomy_ids:
                if _taxonomy_id not in category_metrics:
                    category_metrics[_taxonomy_id] = AverageMeter(Metrics.names())
                category_metrics[_taxonomy_id].update(_metrics)

            if idx % args.vis_freq == 0:
                input_pc = partial.squeeze().detach().cpu().numpy()
                sparse_pc = coarse_points.squeeze().detach().cpu().numpy()
                dense_pc = dense_points.squeeze().detach().cpu().numpy()
                gt_pc = gt.squeeze().detach().cpu().numpy()

                # Colorize point cloud by adding labels ([1,14])
                # GT - 1, Coarse - 2, Dense - 3, Center - 4
                colored_input_pc = colorized_pc(input_pc, color=14)
                colored_gt_pc = colorized_pc(gt_pc, color=1)
                colored_sparse_pc = colorized_pc(sparse_pc, color=2)
                colored_dense_pc = colorized_pc(dense_pc, color=3)

                # GT and prediction
                gt_coarse = np.concatenate((colored_sparse_pc, colored_gt_pc), axis=0)
                gt_dense = np.concatenate((colored_dense_pc, colored_gt_pc), axis=0)

                if skel_xyz is not None:
                    skel_pc = skel_xyz.squeeze().detach().cpu().numpy()
                    colored_skel_pc = colorized_pc(skel_pc, color=5)

                # Add centers
                if centers is not None:
                    center_pc = centers.squeeze().detach().cpu().numpy()                
                    colored_center_pc = colorized_pc(center_pc, color=4)
                    input_pc = np.concatenate((colored_center_pc, colored_input_pc), axis=0)
                    gt_pc = np.concatenate((colored_center_pc, colored_gt_pc), axis=0)
                    sparse_pc = np.concatenate((colored_center_pc, colored_sparse_pc), axis=0)
                    dense_pc = np.concatenate((colored_center_pc, colored_dense_pc), axis=0)

                    if skel_xyz is not None:
                        skel_pc = np.concatenate((colored_center_pc, colored_skel_pc), axis=0)

                if config.wandb.enable and args.local_rank == 0:
                    wandb.log({f'Visualization/Validation/Model_{idx:03d}/Input_{model_id}': [wandb.Object3D(input_pc)]}, commit=False)
                    wandb.log({f'Visualization/Validation/Model_{idx:03d}/GT_{model_id}': [wandb.Object3D(gt_pc)]}, commit=False)
                    wandb.log({f'Visualization/Validation/Model_{idx:03d}/Sparse': [wandb.Object3D(sparse_pc)]}, commit=False)
                    wandb.log({f'Visualization/Validation/Model_{idx:03d}/Dense': [wandb.Object3D(dense_pc)]}, commit=False)
                    
                    wandb.log({f'Visualization/Validation/Model_{idx:03d}/GT-Sparse': [wandb.Object3D(gt_coarse)]}, commit=False)
                    wandb.log({f'Visualization/Validation/Model_{idx:03d}/GTDense': [wandb.Object3D(gt_dense)]}, commit=False)

                    if skel_xyz is not None:
                        wandb.log({f'Visualization/Validation/Model_{idx:03d}/GT-Skelton': [wandb.Object3D(skel_pc)]}, commit=False)

                elif val_writer is not None:
                    # input_pc = misc.get_ptcloud_img(input_pc)
                    # val_writer.add_image(f'Model_{idx:03d}/Input' , input_pc, epoch, dataformats='HWC')

                    # sparse_img = misc.get_ptcloud_img(sparse)
                    # val_writer.add_image(f'Model_{idx:03d}/Sparse', sparse_img, epoch, dataformats='HWC')

                    # dense_img = misc.get_ptcloud_img(dense)
                    # val_writer.add_image(f'Model_{idx:03d}/Dense', dense_img, epoch, dataformats='HWC')
                    
                    # gt_ptcloud_img = misc.get_ptcloud_img(gt_ptcloud)
                    # val_writer.add_image(f'Model_{idx:03d}/DenseGT', gt_ptcloud_img, epoch, dataformats='HWC')
                    pass
                
            if (idx+1) % interval == 0:
                print_log('Test[%d/%d] Taxonomy = %s Sample = %s Losses = %s Metrics = %s' %
                            (idx + 1, n_samples, taxonomy_id, model_id, ['%.4f' % l for l in test_losses.val()], 
                            ['%.4f' % m for m in _metrics]), logger=logger)
        for _,v in category_metrics.items():
            test_metrics.update(v.avg())
        print_log('[Validation] EPOCH: %d  Metrics = %s' % (epoch, ['%.4f' % m for m in test_metrics.avg()]), logger=logger)

        if args.distributed:
            torch.cuda.synchronize()
     
    # Print testing results
    shapenet_dict = json.load(open('./data/shapenet_synset_dict.json', 'r'))
    print_log('============================ TEST RESULTS ============================',logger=logger)
    msg = ''
    msg += 'Taxonomy\t'
    msg += '#Sample\t'
    for metric in test_metrics.items:
        msg += metric + '\t'
    msg += '#ModelName\t'
    print_log(msg, logger=logger)

    for taxonomy_id in category_metrics:
        msg = ''
        msg += (taxonomy_id + '\t')
        msg += (str(category_metrics[taxonomy_id].count(0)) + '\t')
        for value in category_metrics[taxonomy_id].avg():
            msg += '%.3f \t' % value
        msg += shapenet_dict[taxonomy_id] + '\t'
        print_log(msg, logger=logger)

    msg = ''
    msg += 'Overall\t\t'
    for value in test_metrics.avg():
        msg += '%.3f \t' % value
    print_log(msg, logger=logger)

    # Add testing results to TensorBoard
    if config.wandb.enable and args.local_rank == 0:
        # enumerate(test_losses.items) causes order unmatched
        for i, loss_name in enumerate(category_losses):
            wandb.log({f'Loss/Validation/Epoch/{loss_name}': test_losses.avg(i)}, step=epoch, commit=False)
        for i, metric in enumerate(test_metrics.items):
            commit = True if i == len(test_metrics.items) - 1 else False
            wandb.log({f'Metric/Validation/{metric}': test_metrics.avg(i)}, step=epoch, commit=commit)
    elif val_writer is not None:
        # for i, loss_name in enumerate(test_losses.items):
        #     val_writer.add_scalar(f'Loss/Epoch/{loss_name}', test_losses.avg(i), epoch)
        # for i, metric in enumerate(test_metrics.items):
        #     val_writer.add_scalar(f'Metric/{metric}', test_metrics.avg(i), epoch)
        pass

    return Metrics(config.loss.consider_metric, test_metrics.avg())


crop_ratio = {
    'easy': 1/4,
    'median' :1/2,
    'hard':3/4
}

def test_net(args, config):
    logger = get_logger(args.log_name)
    print_log('Tester start ... ', logger = logger)
    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)
 
    base_model = builder.model_builder(config)
    discriminator = builder.model_builder(config.model.discriminator)
    # load checkpoints
    builder.load_model(base_model, discriminator, args.ckpts, logger = logger)
    if args.use_gpu:
        base_model.to(args.local_rank)

    #  DDP    
    if args.distributed:
        raise NotImplementedError()

    # Criterion
    ChamferDisL1 = ChamferDistanceL1()
    ChamferDisL2 = ChamferDistanceL2()
    DensityAwareChamferDis = DensityAwareChamferDistance()

    test(base_model, test_dataloader, ChamferDisL1, ChamferDisL2, DensityAwareChamferDis, args, config, logger=logger)

def test(base_model, test_dataloader, ChamferDisL1, ChamferDisL2, DensityAwareChamferDis, args, config, logger = None):

    base_model.eval()  # set model to eval mode

    test_losses = AverageMeter(['SparseLossL1', 'SparseLossL2', 'SparseLossDCD', 'DenseLossL1', 'DenseLossL2', 'DenseLossDCD'])
    test_metrics = AverageMeter(Metrics.names())
    category_metrics = dict()
    n_samples = len(test_dataloader) # bs is 1

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            taxonomy_id = taxonomy_ids[0] if isinstance(taxonomy_ids[0], str) else taxonomy_ids[0].item()
            model_id = model_ids[0]

            npoints = config.dataset.test._base_.N_POINTS
            dataset_name = config.dataset.test._base_.NAME
            if dataset_name == 'PCN' or dataset_name == 'Projected_ShapeNet':
                partial = data[0].cuda()
                gt = data[1].cuda()

                ret = base_model(partial)
                coarse_points = ret[0]
                dense_points = ret[1]
                skel_xyz = ret[2] if len(ret) > 2 else None
                skel_r = ret[3] if len(ret) > 2 else None

                sparse_loss_l1 =  ChamferDisL1(coarse_points, gt)
                sparse_loss_l2 =  ChamferDisL2(coarse_points, gt)
                sparse_loss_dcd = DensityAwareChamferDis(coarse_points, gt)
                dense_loss_l1 =  ChamferDisL1(dense_points, gt)
                dense_loss_l2 =  ChamferDisL2(dense_points, gt)
                dense_loss_dcd = DensityAwareChamferDis(dense_points, gt)

                test_losses.update([sparse_loss_l1.item() * 1000, sparse_loss_l2.item() * 1000, sparse_loss_dcd.item(),
                                    dense_loss_l1.item() * 1000, dense_loss_l2.item() * 1000, dense_loss_dcd.item()])

                _metrics = Metrics.get(dense_points, gt, require_emd=True)
                # test_metrics.update(_metrics)

                if taxonomy_id not in category_metrics:
                    category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
                category_metrics[taxonomy_id].update(_metrics)

            elif dataset_name == 'ShapeNet':
                gt = data.cuda()
                choice = [torch.Tensor([1,1,1]),torch.Tensor([1,1,-1]),torch.Tensor([1,-1,1]),torch.Tensor([-1,1,1]),
                            torch.Tensor([-1,-1,1]),torch.Tensor([-1,1,-1]), torch.Tensor([1,-1,-1]),torch.Tensor([-1,-1,-1])]
                num_crop = int(npoints * crop_ratio[args.mode])
                for item in choice:           
                    partial, _ = misc.seprate_point_cloud(gt, npoints, num_crop, fixed_points = item)
                    # NOTE: subsample the input
                    partial = misc.fps(partial, 2048)
                    ret = base_model(partial)
                    coarse_points = ret[0]
                    dense_points = ret[-1]

                    sparse_loss_l1 =  ChamferDisL1(coarse_points, gt)
                    sparse_loss_l2 =  ChamferDisL2(coarse_points, gt)
                    sparse_loss_dcd = DensityAwareChamferDis(coarse_points, gt)
                    dense_loss_l1 =  ChamferDisL1(dense_points, gt)
                    dense_loss_l2 =  ChamferDisL2(dense_points, gt)
                    dense_loss_dcd = DensityAwareChamferDis(dense_points, gt)

                    test_losses.update([sparse_loss_l1.item() * 1000, sparse_loss_l2.item() * 1000, sparse_loss_dcd.item(),
                                        dense_loss_l1.item() * 1000, dense_loss_l2.item() * 1000, dense_loss_dcd.item()])
                    
                    _metrics = Metrics.get(dense_points ,gt)



                    if taxonomy_id not in category_metrics:
                        category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
                    category_metrics[taxonomy_id].update(_metrics)
            elif dataset_name == 'KITTI':
                partial = data.cuda()
                ret = base_model(partial)
                dense_points = ret[-1]
                target_path = os.path.join(args.experiment_path, 'vis_result')
                if not os.path.exists(target_path):
                    os.mkdir(target_path)
                misc.visualize_KITTI(
                    os.path.join(target_path, f'{model_id}_{idx:03d}'),
                    [partial[0].cpu(), dense_points[0].cpu()]
                )
                continue
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

            if (idx+1) % 5 == 0:
                print_log('Test[%d/%d] Taxonomy = %s Sample = %s Losses = %s Metrics = %s' %
                            (idx + 1, n_samples, taxonomy_id, model_id, ['%.4f' % l for l in test_losses.val()], 
                            ['%.4f' % m for m in _metrics]), logger=logger)
        if dataset_name == 'KITTI':
            return
        for _,v in category_metrics.items():
            test_metrics.update(v.avg())
        print_log('[TEST] Metrics = %s' % (['%.4f' % m for m in test_metrics.avg()]), logger=logger)

     

    # Print testing results
    shapenet_dict = json.load(open('./data/shapenet_synset_dict.json', 'r'))
    print_log('============================ TEST RESULTS ============================',logger=logger)
    msg = ''
    msg += 'Taxonomy\t'
    msg += '#Sample\t'
    for metric in test_metrics.items:
        msg += metric + '\t'
    msg += '#ModelName\t'
    print_log(msg, logger=logger)


    for taxonomy_id in category_metrics:
        msg = ''
        msg += (taxonomy_id + '\t')
        msg += (str(category_metrics[taxonomy_id].count(0)) + '\t')
        for value in category_metrics[taxonomy_id].avg():
            msg += '%.3f \t' % value
        msg += shapenet_dict[taxonomy_id] + '\t'
        print_log(msg, logger=logger)

    msg = ''
    msg += 'Overall \t\t'
    for value in test_metrics.avg():
        msg += '%.3f \t' % value
    print_log(msg, logger=logger)
    return 
