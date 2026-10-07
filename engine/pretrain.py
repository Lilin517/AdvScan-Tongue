import os
import time
import math
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import autocast, GradScaler
from data.pretrain_dataset import UnlabeledImageDataset
from data.pretrain_transforms import DINOMultiCrop
from models.dino_module import DINOModule
from losses.dino_loss import DINOLoss
from models.registry import get_backbone_info
from utils.checkpoint import save_checkpoint
from utils.logger import Logger, TimeTracker, is_rank0, get_rank, get_world_size

def _cosine_scheduler(base_lr, warmup_epochs, total_epochs, iters_per_epoch, start_warmup_val=0.0):
    warmup_iters = warmup_epochs * iters_per_epoch
    total_iters = total_epochs * iters_per_epoch

    def schedule(step):
        if step < warmup_iters:
            frac = step / max(1, warmup_iters)
            return start_warmup_val + (1.0 - start_warmup_val) * frac
        else:
            progress = (step - warmup_iters) / max(1, total_iters - warmup_iters)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
    return schedule

def dino_collate_fn(batch):
    global_crops = [item[0] for item in batch]
    local_crops = [item[1] for item in batch]
    n_global = len(global_crops[0])
    n_local = len(local_crops[0])
    global_stacked = [torch.stack([gc[i] for gc in global_crops]) for i in range(n_global)]
    local_stacked = [torch.stack([lc[i] for lc in local_crops]) for i in range(n_local)]
    return (global_stacked, local_stacked)

def _setup_ddp(local_rank: int, world_size: int):
    if world_size > 1 and (not dist.is_initialized()):
        os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
        os.environ.setdefault('MASTER_PORT', str(29500 + os.getpid() % 1000))
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(local_rank)

def run_pretrain(backbone_name: str, stage, version: str, device: torch.device, output_dir: str, cfg, dataset_root: str=None, local_rank: int=0, world_size: int=1, pretrain_ckpt_path: str=None):
    use_ddp = world_size > 1
    if use_ddp:
        _setup_ddp(local_rank, world_size)
    rank = get_rank()
    info = get_backbone_info(backbone_name)
    ssl_method = 'dino'
    stage_display = stage if isinstance(stage, (list, tuple)) else [stage]
    log = Logger(output_dir, f'{backbone_name}_pretrain')
    if rank == 0:
        log.log(f'{'=' * 60}')
        log.log(f'Pretrain: {backbone_name} | stage={stage_display} | version={version}')
        log.log(f'SSL method: {ssl_method.upper()}  Device: {device}  GPUs: {world_size}')
        log.log(f'Output: {output_dir}')
        grad_accum_steps = cfg.get('pretrain_grad_accum_steps', 1)
        eff_bs = cfg.pretrain_batch_size * grad_accum_steps * world_size
        log.log(f'Config: epochs={cfg.pretrain_epochs} per_gpu_bs={cfg.pretrain_batch_size} grad_accum={grad_accum_steps} (effective_bs={eff_bs})')
        log.log(f'MyBlock init: FULL zero-initialization (my_work_3)')
        log.log(f'MyBlock positions: {len(stage_display)} insertion points across 12 blocks')
        log.log(f'{'=' * 60}')
        if device.type == 'cuda':
            gpu_name = torch.cuda.get_device_name(device)
            gpu_mem = torch.cuda.get_device_properties(device).total_memory / 1024 ** 3
            log.log(f'GPU[{rank}]: {gpu_name} ({gpu_mem:.1f} GB)')
    if dataset_root is None:
        dataset_root = cfg.get('pretrain_dataset_root', 'dataset/mae_pretrain')
    transform = DINOMultiCrop(global_crops_size=cfg.get('input_size', 224), local_crops_size=cfg.get('dino_local_crops_size', 96), global_crops_scale=cfg.get('dino_global_crops_scale', [0.4, 1.0]), local_crops_scale=cfg.get('dino_local_crops_scale', [0.05, 0.4]), local_crops_number=cfg.get('dino_local_crops_number', 6), resize_local_to_global=True)
    dataset = UnlabeledImageDataset(root_dir=dataset_root, transform=transform)
    if rank == 0:
        log.log(f'Dataset: {len(dataset)} images from {dataset_root}')
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if use_ddp else None
    loader = DataLoader(dataset, batch_size=cfg.pretrain_batch_size, shuffle=sampler is None, sampler=sampler, num_workers=cfg.get('num_workers', 8), pin_memory=cfg.get('pin_memory', True), prefetch_factor=cfg.get('prefetch_factor', 4), persistent_workers=cfg.get('persistent_workers', True) and cfg.get('num_workers', 8) > 0, drop_last=True, collate_fn=dino_collate_fn)
    iters_per_epoch = len(loader)
    if rank == 0:
        log.log(f'Batches/epoch: {iters_per_epoch} (per GPU)')
    time_tracker = TimeTracker(cfg.pretrain_epochs, iters_per_epoch)
    module = DINOModule(backbone_name, stage, version, cfg, pretrain_ckpt_path=pretrain_ckpt_path).to(device)
    loss_fn = DINOLoss(n_prototypes=cfg.get('dino_n_prototypes', 65536), teacher_temp=cfg.get('dino_teacher_temp', 0.04), student_temp=cfg.get('dino_student_temp', 0.1), center_momentum=cfg.get('dino_center_momentum', 0.9)).to(device)
    if rank == 0:
        log.log(module.student_backbone.architecture_summary())
        for i, mb in enumerate(module.student_backbone.my_blocks):
            mb_params = sum((p.numel() for p in mb.parameters()))
            log.log(f'My_Block[{i}]: {mb.__class__.__name__}  params={mb_params:,}  (FULL zero-init)')
    if use_ddp:
        module = DDP(module, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        loss_fn = loss_fn.to(device)
        if rank == 0:
            log.log('Model wrapped with DistributedDataParallel (DDP)')
    backbone_params = module.student_backbone.parameters() if not use_ddp else module.module.student_backbone.parameters()
    head_params = module.student_head.parameters() if not use_ddp else module.module.student_head.parameters()
    param_groups = [{'params': list(backbone_params), 'lr': cfg.pretrain_lr * cfg.get('backbone_lr_scale', 0.1)}, {'params': list(head_params), 'lr': cfg.pretrain_lr}]
    optimizer = torch.optim.AdamW(param_groups, lr=cfg.pretrain_lr, weight_decay=cfg.get('pretrain_wd', 0.05))
    lr_lambda = _cosine_scheduler(base_lr=cfg.pretrain_lr, warmup_epochs=cfg.warmup_epochs, total_epochs=cfg.pretrain_epochs, iters_per_epoch=iters_per_epoch)
    amp_enabled = cfg.get('amp', True)
    scaler = GradScaler('cuda', enabled=amp_enabled) if device.type == 'cuda' else GradScaler(enabled=False)
    global_step = 0
    best_loss = float('inf')
    start_time = time.time()
    nan_steps_total = 0
    grad_accum_steps = cfg.get('pretrain_grad_accum_steps', 1)
    for epoch in range(cfg.pretrain_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        module.train()
        epoch_loss = 0.0
        epoch_steps = 0
        epoch_nan = 0
        time_tracker.start_epoch()
        momentum = cfg.get('dino_teacher_momentum', 0.996) + (cfg.get('dino_momentum_end', 1.0) - cfg.get('dino_teacher_momentum', 0.996)) * (math.cos(math.pi * epoch / cfg.pretrain_epochs) + 1) / 2
        m = module.module if use_ddp else module
        m.current_momentum = momentum
        for batch_idx, batch in enumerate(loader):
            global_step += 1
            lr_scale = lr_lambda(global_step)
            for pg in optimizer.param_groups:
                pg['lr'] = cfg.pretrain_lr * lr_scale
            global_crops = [g.to(device, non_blocking=True) for g in batch[0]]
            local_crops = [l.to(device, non_blocking=True) for l in batch[1]]
            with autocast('cuda', enabled=amp_enabled):
                teacher_logits, student_logits = module(global_crops, local_crops)
                loss = loss_fn(teacher_logits, student_logits)
            if not torch.isfinite(loss):
                epoch_nan += 1
                nan_steps_total += 1
                if rank == 0:
                    log.log(f'  ⚠ E{epoch + 1:3d}/{cfg.pretrain_epochs} [{batch_idx:4d}/{iters_per_epoch}] loss={loss.item():.4f} — skipping update')
                continue
            loss = loss / grad_accum_steps
            scaler.scale(loss).backward()
            if (batch_idx + 1) % grad_accum_steps == 0 or batch_idx == iters_per_epoch - 1:
                if cfg.get('grad_clip', 3.0) > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(module.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            m._momentum_update_teacher(momentum)
            epoch_loss += loss.item()
            epoch_steps += 1
            if rank == 0 and batch_idx % cfg.print_freq == 0:
                time_tracker.update_batch(batch_idx, epoch)
                eta_str = time_tracker.eta(batch_idx, epoch)
                lr_now = optimizer.param_groups[0]['lr']
                log.log(f'  E{epoch + 1:3d}/{cfg.pretrain_epochs} [{batch_idx:4d}/{iters_per_epoch}] loss={loss.item():.4f}  lr={lr_now:.6f}  mom={momentum:.4f}  ETA={eta_str}')
                log.add_scalar('train/loss', loss.item(), global_step)
                log.add_scalar('train/lr', lr_now, global_step)
                log.add_scalar('train/momentum', momentum, global_step)
        if use_ddp:
            loss_tensor = torch.tensor([epoch_loss, float(epoch_steps)], device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            epoch_loss = loss_tensor[0].item()
            epoch_steps = int(loss_tensor[1].item())
        avg_loss = epoch_loss / max(1, epoch_steps)
        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss
        if rank == 0:
            log.add_scalar('train/epoch_loss', avg_loss, epoch)
            if (epoch + 1) % 10 == 0 or is_best or epoch == cfg.pretrain_epochs - 1:
                m = module.module if use_ddp else module
                backbone_sd = m.student_backbone.state_dict()
                save_checkpoint(state={'epoch': epoch + 1, 'backbone_state_dict': backbone_sd, 'optimizer_state_dict': optimizer.state_dict(), 'loss': avg_loss, 'best_loss': best_loss, 'backbone_name': backbone_name, 'stage': stage, 'version': version, 'ssl_method': ssl_method}, save_dir=os.path.join(output_dir, 'checkpoints'), filename=f'epoch_{epoch + 1:03d}.pth', is_best=is_best)
            nan_msg = f'  ⚠ {epoch_nan} NaN steps' if epoch_nan > 0 else ''
            eta_str = time_tracker.eta(iters_per_epoch - 1, epoch)
            elapsed = time.time() - start_time
            log.log(f'  >>> Epoch {epoch + 1}/{cfg.pretrain_epochs} avg_loss={avg_loss:.4f} best_loss={best_loss:.4f} ETA={eta_str}  elapsed={elapsed / 60:.1f}min{nan_msg}')
        if use_ddp:
            dist.barrier()
    total_time = time.time() - start_time
    final_metrics = {'final_loss': avg_loss, 'best_loss': best_loss, 'epochs': cfg.pretrain_epochs, 'ssl_method': ssl_method, 'world_size': world_size, 'train_time_min': total_time / 60, 'nan_steps_total': nan_steps_total}
    if rank == 0:
        nan_total_msg = f'  ⚠ {nan_steps_total} NaN steps total' if nan_steps_total > 0 else ''
        log.log(f'✓ {backbone_name} pretrain done. Best loss: {best_loss:.4f}  Time: {total_time / 60:.1f}min{nan_total_msg}')
        _write_pretrain_summary(output_dir, backbone_name, stage, version, ssl_method, final_metrics)
    if use_ddp:
        dist.barrier()
    log.close()
    return final_metrics

def _write_pretrain_summary(output_dir, backbone_name, stage, version, ssl_method, metrics):
    path = os.path.join(output_dir, 'pretrain_summary.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(f'Backbone:     {backbone_name}\n')
        f.write(f'Stage:        {stage}\n')
        f.write(f'Version:      {version}\n')
        f.write(f'SSL Method:   {ssl_method}\n')
        f.write(f'Best Loss:    {metrics.get('best_loss', 0):.4f}\n')
        f.write(f'Final Loss:   {metrics.get('final_loss', 0):.4f}\n')
        f.write(f'Epochs:       {metrics.get('epochs', '?')}\n')
        f.write(f'GPUs:         {metrics.get('world_size', 1)}\n')
        f.write(f'Train Time:   {metrics.get('train_time_min', 0):.1f} min\n')
        f.write(f'NaN Steps:    {metrics.get('nan_steps_total', 0)}\n')
