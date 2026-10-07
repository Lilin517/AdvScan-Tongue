import os
import sys
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import autocast, GradScaler
from data.dataset import COCOMultiLabelDataset
from data.transforms import MedicalAug
from models.backbone import BackboneWithMyBlock
from models.mlp_head import MLPHead
from utils.checkpoint import save_checkpoint, load_checkpoint, load_model_state
from utils.logger import Logger, TimeTracker, is_rank0, get_rank, get_world_size, all_reduce_mean
from utils.metrics import compute_metrics

def _setup_ddp(local_rank: int, world_size: int):
    if world_size > 1 and (not dist.is_initialized()):
        os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
        os.environ.setdefault('MASTER_PORT', str(29500 + os.getpid() % 1000))
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(local_rank)

def _cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()

def train_one_backbone(backbone_name: str, stage, version: str, mode: str, device: torch.device, output_dir: str, dataset_root: str, overrides: dict=None, resume: bool=False, source_ckpt: str=None, backbone_mode: int=1, local_rank: int=0, world_size: int=1):
    assert mode == 'bce' and backbone_mode == 2 and (version.lower() == 'v2_3') and (list(stage) == list(range(1, 12)))
    use_ddp = world_size > 1
    if use_ddp:
        _setup_ddp(local_rank, world_size)
    rank = get_rank()
    from config import Config
    cfg = Config()
    if overrides:
        cfg.update(overrides)
    stage_display = stage if isinstance(stage, (list, tuple)) else [stage]
    stage_str = ','.join((str(s) for s in stage_display))
    mode_label = {1: 'freeze', 2: 'e2e'}.get(backbone_mode, f'bm{backbone_mode}')
    log = Logger(output_dir, f'{backbone_name}_stage{stage_str}_{mode}_{mode_label}')
    if rank == 0:
        log.log(f'{'=' * 60}')
        log.log(f'My_Work_3 Training: {backbone_name} | stage={stage_display} | version={version} | mode={mode} | backbone_mode={backbone_mode}')
        log.log(f'Device: {device}  GPUs: {world_size}  Output: {output_dir}')
        log.log(f'Config: epochs={cfg.epochs} bs={cfg.batch_size} lr={cfg.lr} warmup={cfg.warmup_epochs}')
        log.log(f'Training strategy: END-TO-END (backbone LR={cfg.lr * cfg.backbone_lr_scale:.6f})')
        log.log(f'MyBlock init: FULL zero-initialization (my_work_3)')
        log.log(f'MyBlock positions: {len(stage_display)} insertion points across 12 blocks')
        log.log(f'{'=' * 60}')
        if device.type == 'cuda':
            gpu_name = torch.cuda.get_device_name(device)
            gpu_mem = torch.cuda.get_device_properties(device).total_memory / 1024 ** 3
            log.log(f'GPU[{rank}]: {gpu_name} ({gpu_mem:.1f} GB)')
    effective_bs = cfg.batch_size
    if use_ddp:
        log.log(f'DDP: per-GPU bs={effective_bs}, global bs={effective_bs * world_size}')
    classes_txt = os.path.join(dataset_root, 'train', 'classes.txt')
    train_dataset = COCOMultiLabelDataset(root_dir=dataset_root, split='train', transform=MedicalAug(cfg.input_size, train=True), classes_txt=classes_txt)
    val_dataset = COCOMultiLabelDataset(root_dir=dataset_root, split='val', transform=MedicalAug(cfg.input_size, train=False), classes_txt=classes_txt)
    num_classes = train_dataset.num_classes
    if rank == 0:
        log.log(f'Data: train={len(train_dataset)} val={len(val_dataset)} classes={num_classes}')
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if use_ddp else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if use_ddp else None
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=train_sampler is None, sampler=train_sampler, num_workers=cfg.num_workers, pin_memory=cfg.pin_memory, prefetch_factor=cfg.get('prefetch_factor', 4), persistent_workers=cfg.get('persistent_workers', True) and cfg.num_workers > 0, drop_last=use_ddp)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, sampler=val_sampler, num_workers=cfg.num_workers, pin_memory=cfg.pin_memory, prefetch_factor=cfg.get('prefetch_factor', 4), persistent_workers=cfg.get('persistent_workers', True) and cfg.num_workers > 0)
    if rank == 0:
        log.log(f'Loading {backbone_name} with My_Block_V{version.upper()} at stage {stage}...')
    backbone = BackboneWithMyBlock(backbone_name, stage=stage, pretrained=True, version=version, pretrain_ckpt_path=source_ckpt)
    backbone = backbone.to(device)
    if rank == 0:
        log.log(backbone.architecture_summary())
        for i, mb in enumerate(backbone.my_blocks):
            mb_params = sum((p.numel() for p in mb.parameters()))
            log.log(f'My_Block[{i}]: {mb.__class__.__name__}  params={mb_params:,}  (FULL zero-init)')
    if source_ckpt and os.path.isfile(source_ckpt):
        pretrain_ckpt = load_checkpoint(source_ckpt, str(device))
        if pretrain_ckpt is not None:
            backbone_sd = pretrain_ckpt.get('backbone_state_dict', pretrain_ckpt)
            loaded = load_model_state(backbone, backbone_sd, strict=False)
            if rank == 0:
                log.log(f'Loaded pretrain weights from: {source_ckpt}')
        elif rank == 0:
            log.log(f'⚠ Could not load pretrain checkpoint: {source_ckpt}')
    elif source_ckpt and rank == 0:
        log.log(f'⚠ Pretrain checkpoint not found: {source_ckpt}')
        log.log(f'  Using ImageNet-pretrained weights only.')
    head = MLPHead(in_dim=backbone.feature_dim, num_classes=num_classes, hidden_dim=cfg.mlp_hidden_dim, dropout=cfg.mlp_dropout).to(device)
    if rank == 0:
        log.log(f'Head: MLP (hidden={cfg.mlp_hidden_dim}, dropout={cfg.mlp_dropout})')
    if use_ddp:
        backbone = DDP(backbone, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        head = DDP(head, device_ids=[local_rank], output_device=local_rank)
        if rank == 0:
            ddp_what = 'backbone + head'
            log.log(f'Model wrapped with DistributedDataParallel (DDP) — {ddp_what}')
    loss_fn = nn.BCEWithLogitsLoss()
    if rank == 0:
        log.log('Loss: BCEWithLogitsLoss')
    backbone_lr = cfg.lr * cfg.backbone_lr_scale
    head_lr = cfg.lr
    param_groups = [{'params': head.parameters(), 'lr': head_lr}, {'params': [p for p in backbone.parameters() if p.requires_grad], 'lr': backbone_lr}]
    optimizer = torch.optim.AdamW(param_groups, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if rank == 0:
        log.log(f'Optimizer groups: head_lr={head_lr:.6f}, backbone_lr={backbone_lr:.6f}')

    def warmup_lambda(epoch):
        if epoch < cfg.warmup_epochs:
            return (epoch + 1) / max(1, cfg.warmup_epochs)
        return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)
    use_amp = cfg.get('amp', True)
    scaler = GradScaler('cuda', enabled=use_amp) if device.type == 'cuda' else GradScaler(enabled=False)
    start_epoch = 0
    best_mAP = 0.0
    last_ckpt_path = os.path.join(output_dir, 'checkpoints', 'last.pth')
    if resume and os.path.isfile(last_ckpt_path):
        ckpt = load_checkpoint(last_ckpt_path, str(device))
        if ckpt is not None:
            bb = backbone.module if hasattr(backbone, 'module') else backbone
            hd = head.module if hasattr(head, 'module') else head
            load_model_state(bb, ckpt.get('backbone_state_dict', ckpt), strict=False)
            head_key = 'classifier_state_dict' if 'classifier_state_dict' in ckpt else 'decoder_state_dict'
            hd.load_state_dict(ckpt.get(head_key, {}), strict=False)
            optimizer.load_state_dict(ckpt.get('optimizer_state_dict', {}))
            start_epoch = ckpt.get('epoch', 0)
            best_mAP = ckpt.get('best_mAP', 0.0)
            if rank == 0:
                log.log(f'Resumed from epoch {start_epoch} (best mAP={best_mAP:.4f})')
    time_tracker = TimeTracker(cfg.epochs, len(train_loader))
    global_step = start_epoch * len(train_loader)
    nan_steps_total = 0
    training_start = time.time()
    for epoch in range(start_epoch, cfg.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        backbone.train()
        head.train()
        epoch_loss = 0.0
        epoch_steps = 0
        epoch_nan = 0
        time_tracker.start_epoch()
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            global_step += 1
            with autocast('cuda', enabled=use_amp):
                feats = backbone(images)
                pooled = feats['cls'] if isinstance(feats, dict) else feats
                logits = head(pooled)
                loss = loss_fn(logits, labels)
            if not torch.isfinite(loss):
                epoch_nan += 1
                nan_steps_total += 1
                if rank == 0:
                    log.log(f'  ⚠ E{epoch + 1}/{cfg.epochs} step={global_step} loss={loss.item():.4f} — skipping update')
                continue
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            if cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                for pg in param_groups:
                    torch.nn.utils.clip_grad_norm_(pg['params'], cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            epoch_steps += 1
            if rank == 0 and global_step % cfg.print_freq == 0:
                time_tracker.update_batch(batch_idx, epoch)
                eta_str = time_tracker.eta(batch_idx, epoch)
                lr_now = optimizer.param_groups[0]['lr']
                log.add_scalar('train/loss', loss.item(), global_step)
                log.log(f'  E{epoch + 1}/{cfg.epochs} [{batch_idx + 1}/{len(train_loader)}] loss={loss.item():.4f} lr={lr_now:.6f} ETA={eta_str}')
        scheduler.step()
        if use_ddp:
            loss_tensor = torch.tensor([epoch_loss, float(epoch_steps)], device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            epoch_loss = loss_tensor[0].item()
            epoch_steps = int(loss_tensor[1].item())
        avg_loss = epoch_loss / max(1, epoch_steps)
        if rank == 0:
            eta_str = time_tracker.eta(len(train_loader) - 1, epoch)
            elapsed = time.time() - training_start
            nan_msg = f'  ⚠ {epoch_nan} NaN steps' if epoch_nan > 0 else ''
            log.log(f'  ── Epoch {epoch + 1}/{cfg.epochs} avg_loss={avg_loss:.4f} ETA={eta_str}  elapsed={elapsed / 60:.1f}min{nan_msg}')
        if (epoch + 1) % cfg.val_freq == 0 or epoch == cfg.epochs - 1:
            val_metrics = _validate(backbone, head, val_loader, device, cfg, mode, loss_fn, backbone_mode, use_ddp)
            if rank == 0:
                log.add_scalar('val/loss', val_metrics.get('val_loss', 0), epoch + 1)
                log.add_scalar('val/mAP', val_metrics.get('mAP', 0), epoch + 1)
                log.log(f'  >>> Val: loss={val_metrics.get('val_loss', 0):.4f} mAP={val_metrics.get('mAP', 0):.4f} F1={val_metrics.get('f1_macro', 0):.4f} (world_size={world_size})')
                is_best = val_metrics.get('mAP', 0) > best_mAP
                if is_best:
                    best_mAP = val_metrics.get('mAP', 0)
                bb = backbone.module if hasattr(backbone, 'module') else backbone
                hd = head.module if hasattr(head, 'module') else head
                ckpt_state = {'epoch': epoch + 1, 'global_step': global_step, 'backbone_state_dict': bb.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'best_mAP': best_mAP, 'val_metrics': val_metrics, 'backbone_name': backbone_name, 'stage': stage, 'version': version, 'mode': mode, 'backbone_mode': backbone_mode}
                ckpt_state['classifier_state_dict'] = hd.state_dict()
                save_checkpoint(state=ckpt_state, save_dir=os.path.join(output_dir, 'checkpoints'), filename='last.pth', is_best=is_best)
                log.log(f'  Checkpoint saved{(' (BEST)' if is_best else '')}')
    if rank == 0:
        total_time = time.time() - training_start
        log.log(f'\n✓ Training done in {total_time / 60:.1f} min')
        log.log(f'  Best mAP: {best_mAP:.4f}' + (f'  ⚠ Total NaN steps: {nan_steps_total}' if nan_steps_total > 0 else ''))
        test_metrics = _evaluate_on_test(backbone, head, device, cfg, dataset_root, output_dir, log, mode, backbone_mode, use_ddp)
        summary = {'backbone': backbone_name, 'stage': stage, 'version': version, 'mode': mode, 'backbone_mode': backbone_mode, 'best_mAP': best_mAP, 'epochs': cfg.epochs, 'world_size': world_size, 'train_time_min': total_time / 60, 'nan_steps_total': nan_steps_total}
        if test_metrics:
            summary.update({'test_mAP': test_metrics.get('mAP', 0), 'test_f1_macro': test_metrics.get('f1_macro', 0), 'test_auc_macro': test_metrics.get('auc_macro', 0)})
        _write_summary(output_dir, summary)
        _write_test_score(mode, backbone_name, stage, version, test_metrics or {}, backbone_mode)
    if use_ddp:
        dist.barrier()
    log.close()
    return {'best_mAP': best_mAP}

def _forward(backbone, head, images, mode, backbone_mode):
    return _forward_impl(backbone, head, images, mode)

def _forward_impl(backbone, head, images, mode):
    bb = backbone.module if hasattr(backbone, 'module') else backbone
    family = bb.family
    feats = backbone(images)
    if isinstance(feats, dict):
        pooled = feats['cls']
    else:
        pooled = feats
    return head(pooled)

def _validate(backbone, head, val_loader, device, cfg, mode, loss_fn, backbone_mode, use_ddp):
    backbone.eval()
    head.eval()
    all_logits = []
    all_labels = []
    total_loss = 0.0
    world_size = get_world_size() if use_ddp else 1
    rank = get_rank() if use_ddp else 0
    if loss_fn is None:
        loss_fn_val = nn.BCEWithLogitsLoss()
    else:
        loss_fn_val = loss_fn
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = _forward(backbone, head, images, mode, backbone_mode)
            loss = loss_fn_val(logits, labels)
            total_loss += loss.item()
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    if use_ddp:
        local_size = torch.tensor([all_logits.shape[0]], device=device)
        sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
        dist.all_gather(sizes, local_size)
        max_size = max((s.item() for s in sizes))
        if all_logits.shape[0] < max_size:
            pad_logits = torch.zeros(max_size, all_logits.shape[1], device=device)
            pad_labels = torch.zeros(max_size, all_labels.shape[1], device=device)
            pad_logits[:all_logits.shape[0]] = all_logits.to(device)
            pad_labels[:all_labels.shape[0]] = all_labels.to(device)
        else:
            pad_logits = all_logits.to(device)
            pad_labels = all_labels.to(device)
        gathered_logits = [torch.zeros_like(pad_logits) for _ in range(world_size)]
        gathered_labels = [torch.zeros_like(pad_labels) for _ in range(world_size)]
        dist.all_gather(gathered_logits, pad_logits)
        dist.all_gather(gathered_labels, pad_labels)
        all_logits = torch.cat([g[:sizes[i].item()].cpu() for i, g in enumerate(gathered_logits)], dim=0)
        all_labels = torch.cat([g[:sizes[i].item()].cpu() for i, g in enumerate(gathered_labels)], dim=0)
        loss_tensor = torch.tensor([total_loss], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        total_loss = loss_tensor.item() / world_size
    probs = torch.sigmoid(all_logits)
    metrics = compute_metrics(probs.numpy(), all_labels.numpy())
    metrics['val_loss'] = total_loss / max(1, len(val_loader))
    return metrics

def _evaluate_on_test(backbone, head, device, cfg, dataset_root, output_dir, log, mode, backbone_mode, use_ddp):
    test_ann = os.path.join(dataset_root, 'test', 'annotations', 'test.json')
    if not os.path.isfile(test_ann):
        log.log('  ⚠ Test annotations not found — skipping test evaluation.')
        return None
    best_path = os.path.join(output_dir, 'checkpoints', 'best.pth')
    if not os.path.isfile(best_path):
        log.log('  ⚠ best.pth not found — skipping test evaluation.')
        return None
    ckpt = load_checkpoint(best_path, str(device))
    if ckpt is None:
        return None
    bb = backbone.module if hasattr(backbone, 'module') else backbone
    hd = head.module if hasattr(head, 'module') else head
    load_model_state(bb, ckpt.get('backbone_state_dict', ckpt.get('model_state_dict', ckpt)), strict=False)
    head_key = 'classifier_state_dict' if 'classifier_state_dict' in ckpt else 'decoder_state_dict'
    hd.load_state_dict(ckpt.get(head_key, {}), strict=False)
    log.log(f'  Loaded best checkpoint (epoch {ckpt.get('epoch', '?')}) for test eval')
    classes_txt = os.path.join(dataset_root, 'train', 'classes.txt')
    test_dataset = COCOMultiLabelDataset(root_dir=dataset_root, split='test', transform=MedicalAug(cfg.input_size, train=False), classes_txt=classes_txt)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)
    backbone.eval()
    head.eval()
    all_logits, all_labels = ([], [])
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            logits = _forward(backbone, head, images, mode, backbone_mode)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    probs = torch.sigmoid(all_logits)
    metrics = compute_metrics(probs.numpy(), all_labels.numpy())
    log.log(f'  ── Test-set ({len(test_dataset)} images) ──')
    log.log(f'     mAP        : {metrics['mAP']:.4f}')
    log.log(f'     F1  macro  : {metrics['f1_macro']:.4f}')
    log.log(f'     F1  micro  : {metrics['f1_micro']:.4f}')
    log.log(f'     AUC macro  : {metrics['auc_macro']:.4f}')
    log.log(f'     ExactMatch : {metrics['exact_match']:.4f}')
    return metrics

def _write_summary(output_dir: str, summary: dict):
    path = os.path.join(output_dir, 'summary.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(f'Backbone:      {summary.get('backbone', '?')}\n')
        f.write(f'Stage:         {summary.get('stage', '?')}\n')
        f.write(f'Version:       {summary.get('version', '?')}\n')
        f.write(f'Mode:          {summary.get('mode', '?')}\n')
        f.write(f'Backbone Mode: {summary.get('backbone_mode', '?')}\n')
        f.write(f'Best mAP:      {summary.get('best_mAP', 0):.4f}\n')
        f.write(f'Epochs:        {summary.get('epochs', '?')}\n')
        f.write(f'GPUs:          {summary.get('world_size', 1)}\n')
        f.write(f'Time:          {summary.get('train_time_min', 0):.1f} min\n')
        for key in sorted(summary):
            if key.startswith('test_'):
                f.write(f'{key}: {summary[key]:.4f}\n')

def _write_test_score(mode: str, backbone_name: str, stage, version: str, metrics: dict, backbone_mode: int=1):
    import os as _os
    scoring_dir = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), 'output')
    _os.makedirs(scoring_dir, exist_ok=True)
    scoring_path = _os.path.join(scoring_dir, 'test_scoring.log')
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    mode_tag = {'mlp': 'MLP+ASL', 'decoder': 'Decoder+ASL', 'bce': 'MLP+BCE'}.get(mode, mode)
    bm_tag = {1: 'freeze', 2: 'e2e'}.get(backbone_mode, f'bm{backbone_mode}')
    line = f'[{timestamp}] {mode_tag:12s} | {bm_tag:6s} | {backbone_name:18s} | stage={stage} ver={version:5s} | mAP={metrics.get('mAP', 0):.4f} | F1_macro={metrics.get('f1_macro', 0):.4f} | F1_micro={metrics.get('f1_micro', 0):.4f} | AUC_macro={metrics.get('auc_macro', 0):.4f} | ExactMatch={metrics.get('exact_match', 0):.4f}\n'
    with open(scoring_path, 'a') as f:
        f.write(line)
