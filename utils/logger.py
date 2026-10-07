import os
import time
import logging
class SummaryWriter:
    def __init__(self, *args, **kwargs): pass
    def add_scalar(self, *args, **kwargs): pass
    def close(self): pass
    def flush(self): pass

def is_rank0() -> bool:
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            return dist.get_rank() == 0
    except Exception:
        pass
    return True

def get_rank() -> int:
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    return 0

def get_world_size() -> int:
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            return dist.get_world_size()
    except Exception:
        pass
    return 1

def all_reduce_mean(tensor) -> float:
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            world_size = dist.get_world_size()
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            tensor /= world_size
        return tensor.item()
    except Exception:
        return tensor.item() if hasattr(tensor, 'item') else float(tensor)

class TimeTracker:

    def __init__(self, total_epochs: int, steps_per_epoch: int):
        self.total_epochs = total_epochs
        self.steps_per_epoch = steps_per_epoch
        self.epoch_start = None
        self.ema_batch_time = None

    def start_epoch(self):
        self.epoch_start = time.time()

    def update_batch(self, batch_idx: int, current_epoch: int):
        now = time.time()
        elapsed = now - self.epoch_start
        if batch_idx > 0:
            batch_time = elapsed / (batch_idx + 1)
            if self.ema_batch_time is None:
                self.ema_batch_time = batch_time
            else:
                self.ema_batch_time = 0.9 * self.ema_batch_time + 0.1 * batch_time

    def eta(self, batch_idx: int, current_epoch: int) -> str:
        bt = self.ema_batch_time
        if bt is None:
            return '--:--:--'
        remaining = (self.total_epochs - current_epoch - 1) * self.steps_per_epoch + (self.steps_per_epoch - batch_idx - 1)
        eta_secs = max(0, int(remaining * bt))
        h, rem = divmod(eta_secs, 3600)
        m, s = divmod(rem, 60)
        if h > 0:
            return f'{h:02d}:{m:02d}:{s:02d}'
        return f'{m:02d}:{s:02d}'

class Logger:

    def __init__(self, log_dir: str, experiment_name: str='exp'):
        self._rank0 = is_rank0()
        self.log_dir = log_dir
        if self._rank0:
            os.makedirs(log_dir, exist_ok=True)
            self.tb_dir = os.path.join(log_dir, 'tensorboard', experiment_name)
            os.makedirs(self.tb_dir, exist_ok=True)
            self.writer = SummaryWriter(self.tb_dir)
            self.logger = logging.getLogger(experiment_name)
            self.logger.setLevel(logging.INFO)
            self.logger.handlers.clear()
            os.makedirs(os.path.join(log_dir, 'logs'), exist_ok=True)
            fh = logging.FileHandler(os.path.join(log_dir, 'logs', 'train.log'))
            fh.setLevel(logging.INFO)
            fh.setFormatter(logging.Formatter('%(asctime)s | %(message)s'))
            self.logger.addHandler(fh)
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            ch.setFormatter(logging.Formatter('%(message)s'))
            self.logger.addHandler(ch)
        else:
            self.writer = None
            self.logger = None

    def log(self, msg: str):
        if self._rank0 and self.logger:
            self.logger.info(msg)

    def add_scalar(self, tag: str, val: float, step: int):
        if self._rank0 and self.writer:
            self.writer.add_scalar(tag, val, step)

    def close(self):
        if self._rank0 and self.writer:
            self.writer.close()
