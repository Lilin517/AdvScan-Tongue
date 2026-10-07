import argparse, os
from pathlib import Path
import torch
from config import Config

def main():
    p=argparse.ArgumentParser(description="Best-checkpoint ViT-Tiny + v2_3 + MLP/BCE")
    p.add_argument('--phase',choices=['pretrain','classifier'],required=True)
    p.add_argument('--dataset_root',required=True)
    p.add_argument('--source_checkpoint',required=True)
    p.add_argument('--output_dir',required=True)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    if not Path(args.source_checkpoint).is_file():raise FileNotFoundError(args.source_checkpoint)
    cfg=Config();rank=int(os.environ.get('LOCAL_RANK',0));world=int(os.environ.get('WORLD_SIZE',1))
    device=torch.device('cuda:'+str(rank) if torch.cuda.is_available() else 'cpu')
    if args.phase=='pretrain':
        from engine.pretrain import run_pretrain
        run_pretrain('ViT-Tiny',list(range(1,12)),'v2_3',device,args.output_dir,cfg,
                     dataset_root=args.dataset_root,local_rank=rank,world_size=world,pretrain_ckpt_path=args.source_checkpoint)
    else:
        from engine.trainer import train_one_backbone
        train_one_backbone('ViT-Tiny',list(range(1,12)),'v2_3','bce',device,args.output_dir,args.dataset_root,
                           overrides={'epochs':50,'backbone_lr_scale':0.1},resume=args.resume,
                           source_ckpt=args.source_checkpoint,backbone_mode=2,local_rank=rank,world_size=world)
if __name__=='__main__':main()
