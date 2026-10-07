# AdvScan-Tongue

Core implementation of self-supervised tongue-image pretraining and multi-label classification using an enhanced ViT-Tiny backbone.

## Included implementation

- ViT-Tiny with eleven v2_3 blocks inserted after Transformer blocks 0–10.
- DINO self-supervised pretraining.
- End-to-end MLP classification with BCE and a lower backbone learning rate.
- Checkpoint inference on the six-label test subset.

This export preserves the historical checkpoint implementation, including full block-parameter zero initialization and the inner plus outer residual connections (effective `2*x + F(x)`). It does not silently substitute a later block implementation. Alternative block versions, decoder/ASL ablations, datasets, checkpoints, logs and visualization scripts are not included.

## Environment

Inference was verified with Python 3.13.9, PyTorch 2.9.1+cu128, torchvision 0.24.1+cu128, timm 1.0.26, NumPy 1.26.3, scikit-learn 1.7.2 and Pillow 12.0.0 on an NVIDIA A100. These are the observed environment versions, not a clean-install lock file.

The direct dependencies are listed in `requirements.txt`.

## Data and weights

Supply your own local paths. No image data or model weights are bundled.

The supervised dataset uses COCO annotations:

```text
dataset_root/
  train/images/
  train/annotations/train.json
  train/classes.txt
  val/images/
  val/annotations/val.json
  test/images/
  test/annotations/test.json
```

Training and validation use seven labels. Six-label inference retains the same seven-output checkpoint and selects the six output columns excluding `botaishe`.

The inference dataset root contains `test/images`, `test/annotations/test.json`, `test/classes.txt`, and `manifest.json`. The manifest must contain:

```json
{
  "train_classes": ["botaishe", "hongshe", "hongdianshe", "liewenshe", "chihenshe", "baitaishe", "huangtaishe"],
  "test_classes": ["hongshe", "hongdianshe", "liewenshe", "chihenshe", "baitaishe", "huangtaishe"]
}
```

The supplied inference entry point checks for the original 550-image six-label evaluation set. Category IDs in the COCO annotations must follow the listed class order. Missing training image files are skipped by the loader.

## Training

DINO pretraining of the enhanced encoder, initialized from an existing domain-pretrained ViT-Tiny checkpoint:

```bash
python main.py --phase pretrain \
  --dataset_root /path/to/unlabeled_images \
  --source_checkpoint /path/to/domain_pretrained_vit_tiny.pth \
  --output_dir runs/dino_v2_3
```

End-to-end classification:

```bash
python main.py --phase classifier \
  --dataset_root /path/to/supervised_dataset \
  --source_checkpoint runs/dino_v2_3/checkpoints/best.pth \
  --output_dir runs/mlp_bce_e2e
```

The fixed defaults are 100 pretraining epochs and 50 classification epochs. Classification uses a head learning rate of 0.001 and a backbone multiplier of 0.1. Historical initialization checkpoints that serialize custom configuration objects may require their original configuration modules on `PYTHONPATH`. Full retraining of this export has not been verified.

## Checkpoint inference

```bash
python infer.py \
  --checkpoint /path/to/best.pth \
  --dataset_root /path/to/test6 \
  --device cuda:0 \
  --batch_size 16
```

The script prints metrics without writing prediction files. F1 and ExactMatch use a fixed threshold of 0.5. To compare against previously saved probabilities, optionally pass `--reference_predictions /path/to/predictions.npz`.

## Verified inference result

Using the selected epoch-5 historical checkpoint, strict parameter loading and full inference over 550 images succeeded. The maximum difference from the saved reference probabilities was 0.0 in the tested environment.

| Metric | Value |
| --- | ---: |
| mAP | 0.7186135075 |
| Macro-F1 | 0.6261120670 |
| Micro-F1 | 0.7732181425 |
| Macro-AUC | 0.8762455042 |
| ExactMatch | 0.4836363636 |

This validates checkpoint inference compatibility, not reproduction of training from scratch.
