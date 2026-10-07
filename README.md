# AdvScan-Tongue

Core implementation of self-supervised tongue-image pretraining and multi-label classification using an enhanced ViT-Tiny backbone.

## Included implementation

- ViT-Tiny with eleven v2_3 blocks inserted after Transformer blocks 0–10.
- End-to-end MLP classification with BCE and a lower backbone learning rate.
- Checkpoint inference on the six-label test subset.

## Environment

Inference was verified with Python 3.13.9, PyTorch 2.9.1+cu128, torchvision 0.24.1+cu128, timm 1.0.26, NumPy 1.26.3, scikit-learn 1.7.2 and Pillow 12.0.0 on an NVIDIA A100. These are the observed environment versions, not a clean-install lock file.

The direct dependencies are listed in `requirements.txt`.

## Data and weights

The dataset is distributed as four split 7-Zip volumes through [GitHub Releases](https://github.com/Lilin517/AdvScan-Tongue/releases). Open the dataset release and download all four assets:

```text
dataset.7z.001
dataset.7z.002
dataset.7z.003
dataset.7z.004
```

Keep the original filenames and place all volumes in the same directory. They form one archive; do not extract each volume separately. With 7-Zip installed, run the following from the repository root (replace `/path/to/downloads` with your download directory):

```bash
7z x /path/to/downloads/dataset.7z.001 -o.
```

On Windows, open `dataset.7z.001` with 7-Zip and extract it into the repository root. The resulting structure is:

```text
AdvScan-Tongue/
  main.py
  infer.py
  dataset/
    mae_pretrain/
      ... image files and metadata.csv
    shezhenv3-coco-percent-3/
      train/
        images/
        annotations/train.json
        classes.txt
      val/
        images/
        annotations/val.json
      test/
        images/
        annotations/test.json
        classes.txt
```

`dataset/mae_pretrain` contains 18,264 unlabeled images for self-supervised pretraining. `dataset/shezhenv3-coco-percent-3` provides the supervised splits and COCO annotations. The loader uses `train.json`, `val.json`, and `test.json`; backup JSON files and additional images outside those annotations are not used. Missing image files are skipped by the loader.

The six evaluation labels are red tongue (`hongshe`), red spots (`hongdianshe`), fissured tongue (`liewenshe`), tooth-marked tongue (`chihenshe`), white coating (`baitaishe`), and yellow coating (`huangtaishe`).

### Prepare the six-label inference directory

The archive does not include the separate `test6` directory expected by `infer.py`. Run the following Python snippet from the repository root to prepare it. It preserves the 550 test images and their order, selects the six evaluation labels, and creates the manifest needed to select the corresponding checkpoint outputs. The extracted source data are left unchanged. Save the snippet as `prepare_test6.py` and run `python prepare_test6.py`.

```python
import json
import shutil
from pathlib import Path

source = Path("dataset/shezhenv3-coco-percent-3")
target = Path("dataset/test6")
classes = ["hongshe", "hongdianshe", "liewenshe",
           "chihenshe", "baitaishe", "huangtaishe"]
test = json.loads((source / "test/annotations/test.json").read_text(encoding="utf-8"))
train_classes = (source / "train/classes.txt").read_text(encoding="utf-8").splitlines()
train_classes = [name.strip() for name in train_classes if name.strip()]
categories = sorted(test["categories"], key=lambda item: item["id"])
selected = [item for item in categories if item["name"] in classes]
assert [item["name"] for item in selected] == classes
assert all(name in train_classes for name in classes)
assert len(test["images"]) == 550
assert all((source / "test/images" / item["file_name"]).is_file()
           for item in test["images"])
if target.exists():
    raise FileExistsError(f"{target} already exists; use a new output directory.")

ids = {item["id"] for item in selected}
test["categories"] = selected
test["annotations"] = [item for item in test["annotations"]
                       if item["category_id"] in ids]
(target / "test/annotations").mkdir(parents=True)
(target / "test/images").mkdir()
for item in test["images"]:
    relative = Path(item["file_name"])
    destination = target / "test/images" / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / "test/images" / relative, destination)
(target / "test/annotations/test.json").write_text(json.dumps(test), encoding="utf-8")
(target / "test/classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")
(target / "manifest.json").write_text(json.dumps({
    "train_classes": train_classes,
    "test_classes": classes
}, indent=2), encoding="utf-8")
print(f"Prepared {len(test['images'])} images in {target}")
```

Model weights are separate from the dataset archive. The training and inference commands below require the corresponding checkpoint files; replace checkpoint placeholders with their actual local paths.

## Training

DINO pretraining of the enhanced encoder, initialized from an existing domain-pretrained ViT-Tiny checkpoint:

```bash
python main.py --phase pretrain \
  --dataset_root dataset/mae_pretrain \
  --source_checkpoint /path/to/domain_pretrained_vit_tiny.pth \
  --output_dir runs/dino_v2_3
```

End-to-end classification:

```bash
python main.py --phase classifier \
  --dataset_root dataset/shezhenv3-coco-percent-3 \
  --source_checkpoint runs/dino_v2_3/checkpoints/best.pth \
  --output_dir runs/mlp_bce_e2e
```

The fixed defaults are 100 pretraining epochs and 50 classification epochs. Classification uses a head learning rate of 0.001 and a backbone multiplier of 0.1. Historical initialization checkpoints that serialize custom configuration objects may require their original configuration modules on `PYTHONPATH`. Full retraining of this export has not been verified.

## Checkpoint inference

```bash
python infer.py \
  --checkpoint /path/to/best.pth \
  --dataset_root dataset/test6 \
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
