import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from models.backbone import BackboneWithMyBlock
from models.mlp_head import MLPHead
from data.dataset import COCOMultiLabelDataset
from data.transforms import MedicalAug
from utils.metrics import compute_metrics

def strip_prefix(sd):
    return {k.removeprefix('module.'): v for k, v in sd.items()}

def main():
    parser = argparse.ArgumentParser(description='Evaluate the historical best checkpoint on the six-label test set.')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--dataset_root', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--reference_predictions', default=None)
    args = parser.parse_args()
    torch.set_num_threads(4)
    root = Path(args.dataset_root)
    mapping = json.loads((root / 'manifest.json').read_text())
    ds = COCOMultiLabelDataset(str(root), 'test', MedicalAug(224, False), str(root / 'test/classes.txt'))
    if len(ds) != 550 or ds.categories != mapping['test_classes'] or len(ds.categories) != 6:
        raise ValueError('Expected the verified 550-image six-label test set')
    columns = [mapping['train_classes'].index(c) for c in ds.categories]
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if ck.get('version', '').lower() != 'v2_3' or ck.get('stage') != list(range(1,12)) or ck.get('mode') != 'bce':
        raise ValueError('Checkpoint configuration does not match the exported model')
    model = BackboneWithMyBlock('ViT-Tiny', list(range(1,12)), pretrained=False, version='v2_3')
    model.load_state_dict(strip_prefix(ck['backbone_state_dict']), strict=True)
    sd = strip_prefix(ck['classifier_state_dict'])
    if sd['net.6.weight'].shape[0] != 7: raise ValueError('Expected seven output logits')
    head = MLPHead(sd['net.0.weight'].shape[1], 7, hidden_dim=sd['net.0.weight'].shape[0], dropout=0.3)
    head.load_state_dict(sd, strict=True)
    epoch = ck.get('epoch')
    del ck, sd
    device = torch.device(args.device)
    model.to(device).eval(); head.to(device).eval()
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    probabilities, labels = [], []
    start = time.monotonic()
    with torch.no_grad():
        for images, target in loader:
            features = model(images.to(device))
            pooled = features['cls'] if isinstance(features, dict) else features
            probabilities.append(torch.sigmoid(head(pooled).float())[:, columns].cpu().numpy())
            labels.append(target.numpy())
    prob = np.concatenate(probabilities); target = np.concatenate(labels)
    if not np.isfinite(prob).all(): raise ValueError('Non-finite predictions')
    result = compute_metrics(prob, target)
    result.update(samples=len(ds), threshold=0.5, checkpoint_epoch=epoch, seconds=round(time.monotonic()-start,2))
    if args.reference_predictions:
        with np.load(args.reference_predictions, allow_pickle=False) as previous:
            assert np.array_equal(previous['classes'], np.asarray(ds.categories))
            assert [Path(str(p)).name for p in previous['image_paths']] == [Path(p).name for p, _ in ds.samples]
            assert np.array_equal(previous['labels'], target)
            result['reference_max_probability_difference'] = float(np.max(np.abs(previous['probabilities']-prob)))
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    main()
