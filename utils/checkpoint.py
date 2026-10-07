import os
import sys
import torch
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

def save_checkpoint(state: dict, save_dir: str, filename: str='last.pth', is_best: bool=False):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    torch.save(state, path)
    if is_best:
        best_path = os.path.join(save_dir, 'best.pth')
        torch.save(state, best_path)

def load_checkpoint(path: str, device: str='cpu'):
    if not os.path.isfile(path):
        return None
    state = torch.load(path, map_location=device, weights_only=False)
    return state

def load_model_state(model, state_dict, strict: bool=True):
    sd = {k.replace('module.', ''): v for k, v in state_dict.items()}
    try:
        model.load_state_dict(sd, strict=strict)
    except RuntimeError:
        model_sd = model.state_dict()
        matched = {k: v for k, v in sd.items() if k in model_sd and v.shape == model_sd[k].shape}
        if matched:
            model.load_state_dict(matched, strict=False)
            print(f'[checkpoint] Loaded {len(matched)}/{len(model_sd)} params (strict=False)')
