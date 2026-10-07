BACKBONE_REGISTRY = {'ViT-Tiny': {'timm_name': 'vit_tiny_patch16_224', 'family': 'vit', 'input_size': 224, 'feature_dim': 192, 'pretrained_weights': 'IMAGENET1K_V1', 'num_stages': 12, 'num_myblock_positions': 11, 'ssl_method': 'dino'}}

def list_backbones(family: str=None) -> list:
    names = list(BACKBONE_REGISTRY.keys())
    if family:
        names = [n for n in names if BACKBONE_REGISTRY[n]['family'] == family]
    return sorted(names)

def get_backbone_info(name: str) -> dict:
    if name not in BACKBONE_REGISTRY:
        available = ', '.join(sorted(BACKBONE_REGISTRY.keys()))
        raise KeyError(f"Unknown backbone: '{name}'.\nAvailable backbones: {available}")
    info = dict(BACKBONE_REGISTRY[name])
    if 'ssl_method' not in info:
        info['ssl_method'] = 'dino'
    return info

def resolve_backbone_list(backbone_arg: str, backbone_list_arg: str=None) -> list:
    if backbone_arg == 'ALL':
        return list_backbones()
    elif backbone_arg == 'LIST':
        if not backbone_list_arg:
            raise ValueError('--base_model_list is required when --base_model=LIST')
        names = [b.strip() for b in backbone_list_arg.split(',')]
        for n in names:
            if n not in BACKBONE_REGISTRY:
                available = ', '.join(sorted(BACKBONE_REGISTRY.keys()))
                raise KeyError(f"Unknown backbone in list: '{n}'. Available: {available}")
        return names
    else:
        if backbone_arg not in BACKBONE_REGISTRY:
            available = ', '.join(sorted(BACKBONE_REGISTRY.keys()))
            raise KeyError(f"Unknown backbone: '{backbone_arg}'. Available: {available}")
        return [backbone_arg]

def validate_stage(backbone_name: str, stage):
    info = get_backbone_info(backbone_name)
    family = info['family']
    if family == 'vit':
        valid = list(range(1, 12))
        desc = {i: f'after block_{i - 1} (position {i}/11)' for i in range(1, 12)}
    else:
        valid = [0, 1, 2, 3, 4]
        desc = {0: 'after stem (before main stages)', 1: 'after first main stage', 2: 'after second main stage', 3: 'after third main stage', 4: 'after last stage (before head/pooling)'}
    stages_to_check = stage if isinstance(stage, (list, tuple)) else [stage]
    for s in stages_to_check:
        if s not in valid:
            desc_lines = '\n'.join((f'    stage={d} → {desc[d]}' for d in valid))
            raise ValueError(f"[{backbone_name}] Invalid stage={s}. Family='{family}' supports stage ∈ {valid}.\nValid stages:\n{desc_lines}")
