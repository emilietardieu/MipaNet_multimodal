"""
Génération de cartes Score-CAM pour un modèle MipaNet mono-branche.
Une heatmap par classe, superposée à l'image source.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
import torch.nn.functional as F

from config import get_config
from model.core.CAM import ScoreCAM, get_layer
from model.datasets import get_dataset
from model.datasets.transforms import build_source_transforms, get_normalized_mean_std
from model.model import get_mipanet

# ================================================================
# CONFIGURATION — à modifier selon ton expérience
# ================================================================

ROOT       = Path(r"/home/etardieu/Documents/my_data/these/V1/Dataset/Dataset")
MODEL_PATH = Path(r"/home/etardieu/Documents/my_data/these/V1/Resultats/1-Modalite/1-modalite_V3/MNH/1772192075.pth")
BRANCHES   = [['MNH']]       # mono-branche obligatoire pour ScoreCAM

IMAGE_IDX = [        # nom(s) de fichier dans le split 'test'
    '09_201_3',
    '09_201_39',
    '09_283_0',
    '09_604_1',
    '09_676_10',
    '09_710_1',
    '09_74_28',
    '09_74_9',
    '09_922_0',
    '11_416_18',
    '11_536_2',
    '11_803_40',
    '11_902_9',
    '31_188_24',
    '31_193_7',
    '31_196_59',
    '31_383_35',
    '31_384_0',
    '31_467_37',
    '31_467_90',
    '31_517_57',
    '31_517_68',
    '31_517_74',
    '31_636_38',
    '31_636_51',
    '31_684_15',
    '31_734_1',
    '31_769_25',
    '31_782_19',
    '31_889_8',
    '31_9_1',
    '65_10_34',
    '65_191_1',
    '65_20_4',
    '65_228_24',
    '65_233_3',
    '65_245_23',
    '65_274_14',
    '65_287_12',
    '65_296_0',
    '65_323_12',
    '65_332_10',
    '65_410_11',
    '65_410_6',
    '65_414_5',
    '65_42_38',
    '65_543_15',
    '65_545_10',
    '65_572_2',
    '65_641_8',
    '65_726_8',
    '65_728_93',
    '65_859_2',
    '65_892_28',
    '65_915_2',
    '65_98_8',
    '66_241_7',
    '66_337_0',
    '66_580_37',
    '66_580_38',
    '66_580_56',
    '66_69_106',
    '66_69_138',
    '66_758_12',
    '66_800_33',
    '66_800_48',
    '66_816_23',
]
OUTPUT_DIR = Path("/home/etardieu/Documents/my_data/these/V1/Resultats/1-Modalite/1-modalite_V3/CAM/MNH")  # dossier de sauvegarde des figures CAM

CAM_LAYER   = 'layer4'       # 'layer1', 'layer2', 'layer3' ou 'layer4'
CLASS_NAMES = ['sol', 'forêt', 'vieille forêt']
SIGNED_CAM  = False          # True → CAM signée [−1,1] (rouge=active, bleu=inhibe)
                             # False → CAM positive [0,1] (jet)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ================================================================


def _load_model(model_path, branches, config, device):
    """Charge le modèle et ses poids depuis model_path."""
    model = get_mipanet(
        dataset       = config['testing']['dataset'],
        branches      = branches,
        sources       = config['sources'],
        pass_rff      = [True] * len(branches),
        first_fusions = config['encoder']['first_fusions'],
        last_fusion   = config['encoder']['last_fusion'],
        use_tgcc      = config['TGCC']['use_TGCC'],
    )
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model


def _denormalize(tensor, mean, std):
    """Dénormalise un tenseur [C, H, W] et le clamp dans [0, 1]."""
    m = torch.tensor(mean).view(-1, 1, 1)
    s = torch.tensor(std).view(-1, 1, 1)
    return (tensor * s + m).clamp(0, 1)


def _process_one(image_name, dataset, model, norm_stats, sources):
    """Génère et sauvegarde la figure CAM pour une image donnée."""
    idx = dataset.file_names.index(image_name)
    sources_dict, target, file_name = dataset[idx]
    sources_dict = {k: v.unsqueeze(0).to(DEVICE) for k, v in sources_dict.items()}

    # --- Prédiction de référence ---
    with torch.no_grad():
        outputs = model(sources_dict)
    pred = torch.argmax(outputs[0][0], dim=0).cpu()   # [H, W]

    # --- Score-CAM pour chaque classe ---
    layer = get_layer(model, CAM_LAYER)
    cams  = []
    with ScoreCAM(model, target_layer=layer) as cam_gen:
        for cls in range(len(CLASS_NAMES)):
            print(f"  Calcul CAM classe {cls} ({CLASS_NAMES[cls]})...")
            cams.append(cam_gen.generate(sources_dict, target_class=cls, signed=SIGNED_CAM))

    # --- Préparation de l'image de fond pour la visualisation ---
    used_sources = [src for branch in BRANCHES for src in branch]
    main_src = next(
        (s for s in used_sources if sources[s]['channels'] >= 3),
        used_sources[0]
    )
    vis_raw = sources_dict[main_src][0].cpu()   # [C, H, W], valeurs normalisées
    # Étirement min-max par canal pour l'affichage (fonctionne quelle que soit l'unité physique)
    v_min = vis_raw.flatten(1).min(dim=1)[0].view(-1, 1, 1)
    v_max = vis_raw.flatten(1).max(dim=1)[0].view(-1, 1, 1)
    vis = (vis_raw - v_min) / (v_max - v_min + 1e-8)
    if vis.shape[0] >= 3:
        vis_np = vis[[0, 1, 2]].numpy().transpose(1, 2, 0)
    else:
        vis_np = vis[0].numpy()

    # --- Figure : Image | GT | Pred | CAM×n_classes ---
    cmap_disc = mcolors.ListedColormap(['black', 'gray', 'white'])
    n_panels  = 3 + len(CLASS_NAMES)
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4))

    # Image d'entrée
    if vis_np.ndim == 3:
        axes[5].imshow(vis_np)
    else:
        axes[5].imshow(vis_np, cmap='gray')
    axes[5].set_title(f'Image ({main_src})', fontweight='bold')
    axes[5].axis('off')

    # Vérité terrain
    axes[0].imshow(target.numpy(), cmap=cmap_disc, vmin=0, vmax=2)
    axes[0].set_title('Vérité terrain', fontweight='bold')
    axes[0].axis('off')

    # Prédiction
    axes[1].imshow(pred.numpy(), cmap=cmap_disc, vmin=0, vmax=2)
    axes[1].set_title('Prédiction', fontweight='bold')
    axes[1].axis('off')

    # Une heatmap par classe
    if SIGNED_CAM:
        cam_cmap, cam_vmin, cam_vmax, cam_alpha = 'seismic', -1, 1, 0.75
    else:
        cam_cmap, cam_vmin, cam_vmax, cam_alpha = 'jet', 0, 1, 0.5

    for cls, (cam, name) in enumerate(zip(cams, CLASS_NAMES)):
        ax = axes[2 + cls]
        if vis_np.ndim == 3:
            ax.imshow(vis_np)
        else:
            ax.imshow(vis_np, cmap='gray')
        im = ax.imshow(cam, cmap=cam_cmap, alpha=cam_alpha, vmin=cam_vmin, vmax=cam_vmax)
        plt.colorbar(im, ax=ax, fraction=0.046, shrink=0.8)
        ax.set_title(f'CAM — {name}', fontweight='bold')
        ax.axis('off')

    plt.suptitle(f'{file_name}  |  couche : {CAM_LAYER}', fontsize=11)
    plt.tight_layout()

    out_path = OUTPUT_DIR / f"{Path(file_name).stem}_cam_{CAM_LAYER}.png"
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Sauvegardé : {out_path}")


def run_cam():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    config, _ = get_config()
    sources    = config['sources']

    # --- Chargement du dataset et du modèle (une seule fois) ---
    source_transforms = build_source_transforms(sources)
    norm_stats        = get_normalized_mean_std(sources)

    dataset = get_dataset(
        config['testing']['dataset'],
        root              = ROOT,
        mode              = 'test',
        sources           = sources,
        source_transforms = source_transforms,
    )

    model = _load_model(MODEL_PATH, BRANCHES, config, DEVICE)

    # --- Traitement de chaque image ---
    names = IMAGE_IDX if isinstance(IMAGE_IDX, list) else [IMAGE_IDX]
    for i, name in enumerate(names):
        print(f"\n[{i+1}/{len(names)}] {name}")
        _process_one(name, dataset, model, norm_stats, sources)


if __name__ == '__main__':
    run_cam()
