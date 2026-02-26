"""
Test du modèle MIPANet multi-branches pour la segmentation d'images.
"""
import os
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils import data
from tqdm import tqdm
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Spécification de la carte GPU à utiliser
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Importation des modules du projet
from config import get_config, SOURCES
from model.core.metrics import batch_pix_accuracy, batch_intersection_union
from model.datasets import get_dataset
from model.datasets.transforms import build_source_transforms, build_augmented_transforms, get_normalized_mean_std
from model.model import get_mipanet

class Tester():
    """
    Classe Tester pour évaluer le modèle MIPANet multi-branches.
    """
    def __init__(self, root, model_path, branches, output_dir=None):
        self.root = Path(root)
        self.model_path = Path(model_path)
        self.config, _ = get_config()

        # Dépackage du format [branch_config, lr, ratio] ou branch_config seul
        if isinstance(branches, (list, tuple)) and len(branches) >= 2 and isinstance(branches[1], float):
            self.branches = branches[0]
        else:
            self.branches = branches

        # Config sources
        self.sources = self.config['sources']
        self.source_names = list(self.sources.keys())
        self.norm_stats = get_normalized_mean_std(self.sources)

        # Construire les transforms
        source_transforms = build_source_transforms(self.sources)

        # Extraire les informations sur l'expérience depuis les chemins
        self.experiment_name = None
        self.weights_name = self.model_path.stem

        if len(self.model_path.parts) >= 2:
            for part in reversed(self.model_path.parts):
                if part.startswith("runs-"):
                    self.experiment_name = part
                    break

        # Configuration du répertoire de sortie
        if output_dir is None:
            if self.experiment_name:
                folder_name = f"{self.experiment_name}_{self.weights_name}"
            else:
                folder_name = f"test_results_{self.weights_name}"
            self.output_dir = self.root / folder_name
        else:
            self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir = self.output_dir / "predictions"
        self.images_dir.mkdir(exist_ok=True)

        # Chargement du dataset de test
        self.testset = get_dataset(
            self.config['testing']['dataset'],
            root              = self.root,
            mode              = 'test',
            sources           = self.sources,
            source_transforms = source_transforms,
        )

        # Création du DataLoader
        kwargs = {'num_workers': self.config['testing']['workers'], 'pin_memory': True} if self.config['testing']['use_cuda'] else {}
        self.testloader = data.DataLoader(
            self.testset,
            batch_size=self.config['testing']['batch_size'],
            drop_last=False,
            shuffle=False,
            **kwargs
        )

        self.nclass = self.testset.num_class
        self.class_names = ['sol', 'foret', 'vieille_foret']

        # Configuration du dispositif
        self.device = torch.device("cuda:0" if self.config['testing']['use_cuda'] else "cpu")

        # Initialisation du modèle
        pass_rff = [True] * len(self.branches)
        self.model = get_mipanet(
            dataset       = self.config['testing']['dataset'],
            branches      = self.branches,
            sources       = self.sources,
            pass_rff      = pass_rff,
            first_fusions = self.config['encoder']['first_fusions'],
            last_fusion   = self.config['encoder']['last_fusion'],
            use_tgcc      = self.config['TGCC']['use_TGCC'],
        )

        # Chargement des poids
        self._load_model()

        used_sources = [src for branch in self.branches for src in branch]
        print(f"\nModalités utilisées : {' // '.join(used_sources)}")

    def _load_model(self):
        """
        Charge les poids du modèle depuis le fichier .pth.
        """
        if not self.model_path.exists():
            raise FileNotFoundError(f"Le fichier de modèle {self.model_path} n'existe pas")

        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)

        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        try:
            self.model.load_state_dict(state_dict)
        except RuntimeError as e:
            if "Missing key(s)" in str(e) or "Unexpected key(s)" in str(e):
                print(f"Incompatibilité partielle détectée.")
                print("Tentative de chargement partiel des poids compatibles...")

                model_dict = self.model.state_dict()
                compatible_dict = {}
                ignored_count = 0

                for k, v in state_dict.items():
                    if k in model_dict and model_dict[k].shape == v.shape:
                        compatible_dict[k] = v
                    else:
                        ignored_count += 1
                        if ignored_count <= 10:
                            print(f"  Paramètre ignoré: {k}")
                        elif ignored_count == 11:
                            print(f"  ... et d'autres paramètres ignorés")

                model_dict.update(compatible_dict)
                self.model.load_state_dict(model_dict)

                print(f"Chargement partiel réussi: {len(compatible_dict)}/{len(state_dict)} paramètres chargés.")
            else:
                raise e

        self.model.to(self.device)
        self.model.eval()

    def _sources_to_device(self, sources_dict):
        """Transfère toutes les sources sur le device."""
        return {name: tensor.to(self.device) for name, tensor in sources_dict.items()}

    def denormalize(self, tensor, mean, std):
        mean = torch.tensor(mean).view(-1, 1, 1).to(tensor.device)
        std = torch.tensor(std).view(-1, 1, 1).to(tensor.device)
        return tensor * std + mean

    def _denormalize_source(self, tensor, source_name):
        """Dénormalise un tenseur d'une source spécifique."""
        stats = self.norm_stats[source_name]
        return self.denormalize(tensor, stats['mean'], stats['std'])

    def colorize_mask(self, mask):
        mask_np = mask.cpu().numpy()
        class_colors = {
            0: (0, 0, 0),
            1: (127, 127, 127),
            2: (255, 255, 255),
        }
        h, w = mask_np.shape
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)
        for class_id, color in class_colors.items():
            color_mask[mask_np == class_id] = color
        return color_mask

    def calculate_metrics(self, predictions, targets):
        total_correct = 0
        total_labeled = 0
        total_inter = np.zeros(self.nclass)
        total_union = np.zeros(self.nclass)

        for pred, target in zip(predictions, targets):
            correct, labeled = batch_pix_accuracy(pred.unsqueeze(0), target.unsqueeze(0))
            total_correct += correct
            total_labeled += labeled

            inter, union = batch_intersection_union(pred.unsqueeze(0), target.unsqueeze(0), self.nclass)
            total_inter += inter
            total_union += union

        pixel_acc = 1.0 * total_correct / (np.spacing(1) + total_labeled)
        iou_per_class = 1.0 * total_inter / (np.spacing(1) + total_union)
        miou = iou_per_class.mean()

        f1_per_class = 2 * iou_per_class / (1 + iou_per_class)
        f1_mean = f1_per_class.mean()

        return {
            'pixel_accuracy': pixel_acc,
            'miou': miou,
            'f1_score': f1_mean,
            'iou_per_class': iou_per_class,
            'f1_per_class': f1_per_class
        }

    def save_prediction_image(self, sources_dict, pred_tensor, target_tensor, file_name, batch_idx=0):
        """
        Sauvegarde une image composite avec les sources, target et prédiction.
        """
        # Trouver la première source multi-canal (IRC) pour l'affichage principal
        main_source = None
        for name in self.source_names:
            if self.sources[name]['channels'] >= 3:
                main_source = name
                break
        if main_source is None:
            main_source = self.source_names[0]

        vis_main = self._denormalize_source(sources_dict[main_source][batch_idx], main_source)
        if vis_main.shape[0] >= 3:
            vis_main = vis_main[[0, 1, 2]].clamp(0, 1)  # CIR : NIR→R, R→G, G→B
        else:
            vis_main = vis_main.clamp(0, 1)
        vis_main = TF.to_pil_image(vis_main.cpu())

        pred_vis = self.colorize_mask(pred_tensor)
        target_vis = self.colorize_mask(target_tensor)

        pred_pil = Image.fromarray(pred_vis)
        target_pil = Image.fromarray(target_vis)

        w, h = vis_main.size
        separator_width = 3
        separator_color = (255, 255, 255)
        text_height = 25

        total_width = w * 3 + separator_width * 2
        total_height = h + text_height
        composite = Image.new('RGB', (total_width, total_height), color=(240, 240, 240))

        try:
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(composite)
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
            except:
                try:
                    font = ImageFont.truetype("arial.ttf", 16)
                except:
                    font = ImageFont.load_default()

            labels = [f"Image {main_source}", "Vérité terrain", "Prédiction"]
            positions = [
                (w // 2 - 40, 5),
                (w + separator_width + w // 2 - 60, 5),
                (2 * w + 2 * separator_width + w // 2 - 40, 5)
            ]

            for label, pos in zip(labels, positions):
                draw.text(pos, label, fill=(0, 0, 0), font=font)
        except ImportError:
            pass

        y_offset = text_height
        composite.paste(vis_main, (0, y_offset))

        separator1 = Image.new('RGB', (separator_width, h), separator_color)
        composite.paste(separator1, (w, y_offset))
        composite.paste(target_pil, (w + separator_width, y_offset))

        separator2 = Image.new('RGB', (separator_width, h), separator_color)
        composite.paste(separator2, (2 * w + separator_width, y_offset))
        composite.paste(pred_pil, (2 * w + 2 * separator_width, y_offset))

        clean_name = file_name.replace('.tif', '').replace('.png', '')
        output_path = self.images_dir / f"{clean_name}_composite.png"
        composite.save(output_path)

    def save_metrics_to_csv(self, metrics):
        csv_path = self.output_dir / "test_metrics.csv"

        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Métrique', 'Valeur'])
            writer.writerow(['Pixel Accuracy', f"{metrics['pixel_accuracy']:.6f}"])
            writer.writerow(['Mean IoU', f"{metrics['miou']:.6f}"])
            writer.writerow(['Mean F1-Score', f"{metrics['f1_score']:.6f}"])
            writer.writerow([])
            writer.writerow(['Classe', 'IoU', 'F1-Score'])
            for i, class_name in enumerate(self.class_names):
                writer.writerow([
                    class_name,
                    f"{metrics['iou_per_class'][i]:.6f}",
                    f"{metrics['f1_per_class'][i]:.6f}"
                ])

    def save_probability_heatmaps(self, sources_dict, pred_logits, target_tensor, file_name, batch_idx=0):
        """
        Sauvegarde un composite avec les sources, vérité terrain,
        prédiction et probabilité vieille forêt.
        """
        try:
            probabilities = F.softmax(pred_logits, dim=0)
            pred_classes = torch.argmax(pred_logits, dim=0)

            old_forest_prob = probabilities[2, :, :].cpu().numpy()

            target_np = target_tensor.cpu().numpy()
            pred_np = pred_classes.cpu().numpy()

            if np.any(np.isnan(old_forest_prob)) or np.any(np.isinf(old_forest_prob)):
                old_forest_prob = np.nan_to_num(old_forest_prob, nan=0.0, posinf=1.0, neginf=0.0)

            # Nombre de panneaux = sources utilisées + 3 (target, pred, prob vieille foret)
            used_sources = [src for branch in self.branches for src in branch]
            n_panels = len(used_sources) + 3
            fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4))
            plt.style.use('default')

            # Afficher chaque source utilisée
            for idx, src_name in enumerate(used_sources):
                src_tensor = sources_dict[src_name][batch_idx]
                vis = self._denormalize_source(src_tensor, src_name)

                if vis.shape[0] >= 3:
                    vis_np = vis[[0, 1, 2]].clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
                    axes[idx].imshow(vis_np)
                else:
                    vis_np = vis.squeeze(0).cpu().numpy()
                    vis_np = np.nan_to_num(vis_np, nan=0.0, posinf=1.0, neginf=0.0)
                    mnh_min, mnh_max = vis_np.min(), vis_np.max()
                    if mnh_max > mnh_min and np.isfinite(mnh_min) and np.isfinite(mnh_max):
                        vis_np = (vis_np - mnh_min) / (mnh_max - mnh_min)
                    else:
                        vis_np = np.full_like(vis_np, 0.5)
                    if src_name == 'biomasse':
                        cmap = 'Greens'
                    else:
                        cmap = 'gray'
                    im = axes[idx].imshow(vis_np, cmap=cmap)
                    plt.colorbar(im, ax=axes[idx], fraction=0.046, shrink=0.8)

                axes[idx].set_title(src_name, fontsize=12, fontweight='bold')
                axes[idx].axis('off')

            # Vérité terrain
            ax_target = axes[len(used_sources)]
            class_colors = ['black', 'gray', 'white']
            cmap_discrete = mcolors.ListedColormap(class_colors)
            ax_target.imshow(target_np, cmap=cmap_discrete, vmin=0, vmax=2)
            ax_target.set_title('Vérité terrain', fontsize=12, fontweight='bold')
            ax_target.axis('off')

            # Prédiction
            ax_pred = axes[len(used_sources) + 1]
            ax_pred.imshow(pred_np, cmap=cmap_discrete, vmin=0, vmax=2)
            ax_pred.set_title('Prédiction', fontsize=12, fontweight='bold')
            ax_pred.axis('off')

            # Probabilité Vieille Forêt
            ax_prob = axes[len(used_sources) + 2]
            im5 = ax_prob.imshow(old_forest_prob, cmap='YlOrBr', vmin=0, vmax=1)
            ax_prob.set_title('Prob Vieille Forêt', fontsize=12, fontweight='bold')
            ax_prob.axis('off')
            plt.colorbar(im5, ax=ax_prob, fraction=0.046, shrink=0.8)

            plt.tight_layout(pad=1.0)

            clean_name = file_name.replace('.tif', '').replace('.png', '')
            output_path = self.images_dir / f"{clean_name}_probability_maps.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()

        except Exception as e:
            print(f"Erreur lors de la sauvegarde du heatmap {file_name}: {str(e)}")
            try:
                plt.close()
            except:
                pass

    def calculate_confusion_matrix(self, predictions, targets):
        all_pred_classes = []
        all_target_classes = []

        for pred_logits, target in zip(predictions, targets):
            pred_classes = torch.argmax(pred_logits, dim=0)
            pred_flat = pred_classes.cpu().numpy().flatten()
            target_flat = target.cpu().numpy().flatten()
            all_pred_classes.extend(pred_flat)
            all_target_classes.extend(target_flat)

        conf_matrix = confusion_matrix(all_target_classes, all_pred_classes,
                                       labels=list(range(self.nclass)))
        return conf_matrix

    def save_confusion_matrix(self, conf_matrix):
        csv_path = self.output_dir / "confusion_matrix.csv"
        import pandas as pd

        df_conf = pd.DataFrame(conf_matrix,
                               index=[f"Vrai_{name}" for name in self.class_names],
                               columns=[f"Pred_{name}" for name in self.class_names])
        df_conf.to_csv(csv_path)

        conf_matrix_pct = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100

        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names, yticklabels=self.class_names,
                    cbar_kws={'label': 'Nombre de pixels'})
        plt.title('Matrice de Confusion - Nombre de pixels', fontsize=16, fontweight='bold')
        plt.xlabel('Prédiction', fontsize=14)
        plt.ylabel('Vérité terrain', fontsize=14)
        plt.tight_layout()
        plot_path = self.output_dir / "confusion_matrix_counts.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix_pct, annot=True, fmt='.1f', cmap='Blues',
                    xticklabels=self.class_names, yticklabels=self.class_names,
                    cbar_kws={'label': 'Pourcentage (%)'})
        plt.title('Matrice de Confusion - Pourcentages', fontsize=16, fontweight='bold')
        plt.xlabel('Prédiction', fontsize=14)
        plt.ylabel('Vérité terrain', fontsize=14)
        plt.tight_layout()
        plot_path_pct = self.output_dir / "confusion_matrix_percentages.png"
        plt.savefig(plot_path_pct, dpi=300, bbox_inches='tight')
        plt.close()

    def test(self, save_images=True, max_saved_images=50, save_probability_maps=True, max_probability_maps=20):
        """
        Lance le test complet du modèle.
        """
        print("\n=============== Début du test ===============")

        predictions = []
        targets = []
        saved_images_count = 0
        saved_probability_maps_count = 0

        with torch.no_grad():
            for i, (sources_dict, target, file_name) in tqdm(enumerate(self.testloader), total=len(self.testloader), desc="Test en cours"):
                sources_dict = self._sources_to_device(sources_dict)
                target = target.to(self.device)

                # Prédiction
                outputs = self.model(sources_dict)
                pred_logits = outputs[0]
                pred_classes = torch.argmax(pred_logits, dim=1)

                # Stocker pour calcul des métriques
                for b in range(pred_classes.shape[0]):
                    predictions.append(pred_logits[b])
                    targets.append(target[b])

                    batch_file_name = file_name[b] if isinstance(file_name, (list, tuple)) else file_name

                    if save_images and saved_images_count < max_saved_images:
                        self.save_prediction_image(
                            sources_dict, pred_classes[b], target[b], batch_file_name, batch_idx=b
                        )
                        saved_images_count += 1

                    if save_probability_maps and saved_probability_maps_count < max_probability_maps:
                        self.save_probability_heatmaps(
                            sources_dict, pred_logits[b], target[b], batch_file_name, batch_idx=b
                        )
                        saved_probability_maps_count += 1

        # Calcul des métriques
        print("\nCalcul des métriques...")
        metrics = self.calculate_metrics(predictions, targets)

        print("Calcul de la matrice de confusion...")
        conf_matrix = self.calculate_confusion_matrix(predictions, targets)
        self.save_confusion_matrix(conf_matrix)

        print("\n=============== Résultats du test ===============")
        print("\nMétriques globales :")
        print(f"  Pixel Accuracy : {metrics['pixel_accuracy']:.3f}")
        print(f"  Mean IoU       : {metrics['miou']:.3f}")
        print(f"  Mean F1-Score  : {metrics['f1_score']:.3f}")
        print("\nMétriques par classe :")
        for i, class_name in enumerate(self.class_names):
            print(f"  {class_name:14} - IoU: {metrics['iou_per_class'][i]:.3f}, F1: {metrics['f1_per_class'][i]:.3f}")
        print("\n")

        self.save_metrics_to_csv(metrics)

        print("=============== Test terminé ===============")

        return metrics


def test_model(root, model_path, branches, output_dir=None, save_images=True):
    tester = Tester(root, model_path, branches, output_dir)
    return tester.test(save_images=save_images)
