"""
Entrainement du modèle MIPANet multi-branches pour la segmentation d'images.
"""
import csv
import io
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import wandb

from pathlib import Path
from torch.amp import autocast, GradScaler
from torch.utils import data
from tqdm import tqdm


# Spécification de la carte GPU à utiliser
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Importation des modules du projet
from config import get_config, SOURCES
from model.datasets.files import save_checkpoint, mkdir
from model.datasets.transforms import build_source_transforms, build_augmented_transforms, get_normalized_mean_std
from model.datasets import get_dataset
from model.model import get_mipanet
from model.core.early_stopping import EarlyStopping
from model.core.loss import CombinedLoss
from model.core.lr_range_test import LRRangeTest
from model.core.metrics import batch_pix_accuracy, batch_intersection_union
from model.core.optimizer import select_optimizer
from model.core.scheduler import SchedulerWithHead
from model.core.util import get_param_ids, set_reproducibility


class Trainer():
    """
    Classe Trainer pour gérer l'entraînement et l'évaluation du modèle.
    :attribut Path root: Chemin vers le répertoire racine du dataset.
    :attribut list branches: Liste de listes de sources par branche.
    """
    def __init__(self, root, branches):

        self.root = root
        self.config, self.hyperparam = get_config()

        # Dépackage du format [branch_config, lr_override] ou branch_config seul
        if isinstance(branches, (list, tuple)) and len(branches) == 3 and isinstance(branches[1], float):
            self.branches = branches[0]
            self.config['training']['lr'] = branches[1]
            self.config['training']['onecycle']['max_lr'] = branches[1]
            self.ratio_lr = branches[2] if len(branches) > 2 else 5.0
        else:
            self.branches = branches

        # Récupérer les configs sources et branches
        self.sources = self.config['sources']
        self.source_names = list(self.sources.keys())
        self.norm_stats = get_normalized_mean_std(self.sources)

        # Construire les transforms dynamiquement
        source_transforms    = build_source_transforms(self.sources)
        augmented_transforms = build_augmented_transforms(self.sources)

        # Nom de l'entrainement basé sur les paramètres
        branches_str = '_X_'.join('_-_'.join(branch) for branch in self.branches)
        self.NAME = f"{branches_str}"

        print(f"=======================================================")
        print(f"Sources utilisées : {self.branches} soit {branches_str}")
        print(f"=======================================================")

        # # Chemin par défaut pour les résultats
        self.RESULT_PATH = Path('/home/etardieu/Documents/my_data/these/V1/Resultats/2-Late_fusion/entrainement') / self.NAME

        # Chargement des datasets d'entraînement et de validation
        trainset = get_dataset(
            self.config['training']['dataset'],
            root                = self.root,
            mode                = 'train',
            sources             = self.sources,
            source_transforms   = source_transforms,
            augmented_transform = augmented_transforms['train_augmented_transform']
        )

        valset = get_dataset(
            self.config['training']['dataset'],
            root                = self.root,
            mode                = 'val',
            sources             = self.sources,
            source_transforms   = source_transforms,
            augmented_transform = augmented_transforms['val_augmented_transform']
        )

        # Création des DataLoaders
        kwargs = {'num_workers': self.config['training']['workers'], 'pin_memory': True} if self.config['training']['use_cuda'] else {}
        self.trainloader = data.DataLoader(trainset, batch_size=self.config['training']['batch_size'], drop_last=True,  shuffle=True, **kwargs)
        self.valloader   = data.DataLoader(valset,   batch_size=self.config['training']['batch_size'], drop_last=False, shuffle=True, **kwargs)

        self.train_step = max(len(self.trainloader) // 4, 1)
        self.val_step   = max(len(self.valloader)   // 4, 1)
        self.nclass     = trainset.num_class

        # Initialisation du modèle
        pass_rff = [True] * len(self.branches)
        model = get_mipanet(
            dataset       = self.config['training']['dataset'],
            branches      = self.branches,
            sources       = self.sources,
            pass_rff      = pass_rff,
            first_fusions = self.config['encoder']['first_fusions'],
            last_fusion   = self.config['encoder']['last_fusion'],
            use_tgcc      = self.config['TGCC']['use_TGCC'],
        )

        # Toutes les branches utilisent des poids pré-entraînés (ResNet18 adapté)
        # → groupe 0 = tous les backbones (lr bas), groupe 1 = décodeur + fusion (lr normal)
        base_modules = list(model.encoder.encoder.branch_bases)
        base_ids     = get_param_ids(base_modules)
        base_params  = filter(lambda p: id(p) in base_ids, model.parameters())
        other_params = filter(lambda p: id(p) not in base_ids, model.parameters())

        self.optimizer = select_optimizer(
            self.config['training']['optimizer'],
            [{'params': base_params, 'lr': self.config['training']['lr']/self.ratio_lr},
             {'params': other_params, 'lr': self.config['training']['lr']}],
            self.config['training']
        )

        # Configuration du scheduler
        self.scheduler = SchedulerWithHead(
            self.optimizer,
            self.config['training']['scheduler_type'],
            config =self.config['training'],
            steps_per_epoch=len(self.trainloader),
            ratio_lr=self.ratio_lr)

        # Définition de la fonction de perte
        self.criterion = CombinedLoss(config=self.config['training'])

        # Initialisation de l'early stopping
        self.early_stopping = EarlyStopping(patience=self.config['training']['early_stopping_patience'])

        # Utilisation de CUDA si disponible
        use_cuda = self.config['training']['use_cuda'] and torch.cuda.is_available()
        self.device = torch.device("cuda:0" if use_cuda else "cpu")
        if use_cuda:
            if torch.cuda.device_count() > 1:
                GPUS = list(range(torch.cuda.device_count()))
                print("Utilisation de", torch.cuda.device_count(), "GPUs !")
                model = nn.DataParallel(model, device_ids=GPUS)
                self.multi_gpu = True
            else:
                self.multi_gpu = False
        else:
            print("CUDA non dispo → CPU")
            self.multi_gpu = False

        self.best_pred = (0.0, 0.0)

        self.criterion = self.criterion.to(self.device)
        self.model = model.to(self.device)

        # Mixed precision training (AMP)
        self.scaler = GradScaler("cuda")

        if self.config['training']['wandb_activate'] == True:
            self.hyperparam['branches'] = self.branches
            self.hyperparam['pass_rff'] = [True] * len(self.branches)
            wandb.init(project="V1", name=self.NAME, config=self.hyperparam)
            wandb.watch(self.model, log="all")

        # Initialisation du fichier CSV pour sauvegarder les métriques
        mkdir(self.RESULT_PATH)
        self.csv_path = self.RESULT_PATH / "metrics.csv"
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch',
                'train_mean_iou', 'train_pixel_acc', 'train_loss', 'train_iou_vieille_foret',
                'val_mean_iou', 'val_pixel_acc', 'val_loss', 'val_iou_vieille_foret'
            ])
        print(f"Métriques CSV seront sauvegardées dans : {self.csv_path}")

    def _sources_to_device(self, sources_dict):
        """Transfère toutes les sources sur le device."""
        return {name: tensor.to(self.device) for name, tensor in sources_dict.items()}

    def denormalize(self, tensor, mean, std):
        mean = torch.tensor(mean).view(-1, 1, 1).to(tensor.device)
        std  = torch.tensor(std).view(-1, 1, 1).to(tensor.device)
        return tensor * std + mean

    def colorize_mask(self, mask):
        mask_np = mask.cpu().numpy()
        class_colors = {
            0: (0, 0, 0),        # noir = sol
            1: (127, 127, 127),  # gris = foret
            2: (255, 255, 255),  # blanc = vieille foret
        }
        h, w = mask_np.shape
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)
        for class_id, color in class_colors.items():
            color_mask[mask_np == class_id] = color
        return color_mask

    def run_lr_range_test(self, start_lr=1e-7, end_lr=1e-1, num_iters=200):
        """Lance le LR Range Test et affiche la courbe loss vs lr."""
        print("\n--- LR Range Test ---")
        lrs, losses = LRRangeTest.run(
            model      = self.model,
            optimizer  = self.optimizer,
            criterion  = self.criterion,
            dataloader = self.trainloader,
            device     = self.device,
            start_lr   = start_lr,
            end_lr     = end_lr,
            num_iters  = num_iters,
        )
        plt.figure(figsize=(8, 4))
        plt.plot(lrs, losses)
        plt.xscale('log')
        plt.xlabel('Learning rate (log scale)')
        plt.ylabel('Loss')
        plt.title('LR Range Test')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('lr_range_test.png', dpi=150)
        plt.show()
        print("Courbe sauvegardée dans lr_range_test.png")
        return lrs, losses

    def _visualize_source(self, sources_dict, source_name, batch_idx=0):
        """Dénormalise et prépare une source pour la visualisation."""
        tensor = sources_dict[source_name][batch_idx]
        stats  = self.norm_stats[source_name]

        # Prendre le premier canal si mono-canal
        if tensor.shape[0] == 1:
            vis = self.denormalize(tensor, stats['mean'], stats['std'])
        else:
            vis = self.denormalize(tensor, stats['mean'], stats['std'])

        # Min-max normalization pour l'affichage
        vis = (vis - vis.min()) / (vis.max() - vis.min() + 1e-8)
        vis = vis.clamp(0, 1)
        return vis

    def _build_composite_image(self, sources_dict, target_vis, pred_vis):
        """Assemble les modalités utilisées + target + prédiction en une grille matplotlib."""
        used_sources = [src for branch in self.branches for src in branch]
        panels = []
        for src_name in used_sources:
            vis = self._visualize_source(sources_dict, src_name)
            vis_np = vis.cpu().numpy()
            if vis_np.shape[0] == 1:
                panels.append((src_name, vis_np[0], "gray"))
            else:
                panels.append((src_name, np.transpose(vis_np, (1, 2, 0)), None))
        panels.append(("target", target_vis / 255.0, None))
        panels.append(("prédiction", pred_vis / 255.0, None))

        n = len(panels)
        fig, axes = plt.subplots(1, n, figsize=(8 * n, 8))
        if n == 1:
            axes = [axes]
        for ax, (title, img, cmap) in zip(axes, panels):
            ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
            ax.set_title(title, fontsize=10)
            ax.axis("off")
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        from PIL import Image as PILImage
        composite = np.array(PILImage.open(buf).convert('RGB'))
        plt.close(fig)
        return composite

    def training(self, epoch):
        """
        Entraîne le modèle pour une époque donnée.
        :param int epoch: Numéro de l'époque en cours d'entraînement.
        """
        train_loss = 0.0
        self.model.train()

        total_inter, total_union, total_correct, total_label, total_loss = 0, 0, 0, 0, 0
        train_loss_total = 0.0
        for i, (sources_dict, target, file_name) in tqdm(enumerate(self.trainloader), total=len(self.trainloader)):
            self.optimizer.zero_grad()

            sources_dict = self._sources_to_device(sources_dict)
            target = target.to(self.device)

            with autocast("cuda"):
                outputs = self.model(sources_dict)
                loss    = self.criterion(*outputs, target)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.config["training"]["scheduler_type"] != "plateau":
                self.scheduler.step()

            correct, labeled = batch_pix_accuracy(outputs[0].data, target)
            inter, union = batch_intersection_union(outputs[0].data, target, self.nclass)
            total_correct += correct
            total_label   += labeled
            total_inter   += inter
            total_union   += union
            train_loss    += loss.item()
            train_loss_total += loss.item()

            if (i + 1) % self.train_step == 0:
                avg_loss = train_loss / self.train_step
                print('Époque {}, étape {}, perte {}'.format(epoch + 1, i + 1, avg_loss))
                train_loss = 0.0

        pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
        IOU = 1.0 * total_inter / (np.spacing(1) + total_union)
        mIOU = IOU.mean()
        print('Époque {}, précision pixel {}, IOU moyen {}'.format(epoch + 1, pixAcc, mIOU))

        # Visualisation
        if len(self.trainloader) > 0:
            logits = outputs[0]
            pred_classes = torch.argmax(logits, dim=1)
            target_vis = self.colorize_mask(target[0])
            pred_vis = self.colorize_mask(pred_classes[0])

        if self.config['training']['wandb_activate'] == True:
            train_metrics = {
                "train/loss": train_loss_total / len(self.trainloader),
                "train/epoch": epoch,
                "train/pixel_acc": pixAcc,
                "train/mean_iou": mIOU,
                "lr/base": self.optimizer.param_groups[0]['lr'],
            }
            if len(self.optimizer.param_groups) > 1:
                train_metrics["lr/other"] = self.optimizer.param_groups[1]['lr']

            class_names = {0: "sol", 1: "foret", 2: "vieille_foret"}
            for class_idx in range(self.nclass):
                class_name = class_names.get(class_idx, f"class_{class_idx}")
                train_metrics[f"train/iou_{class_name}"] = IOU[class_idx]

            if len(self.trainloader) > 0:
                composite = self._build_composite_image(sources_dict, target_vis, pred_vis)
                train_metrics["train/composite"] = wandb.Image(composite, caption=f"Composite : {file_name[0]}")

            wandb.log(train_metrics, step=epoch)

        avg_train_loss = train_loss_total / len(self.trainloader)
        return pixAcc, mIOU, IOU, avg_train_loss


    def train_n_evaluate(self):
        """
        Entraîne et évalue le modèle sur plusieurs époques.
        """
        best_metric = -1e9
        best_state_dict = None
        results = {'miou': [], 'pix_acc': []}

        for epoch in range(self.config['training']['epochs']):
            print("\n=============== Entraînement de l'époque {}/{} ==========================".format(epoch + 1, self.config['training']['epochs']))

            train_pixAcc, train_mIOU, train_IOU, train_loss = self.training(epoch)
            train_iou_vieille_foret = train_IOU[2] if len(train_IOU) > 2 else 0.0

            print('\n=============== Début de l\'évaluation, époque {} ===============\n'.format(epoch + 1))
            val_pixAcc, val_mIOU, val_loss, val_IOU = self.validation(epoch)
            print('Évaluation précision pixel {}, IOU moyen {}, perte {}'.format(val_pixAcc, val_mIOU, val_loss))

            # Sauvegarde des métriques dans le fichier CSV
            val_iou_vieille_foret = val_IOU[2] if len(val_IOU) > 2 else 0.0
            with open(self.csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch + 1,
                    train_mIOU, train_pixAcc, train_loss, train_iou_vieille_foret,
                    val_mIOU, val_pixAcc, val_loss, val_iou_vieille_foret
                ])

            if self.config["training"]["scheduler_type"] == "plateau":
                self.scheduler.step(val_loss)

            results['miou'].append(round(val_mIOU, 6))
            results['pix_acc'].append(round(val_pixAcc, 6))

            # Sauvegarde du meilleur modèle basée sur la métrique pondérée
            weighted_metric = (2 * val_iou_vieille_foret + val_mIOU + val_pixAcc) / 4
            print(f"Métrique pondérée: (2*IoU_vieille_foret + mIOU + PixAcc)/4 = (2*{val_iou_vieille_foret:.6f} + {val_mIOU:.6f} + {val_pixAcc:.6f})/4 = {weighted_metric:.6f}")
            best_metric = max(best_metric, weighted_metric)

            # Vérification de l'early stopping
            self.early_stopping.on_epoch_end(epoch, weighted_metric, self.model)
            if self.early_stopping.stop_training:
                print(f"\nEarly stopping activé à l'époque {epoch + 1}")
                print(f"Meilleure métrique combinée atteinte: {self.early_stopping.best_value:.6f} à l'époque {epoch + 1 - self.early_stopping.wait}")
                break

            # Sauvegarde du modèle si c'est le meilleur jusqu'à présent
            is_best = False
            new_pred = (round(val_mIOU, 6), round(val_pixAcc, 6))
            if weighted_metric > getattr(self, 'best_weighted_metric', 0.0):
                is_best = True
                self.best_pred = new_pred
                self.best_weighted_metric = weighted_metric
                best_state_dict = self.model.module.state_dict() if self.multi_gpu else self.model.state_dict()
                print(f"Nouveau meilleur modèle! Métrique pondérée: {weighted_metric:.6f}")

                if is_best:
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': self.model.module.state_dict() if self.multi_gpu else self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'best_pred': self.best_pred,
                        'best_weighted_metric': self.best_weighted_metric,
                        'iou_vieille_foret': val_iou_vieille_foret,
                    }, self.config, is_best)

        # Moyenne des 5 dernières époques
        n_last = min(5, len(results['miou']))
        final_miou = sum(results['miou'][-n_last:]) / max(n_last, 1)
        final_pix_acc = sum(results['pix_acc'][-n_last:]) / max(n_last, 1)

        final_result = '\nPerformance des %d dernières époques\n[mIoU]: %4f\n[Pixel_Acc]: %4f\n[Meilleure Prédiction]: %s\n' % (
            n_last, final_miou, final_pix_acc, self.best_pred)
        print(final_result)

        # Exportation des poids si nécessaire
        format3_flag = (self.config['training']['dataset'] == 'format3' and final_miou > 0.1)
        print(format3_flag, self.config['training']['dataset'], round(final_miou, 2))

        if self.config['training']['export'] or format3_flag:
            export_info = '_'.join(sys.argv[1:-1] + [str(int(time.time()))])
            mkdir(self.RESULT_PATH)
            export_path = self.RESULT_PATH / f"{export_info}.pth"
            torch.save(best_state_dict, export_path)
            print(f'Exporté sous {export_path}')

            if self.config['training']['wandb_activate'] == True:
                wandb.log({
                    "export/model_path": str(export_path),
                    "export/final_miou": final_miou,
                    "export/final_pixel_acc": final_pix_acc,
                    "export/best_pred_miou": self.best_pred[0],
                    "export/best_pred_pixel_acc": self.best_pred[1]
                })

                artifact = wandb.Artifact(f"model-{self.NAME}", type="model")
                artifact.add_file(str(export_path))
                wandb.log_artifact(artifact)

        # Fermeture de la session WandB
        if self.config['training']['wandb_activate'] == True:
            wandb.finish()
            print("Session WandB fermée.")

        return best_metric


    def validation(self, epoch):
        """
        Évalue le modèle sur le jeu de validation.
            :param int epoch: Numéro de l'époque en cours d'évaluation.
            :return: Précision pixel, IOU moyen et perte moyenne.
            :rtype: tuple
        """
        def eval_batch(model, sources_dict, target):
            pred = model(sources_dict)
            loss = self.criterion(*pred, target)
            correct, labeled = batch_pix_accuracy(pred[0].data, target)
            inter, union = batch_intersection_union(pred[0].data, target, self.nclass)
            return correct, labeled, inter, union, loss, pred

        self.model.eval()
        total_inter, total_union, total_correct, total_label, total_loss = 0, 0, 0, 0, 0

        for i, (sources_dict, target, file_name) in enumerate(self.valloader):
            sources_dict = self._sources_to_device(sources_dict)
            target = target.to(self.device)

            with torch.no_grad():
                correct, labeled, inter, union, loss, pred = eval_batch(self.model, sources_dict, target)

            total_correct += correct
            total_label   += labeled
            total_inter   += inter
            total_union   += union
            total_loss    += loss.item()

            if i % self.val_step == 0:
                IOU_temp = 1.0 * total_inter / (np.spacing(1) + total_union)
                mIOU_temp = IOU_temp.mean()
                print('Évaluation IOU moyen {}'.format(mIOU_temp))

        # Calculer les métriques finales
        pixAcc   = 1.0 * total_correct / (np.spacing(1) + total_label)
        IOU      = 1.0 * total_inter / (np.spacing(1) + total_union)
        mIOU     = IOU.mean()
        loss_avg = total_loss / len(self.valloader)

        if len(self.valloader) > 0:
            # Target et prédiction
            logits = pred[0]
            pred_classes = torch.argmax(logits, dim=1)
            target_vis = self.colorize_mask(target[0])
            pred_vis   = self.colorize_mask(pred_classes[0])

            # Log dans WandB (UNE FOIS PAR ÉPOQUE)
            if self.config['training']['wandb_activate'] == True:
                val_metrics = {
                    "val/loss": loss_avg,
                    "val/epoch": epoch,
                    "val/pixel_acc": pixAcc,
                    "val/mean_iou": mIOU,
                }

                class_names = {0: "sol", 1: "foret", 2: "vieille_foret"}
                for class_idx in range(self.nclass):
                    class_name = class_names.get(class_idx, f"class_{class_idx}")
                    val_metrics[f"val/iou_{class_name}"] = IOU[class_idx]

                iou_vieille_foret = IOU[2] if len(IOU) > 2 else 0.0
                weighted_metric = (2 * iou_vieille_foret + mIOU + pixAcc) / 4
                val_metrics["val/weighted_metric"] = weighted_metric

                composite = self._build_composite_image(sources_dict, target_vis, pred_vis)
                val_metrics["val/composite"] = wandb.Image(composite, caption=f"Composite : {file_name[0]}")

                wandb.log(val_metrics, step=epoch)

        return pixAcc, mIOU, loss_avg, IOU


def train(root, branches):
    start_time = time.time()
    print("\n------- Début du programme ----------\n")

    set_reproducibility()

    trainer = Trainer(root, branches)
    trainer.train_n_evaluate()

    elapsed_secs = int(time.time() - start_time)
    hours, remainder = divmod(elapsed_secs, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"[Temps écoulé] : {hours}h {minutes}min {seconds}s")
