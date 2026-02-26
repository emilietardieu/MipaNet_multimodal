import itertools
import torch
from torch.amp import autocast, GradScaler
from tqdm import tqdm


class LRRangeTest:
    @staticmethod
    def run(model, optimizer, criterion, dataloader, device,
            start_lr=1e-7, end_lr=1e-1, num_iters=200):
        """
        LR Range Test (Smith 2017).
        Fait monter le lr exponentiellement de start_lr à end_lr sur num_iters batches
        et enregistre la loss à chaque pas.
        Le lr loggé est celui du groupe 1 (tête, référence).
        Le groupe 0 (backbone) suit le même schéma mais avec un lr divisé par 3.33.
            param model: Modèle à tester
            param optimizer: Optimiseur utilisé pour l'entraînement
            param criterion: Fonction de perte utilisée pour l'entraînement
            param dataloader: DataLoader pour fournir les données d'entraînement
            param device: Appareil (CPU/GPU) sur lequel exécuter le test
            param float start_lr: Taux d'apprentissage de départ (ex: 1e-7)
            param float end_lr: Taux d'apprentissage de fin (ex: 1e-1)
            param int num_iters: Nombre total d'itérations (batches) pour faire monter le lr
            return: lrs, losses: Listes des taux d'apprentissage et des pertes correspondantes à chaque itération
            rtype: (list, list)
        """
        init_model_state = {k: v.clone() for k, v in model.state_dict().items()}
        init_optim_state = optimizer.state_dict()

        n_groups = len(optimizer.param_groups)

        # Initialiser le lr de référence (groupe 1 si 2 groupes, groupe 0 sinon)
        ref_group = 1 if n_groups > 1 else 0
        for g in optimizer.param_groups:
            g["lr"] = start_lr
        if n_groups > 1:
            optimizer.param_groups[0]["lr"] = start_lr / 3.33

        mult = (end_lr / start_lr) ** (1 / num_iters)
        scaler = GradScaler("cuda")

        lrs, losses = [], []
        model.train()

        for it, (sources_dict, target, _) in tqdm(enumerate(itertools.cycle(dataloader)), total=num_iters, desc="LR Range Test"):
            if it >= num_iters:
                break

            sources_dict = {k: v.to(device) for k, v in sources_dict.items()}
            target = target.to(device)

            optimizer.zero_grad()
            with autocast("cuda"):
                outputs = model(sources_dict)
                loss = criterion(*outputs, target)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            loss_val = loss.item()
            lrs.append(optimizer.param_groups[ref_group]["lr"])
            losses.append(loss_val)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[LR Range Test] Loss NaN/Inf à lr={lrs[-1]:.2e} — arrêt.")
                break

            # Faire monter le lr
            optimizer.param_groups[ref_group]["lr"] *= mult
            if n_groups > 1:
                optimizer.param_groups[0]["lr"] = optimizer.param_groups[ref_group]["lr"] / 10

            it += 1

        # Restaurer l'état initial
        model.load_state_dict(init_model_state)
        optimizer.load_state_dict(init_optim_state)

        return lrs, losses
