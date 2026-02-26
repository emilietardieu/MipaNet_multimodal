from torch.optim.lr_scheduler import CyclicLR, OneCycleLR, ReduceLROnPlateau


class SchedulerWithHead:
    """
    Wrapper pour les schedulers de lr :
      - CyclicLR
      - OneCycleLR
      - ReduceLROnPlateau
    avec un facteur sur les lr des groupes > 0 (tête du modèle).
    """
    def __init__(self, optimizer, scheduler_name, config, steps_per_epoch, ratio_lr):
        """
        L'init de ce wrapper prend en charge la création du scheduler choisi et applique un facteur de division sur le groupe 0 (backbone) par rapport au groupe 1 (tête).
            param optimizer: Optimiseur utilisé pour l'entraînement
            param scheduler_name: Nom du scheduler à utiliser ("cyclic", "onecycle", "plateau")
            param config: Dictionnaire de configuration contenant les paramètres spécifiques à chaque scheduler
            param steps_per_epoch: Nombre de batches par époque (utilisé pour CyclicLR et OneCycleLR)
            param ratio_lr: Facteur de division pour le groupe 0 (backbone) par rapport au groupe 1 (tête)
        """
        self.optimizer = optimizer
        name = scheduler_name.lower()
        self.ratio_lr = ratio_lr

        if name == "cyclic":
            self.scheduler = CyclicLR(
                optimizer,
                base_lr        =config['cyclic']["base_lr"],
                max_lr         =config['cyclic']["max_lr"],
                step_size_up   =4 * steps_per_epoch,  # 4 époques pour un cycle complet
                cycle_momentum =False
            )

        elif name == "onecycle":
            self.scheduler = OneCycleLR(
                optimizer,
                max_lr          =config['onecycle']["max_lr"],
                epochs          =config["epochs"],
                steps_per_epoch =steps_per_epoch,
            )

        elif name == "plateau":
            self.scheduler = ReduceLROnPlateau(
                optimizer,
                mode     ="min",
                factor   = config['plateau']["scheduler_factor"],
                patience = config['plateau']["scheduler_patience"]
            )

        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    def _apply_head_factor(self):
        """Maintient le groupe 0 (préentraîné) à 10x moins que le groupe 1 (tête)."""
        if len(self.optimizer.param_groups) > 1:
            head_lr = self.optimizer.param_groups[1]['lr']
            self.optimizer.param_groups[0]['lr'] = head_lr / self.ratio_lr

    def step(self, metric=None):
        """Step pour plateau ou normal."""
        if isinstance(self.scheduler, ReduceLROnPlateau):
            self.scheduler.step(metric)
        else:
            self.scheduler.step()

        self._apply_head_factor()
