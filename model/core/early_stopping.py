class EarlyStopping:
    """
    Classe pour implémenter un mécanisme d'early stopping.
        :param int patience: Nombre d'époques à attendre après la dernière amélioration avant d'arrêter l'entraînement.
        :param float min_delta: Seuil minimum d'amélioration pour considérer que la performance s'est améliorée.
        :param str monitor: Nom de la métrique à surveiller (ex: 'val_accuracy').
        :param bool restore_best_weights: Si True, restaure les poids du modèle à ceux de la meilleure époque après l'arrêt de l'entraînement.
    """
    def __init__(self, patience=20, min_delta=0.001, monitor='val_accuracy', restore_best_weights=True):
        self.patience   = patience
        self.min_delta  = min_delta
        self.monitor    = monitor
        self.restore_best_weights = restore_best_weights
        self.best_value = None
        self.best_weights = None
        self.wait = 0
        self.stopped_epoch = 0
        self.stop_training = False

    def on_epoch_end(self, epoch, current_value, model):
        """
        Vérifie si l'entraînement doit être arrêté après chaque époque.
            :param epoch: Numéro de l'époque actuelle.
            :param current_value: Valeur actuelle de la métrique surveillée.
            :param model: Modèle en cours d'entraînement.
        """
        print(f"Early Stopping - Époque {epoch}: current_value={current_value:.6f}, best_value={self.best_value}, wait={self.wait}/{self.patience}")
        
        if self.best_value is None or current_value > self.best_value + self.min_delta:
            print(f"Early Stopping - Amélioration détectée! Nouvelle meilleure valeur: {current_value:.6f}")
            self.best_value   = current_value
            self.best_weights = model.state_dict() if self.restore_best_weights else None
            self.wait         = 0
        else:
            self.wait += 1
            print(f"Early Stopping - Pas d'amélioration. Compteur: {self.wait}/{self.patience}")
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.stop_training = True
                print(f"Early Stopping - Seuil de patience atteint! Arrêt de l'entraînement.")
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                    print(f"Early Stopping - Restauration des meilleurs poids (époque {epoch - self.patience})")
