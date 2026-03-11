"""
Lancement de l'inférence MIPANet sur un nouveau territoire.

Renseigner :
  - root_path  : répertoire contenant les sous-dossiers des sources (IRC/, MNH/, ...)
  - model_path : chemin vers le fichier .pth du modèle
  - branches   : architecture du modèle (même syntaxe que pour l'entraînement)
  - output_dir : répertoire de sortie (optionnel, créé automatiquement sinon)

Structure attendue de root_path :
    root_path/
      IRC/      *.tif
      MNH/      *.tif    (si utilisé)
      BIOM/     *.tif    (si utilisé)
      predict.txt        (optionnel — une ligne par stem à traiter)

Sorties générées :
    output_dir/
      predictions/   {stem}.tif              — masque de classes géoréférencé (uint8 : 0/1/2)
      heatmaps/      {stem}_heatmap.tif      — probabilité vieille forêt géoréférencée (float32)
"""

from pathlib import Path
from inference import Inferencer

# ================================================================
# CONFIGURATION — à modifier selon l'expérience
# ================================================================

dossier  = "IRC"
poids    = "1772188618.pth"

# Architecture identique à celle utilisée lors de l'entraînement
# Exemples :
#   [['IRC']]                        → mono-branche IRC
#   [['IRC'], ['MNH']]               → 2 branches late fusion
#   [['IRC', 'MNH'], ['biomasse']]   → early fusion IRC+MNH + branche biomasse
branches = [['IRC']]

# Territoire à prédire (pas de MASK ni de fichier de split nécessaires)
root_path = Path(r"/home/etardieu/Documents/my_data/these/pnr_ariege/Dataset")

# Chemin vers le modèle entraîné
model_path = (
    Path(r"/home/etardieu/Documents/my_data/these/V1/Resultats/1-Modalite/1-modalite_V3")
    / dossier
    / poids
)

# Répertoire de sortie (laisser None pour créer automatiquement dans root_path)
output_dir = Path(r"/home/etardieu/Documents/my_data/these/pnr_ariege/Resultat/IRC") / f"{dossier}_{Path(poids).stem}"

# ================================================================

inferencer = Inferencer(root_path, model_path, branches, output_dir)
inferencer.run()
