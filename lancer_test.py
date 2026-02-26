from pathlib import Path

from test import Tester

dossier  = "IRC_X_MNH"
poids    = "1771588396.pth"
branches = [['IRC'], ['MNH']]

dossier_path = Path(dossier)
poids_path   = Path(poids)

root_path   = Path(r"/home/etardieu/Documents/my_data/these/V1/Dataset/Dataset")
model_path  = Path(r"/home/etardieu/Documents/my_data/these/V1/Resultats/2-Late_fusion/entrainement") / dossier_path / poids_path
output_dir  = Path(r"/home/etardieu/Documents/my_data/these/V1/Resultats/2-Late_fusion/test_couleurs") / f"{dossier}_{poids_path.stem}"

tester = Tester(root_path, model_path, branches, output_dir)
tester.test(save_images=False, save_probability_maps=True, max_probability_maps=50)
