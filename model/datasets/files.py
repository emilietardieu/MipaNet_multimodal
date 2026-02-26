import torch
import shutil
from pathlib import Path


def save_checkpoint(state, args, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training state to a file.
        :param dict state: The state to save, containing epochs, state_dict, optimizer, and best_pred.
        :param dict args: The arguments setup in config.py, including dataset, epochs, batch size, and learning rate.
        :param bool is_best: Whether this checkpoint is the best one so far.
        :param str filename: The name of the file to save the checkpoint as.
    """
    directory = Path(f"runs/runs-{args['training']['dataset']}-ep_{args['training']['epochs']}-bs_{args['training']['batch_size']}-lr_{args['training']['lr']}")
    directory.mkdir(parents=True, exist_ok=True)
    filename = directory / filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, directory / 'model_best.pth.tar')

def mkdir(path):
    """
    Create a directory if it does not exist.
        :param str path: The path to the directory to create.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)