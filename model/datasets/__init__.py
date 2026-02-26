from .format2 import ForMat2

datasets = {
    'format2': ForMat2,
    'format3': ForMat2,
}

def get_dataset(name, root, mode, sources, source_transforms=None, augmented_transform=None):
    return datasets[name](root, mode, sources=sources, source_transforms=source_transforms, augmented_transform=augmented_transform)
