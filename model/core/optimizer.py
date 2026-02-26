import torch

def select_optimizer(optimizer_name, model_parameters, config):
    """
    Selects and returns the optimizer based on the provided name.
        :param str optimizer_name : The name of the optimizer to use.
        :param iterable model_parameters : The parameters of the model to optimize.
        :param dict config : Configuration dictionary containing optimizer settings.
        :return : The selected optimizer instance.
        :rtypes : torch.optim.Optimizer
    """
    
    if optimizer_name == "SGD" or optimizer_name == "sgd":
        return torch.optim.SGD(model_parameters, lr=config['lr'], momentum=config['momentum'], weight_decay=config['weight_decay'])

    elif optimizer_name == "AdamW" or optimizer_name == "Adamw" or optimizer_name == "adamw":
        return torch.optim.AdamW(model_parameters, lr=config['lr'], weight_decay=config['weight_decay'])

    elif optimizer_name == "Adam" or optimizer_name == "adam":
        return torch.optim.Adam(model_parameters, lr=config['lr'], weight_decay=config['weight_decay'])

    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")