"""
Ce module gère la configuration du projet.
"""

SOURCES = {
    'IRC':        {'dir': 'IRC',   'channels': 3, 'ext': '.tif', 'mean': [101, 71, 78], 'std': [42, 40, 37]},
    'biomasse':   {'dir': 'BIOM',  'channels': 1, 'ext': '.tif', 'mean': [202],         'std': [73]},
    'MNH':        {'dir': 'MNH',   'channels': 1, 'ext': '.tif', 'mean': [12.3],        'std': [9.5]},
    'historique': {'dir': 'HISTO', 'channels': 1, 'ext': '.tif', 'mean': [122],         'std': [42]},
}

# Chaque sous-liste = une branche d'encodeur.
# Les sources dans la même liste sont concaténées (early fusion).
# Exemples :
#   [['IRC'], ['MNH']]                                  → 2 branches late fusion
#   [['IRC', 'MNH'], ['historique']]                    → early fusion IRC+MNH, late fusion avec histo
#   [['IRC'], ['biomasse'], ['MNH'], ['historique']]    → 4 branches late fusion


# BRANCHES = [['IRC'], ['biomasse']]

def get_config():
    """
    Génère un dictionnaire de configuration pour l'entraînement, la phase de test et l'encodeur
    """
    CONFIG = {

        'training': {
            'wandb_activate': True,
            'commentaire'   : "run_du_wkend",

            'epochs'      : 55,
            'batch_size'  : 64,

            ### EARLY STOPPING CONFIGURATION ###
            'early_stopping_patience' : 15,

            ### OPTIMIZER CONFIGURATION ###
            'optimizer': "AdamW",                       # Choix de l'optimiseur, options: 'SGD', 'AdamW', 'Adam'
            'lr'          : 1e-3,
            'momentum'    : 0.9,                        # Seulement pour SGD
            'weight_decay': 5e-5,

            'aux_weight'  : 0,

            ### LEARNING RATE SCHEDULER CONFIGURATION ###
            'scheduler_type'    : 'onecycle',           # Options: 'plateau', 'cyclic', onecycle

            'plateau' : {
                'scheduler_factor'  : 0.5,              # lr * factor lorsque la métrique stagne
                'scheduler_patience': 7,
                },

            'onecycle': {
                'max_lr'      : 1e-3,                   # Learning rate maximum pour le scheduler onecycle
            },

            'cyclic': {
                'base_lr'     : 1e-4,                   # Learning rate minimum pour le scheduler cyclique
                'max_lr'      : 5e-4,                   # Learning rate maximum pour le scheduler cyclique
            },

            ### LOSS FUNCTION CONFIGURATION ###
            'first_loss':{
                'ce_weight'        : 0.4,               # Poids de la Cross Entropy dans la perte combinée
                'class_weight'     : [1.0, 1.0, 1.5],   # sol, foret, vieille foret
            },
            'second_loss':{
                'loss_type'     : 'dice',               # Options: 'dice' or 'focal_tversky'
                'focal_tversky' : {
                    'ft_alpha'         : 0.7,           #FN
                    'ft_beta'          : 0.3,           #FP
                    'ft_gamma'         : 1.33,
                }
            },

            ### AUTRES PARAMÈTRES D'ENTRAÎNEMENT ###
            'use_cuda'    : True,
            'seed'        : 150,
            'export'      : True,
            'dataset'     : "format2",
            'workers'     : 4,
            'train_split' : 'train',
        },

        'testing': {
            'use_cuda'  : True,
            'dataset'   : 'format3',
            'batch_size': 4,
            'workers'   : 4
        },

        'sources': SOURCES,
        # 'branches': BRANCHES,

        'encoder': {
            'first_fusions'     : 'PAM',
            'last_fusion'       : 'MIPA',
        },
        'TGCC': {
            'use_TGCC' : False,
            'TGCC_path': '/ccc/cont003/dsku/blanchet/home/user/inp/tardieue/MY_MIPANet_3_branches'
        },
    }

    HYPERPARAM = {
        'commentaire'        : CONFIG['training']['commentaire'],
        'epochs'             : CONFIG['training']['epochs'],
        'lr'                 : CONFIG['training']['lr'],
        'batch_size'         : CONFIG['training']['batch_size'],
        'momentum'           : CONFIG['training']['momentum'],
        'optimizer'          : CONFIG['training']['optimizer'],
        'scheduler_type'     : CONFIG['training']['scheduler_type'],
        'scheduler_factor'   : CONFIG['training']['plateau']['scheduler_factor'],
        'scheduler_patience' : CONFIG['training']['plateau']['scheduler_patience'],
        'max_lr_onecycle'    : CONFIG['training']['onecycle']['max_lr'],
        'base_lr'            : CONFIG['training']['cyclic']['base_lr'],
        'max_lr_cyclic'      : CONFIG['training']['cyclic']['max_lr'],
        'first_fusions'      : CONFIG['encoder']['first_fusions'],
        'last_fusion'        : CONFIG['encoder']['last_fusion'],
        'aux_weight'         : CONFIG['training']['aux_weight'],
        'weight_decay'       : CONFIG['training']['weight_decay'],
        'sources'            : list(CONFIG['sources'].keys()),
        'class_weight'       : CONFIG['training']['first_loss']['class_weight'],
        'ce_weight'          : CONFIG['training']['first_loss']['ce_weight'],
        'loss_type'          : CONFIG['training']['second_loss']['loss_type'],
        }
    return CONFIG, HYPERPARAM
