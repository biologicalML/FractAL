'''
A dictionary containing the model type and dataset specific model params to use
'''


dm_dict = {
    "diamonds": {
        "type": "mlp",
        "kwargs":{
            "hidden_sizes": [512]*2
        }
    },
    "sarcos": {
        "type": "mlp",
        "kwargs":{
            "hidden_sizes": [512]*2
        }
    },
    "kegg_undir_uci": {
        "type": "mlp",
        "kwargs":{
            "hidden_sizes": [512]*2
        }
    },
    "bio_macrophage": {
        "type": "mlp",
        "kwargs":{
            "hidden_sizes": [],
            "n_outputs": 4000
        }
    },
    "bio_tcell": {
        "type": "mlp",
        "kwargs":{
            "hidden_sizes": [],
            "n_outputs": 4000
        }
    },
    "cifar10":{
        "type": "resnet18",
        "kwargs":{
            "layers": [2,2,2,2],
            "num_classes" : 10
        }
    },
}