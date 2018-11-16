mngu0_artinv_DNN = {
    'experiment_name': 'articulatory_inversion',  # the name of the folder to save the experiment
    'network_type': 'DNN',
    'dataset': 'mngu0',  # the name of the folder under the experiment
    'task': 'regression',  # regression

    # Dataset parameters
    'feature_type': 'fbank',
    'feature_deltas': True,
    'feature_context': 5,
    'label_deltas': False,
    'monophone_targets': False,
    'ali_dir': 'tri3_ali',

    # Network parameters (all lowercase)
    'num_hidden_layers': 3,
    'layer_size': 300,
    'activation': 'elu',
    'batch_normalization': True,
    'gaussian_noise': 0.01,
    'weight_decay': 0.00001,
    'dropout': 0.3,
    'clipnorm': 0,

    # Training parameters (all lowercase)
    'optimizer': 'adam',  # sgd
    'learning_rate': 0.1,
    'batch_size': 128,
    'epochs': 50
}

mngu0_artinv_LSTM = {
    'experiment_name': 'articulatory_inversion',  # the name of the folder to save the experiment
    'network_type': 'LSTM',
    'dataset': 'mngu0',  # the name of the folder under the experiment
    'task': 'regression',  # regression

    # Dataset parameters
    'feature_type': 'fbank',
    'feature_deltas': True,
    'feature_context': 5,
    'label_deltas': False,
    'monophone_targets': False,
    'ali_dir': 'tri3_ali',

    # Network parameters (all lowercase)
    'bidirectional': True,
    'timestep': 50,
    'num_layers': 2,
    'layer_size': 300,
    'batch_normalization': False,
    'gaussian_noise': 0.01,
    'dropout': 0.1,
    'weight_decay': 0,
    'clipnorm': 0,

    # Training parameters (all lowercase)
    'optimizer': 'adam',  # sgd
    'learning_rate': 0.1,
    'batch_size': 64,
    'epochs': 50
}

mngu0_artinv_CNN = {
    'experiment_name': 'articulatory_inversion',  # the name of the folder to save the experiment
    'network_type': 'CNN',
    'dataset': 'mngu0',  # the name of the folder under the experiment
    'task': 'regression',  # regression

    # Dataset parameters
    'feature_type': 'fbank',
    'feature_deltas': True,
    'feature_context': 5,
    'label_deltas': False,
    'monophone_targets': False,
    'ali_dir': 'tri3_ali',

    # Network parameters (all lowercase)
    'conv1d': True,
    'kernel_size': 6,
    'pool_size': 2,
    'filters': 90,
    'padding': 'causal',
    'num_cnn_layers': 3,
    'num_ff_layers': 1,
    'ff_layer_size': 400,
    'activation': 'elu',
    'batch_normalization': True,
    'gaussian_noise': 0.1,
    'weight_decay': 0.0,
    'dropout': 0.2,
    'clipnorm': 0,

    # Training parameters (all lowercase)
    'optimizer': 'adam',  # sgd
    'learning_rate': 0.1,
    'batch_size': 128,
    'epochs': 50
}
