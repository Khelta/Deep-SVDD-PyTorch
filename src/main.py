import torch
import logging
import random
import numpy as np
import os
import sys

from utils.config import Config
from utils.visualization.plot_images_grid import plot_images_grid
from deepSVDD import DeepSVDD
from datasets.main import load_dataset


def main(dataset_name, net_name, xp_path, data_path, load_config, load_model, objective, nu, device, seed,
         optimizer_name, lr, n_epochs, lr_milestone, batch_size, weight_decay, pretrain, ae_optimizer_name, ae_lr,
         ae_n_epochs, ae_lr_milestone, ae_batch_size, ae_weight_decay, n_jobs_dataloader, normal_class, p=0.25):
    """
    Deep SVDD, a fully deep method for anomaly detection.

    :arg DATASET_NAME: Name of the dataset to load.
    :arg NET_NAME: Name of the neural network to use.
    :arg XP_PATH: Export path for logging the experiment.
    :arg DATA_PATH: Root path of data.
    """

    # Get configuration
    cfg = Config(locals().copy())

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = xp_path + '/log.txt'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Print arguments
    logger.info('Log file is %s.' % log_file)
    logger.info('Data path is %s.' % data_path)
    logger.info('Export path is %s.' % xp_path)

    logger.info('Dataset: %s' % dataset_name)
    logger.info('Normal class: %d' % normal_class)
    logger.info('Network: %s' % net_name)

    # If specified, load experiment config from JSON-file
    if load_config:
        cfg.load_config(import_json=load_config)
        logger.info('Loaded configuration from %s.' % load_config)

    # Print configuration
    logger.info('Deep SVDD objective: %s' % cfg.settings['objective'])
    logger.info('Nu-paramerter: %.2f' % cfg.settings['nu'])

    # Set seed
    if cfg.settings['seed'] != -1:
        random.seed(cfg.settings['seed'])
        np.random.seed(cfg.settings['seed'])
        torch.manual_seed(cfg.settings['seed'])
        logger.info('Set seed to %d.' % cfg.settings['seed'])

    # Default device to 'cpu' if cuda is not available
    if not torch.cuda.is_available():
        device = 'cpu'
    logger.info('Computation device: %s' % device)
    logger.info('Number of dataloader workers: %d' % n_jobs_dataloader)

    # Load data
    dataset = load_dataset(dataset_name, data_path, normal_class, p=p)

    # Initialize DeepSVDD model and set neural network \phi
    deep_SVDD = DeepSVDD(cfg.settings['objective'], cfg.settings['nu'])
    deep_SVDD.set_network(net_name)
    # If specified, load Deep SVDD model (radius R, center c, network weights, and possibly autoencoder weights)
    if load_model:
        deep_SVDD.load_model(model_path=load_model, load_ae=True)
        logger.info('Loading model from %s.' % load_model)

    logger.info('Pretraining: %s' % pretrain)
    if pretrain:
        # Log pretraining details
        logger.info('Pretraining optimizer: %s' % cfg.settings['ae_optimizer_name'])
        logger.info('Pretraining learning rate: %g' % cfg.settings['ae_lr'])
        logger.info('Pretraining epochs: %d' % cfg.settings['ae_n_epochs'])
        logger.info('Pretraining learning rate scheduler milestones: %s' % (cfg.settings['ae_lr_milestone'],))
        logger.info('Pretraining batch size: %d' % cfg.settings['ae_batch_size'])
        logger.info('Pretraining weight decay: %g' % cfg.settings['ae_weight_decay'])

        # Pretrain model on dataset (via autoencoder)
        deep_SVDD.pretrain(dataset,
                           optimizer_name=cfg.settings['ae_optimizer_name'],
                           lr=cfg.settings['ae_lr'],
                           n_epochs=cfg.settings['ae_n_epochs'],
                           lr_milestones=cfg.settings['ae_lr_milestone'],
                           batch_size=cfg.settings['ae_batch_size'],
                           weight_decay=cfg.settings['ae_weight_decay'],
                           device=device,
                           n_jobs_dataloader=n_jobs_dataloader)

    # Log training details
    logger.info('Training optimizer: %s' % cfg.settings['optimizer_name'])
    logger.info('Training learning rate: %g' % cfg.settings['lr'])
    logger.info('Training epochs: %d' % cfg.settings['n_epochs'])
    logger.info('Training learning rate scheduler milestones: %s' % (cfg.settings['lr_milestone'],))
    logger.info('Training batch size: %d' % cfg.settings['batch_size'])
    logger.info('Training weight decay: %g' % cfg.settings['weight_decay'])

    # Train model on dataset
    deep_SVDD.train(dataset,
                    optimizer_name=cfg.settings['optimizer_name'],
                    lr=cfg.settings['lr'],
                    n_epochs=cfg.settings['n_epochs'],
                    lr_milestones=cfg.settings['lr_milestone'],
                    batch_size=cfg.settings['batch_size'],
                    weight_decay=cfg.settings['weight_decay'],
                    device=device,
                    n_jobs_dataloader=n_jobs_dataloader)

    # Test model
    deep_SVDD.test(dataset, device=device, n_jobs_dataloader=n_jobs_dataloader)

    # Plot most anomalous and most normal (within-class) test samples
    indices, labels, scores = zip(*deep_SVDD.results['test_scores'])
    indices, labels, scores = np.array(indices), np.array(labels), np.array(scores)
    # idx_sorted = indices[labels == 0][np.argsort(scores[labels == 0])]  # sorted from lowest to highest anomaly score
    idx_sorted = indices[np.argsort(scores)]

    if dataset_name in ('mnist', 'fashion', 'cifar10', 'cifar100', 'svhn'):

        if dataset_name in ('mnist', 'fashion'):
            X_normals = dataset.train_set_all.train_data[idx_sorted[:32], ...].unsqueeze(1)
            X_outliers = dataset.train_set_all.train_data[idx_sorted[-32:], ...].unsqueeze(1)

        if dataset_name in ('cifar10', 'cifar100'):
            X_normals = torch.tensor(np.transpose(dataset.train_set_all.data[idx_sorted[:32], ...], (0, 3, 1, 2)))
            X_outliers = torch.tensor(np.transpose(dataset.train_set_all.data[idx_sorted[-32:], ...], (0, 3, 1, 2)))

        if dataset_name == "svhn":
            X_normals = torch.tensor(np.transpose(dataset.train_set_all.data[idx_sorted[:32], ...], (0, 1, 2, 3)))
            X_outliers = torch.tensor(np.transpose(dataset.train_set_all.data[idx_sorted[-32:], ...], (0, 1, 2, 3)))

        plot_images_grid(X_normals, export_img=xp_path + '/normals', title='Most normal examples', padding=2)
        plot_images_grid(X_outliers, export_img=xp_path + '/outliers', title='Most anomalous examples', padding=2)

    # Save results, model, and configuration
    deep_SVDD.save_results(export_json=xp_path + '/results.json')
    cfg.save_config(export_json=xp_path + '/config.json')
    # deep_SVDD.save_model(export_model=xp_path + '/model.tar')


abs_path = os.path.dirname(os.path.realpath(__file__))

if __name__ == '__main__':
    algorithms = ["mnist", "fashion", "cifar10", "cifar100", "svhn"]
    net = {"mnist": "mnist_LeNet",
           "fashion": "mnist_LeNet",
           "cifar10": "cifar10_LeNet",
           "cifar100": "cifar10_LeNet",
           "svhn": "cifar10_LeNet"}
    if len(sys.argv) != 2:
        raise ValueError("Argument missing. Must be " + str(algorithms))
    elif sys.argv[1] not in algorithms:
        raise ValueError('ARGV must be in ' + str(algorithms))

    algorithm = sys.argv[1]

    for cycle in range(5):
        for label in range(10):
            for p in [0.05, 0.15, 0.25]:
                path_to_log = os.path.join(abs_path, "../log/{}/{}/{}/{}".format(algorithm, p, cycle, label))
                path_to_data = os.path.join(abs_path, "../data")
                if not os.path.exists(path_to_log):
                    os.makedirs(path_to_log)

                onlyfiles = next(os.walk(path_to_log))[2]
                if len(onlyfiles) < 2:
                    main(algorithm, net[algorithm], path_to_log, path_to_data, objective='one-class', nu=0.1,
                         device="cuda", seed=-1, optimizer_name="adam", lr=0.001, n_epochs=30, lr_milestone=(), batch_size=256,
                         weight_decay=1e-6, pretrain=False, ae_optimizer_name="adam", ae_lr=0.001, ae_n_epochs=100, ae_lr_milestone=(),
                         ae_batch_size=256, ae_weight_decay=1e-6, n_jobs_dataloader=0, normal_class=label, load_config=None, load_model=None, p=p)
                else:
                    print("\n Skipping Label {} p {} Cycle {}".format(label, p, cycle))
