from torch.utils.data import DataLoader
from model import *
from data import *
from helpers import *
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
import os

# read in configurations
with open("./config.yml", "r") as f:
    cfg = yaml.safe_load(f)
    cfg = cfg['configs']

# either simulate or read in data depending on configurations
if cfg['which_data'] == 'sim':
    nclass = cfg['nclass']
    nitems = cfg['nitems']
    N = cfg['N']

    # Generate class probabilities
    np.random.seed(1)
    class_probs = np.ones(nclass) / nclass

    # Generate conditional probabilities
    cond_probs = np.random.uniform(0.1, 0.9, size=(nclass, nitems))

    # Generate true class membership for each person
    true_class_ix = np.random.choice(np.arange(nclass), size=(N,), p=class_probs)
    true_class = np.zeros((N, nclass))
    true_class[np.arange(N), true_class_ix] = 1

    # simulate responses
    prob = true_class @ cond_probs
    data = np.random.binomial(1, prob).astype(float)
elif cfg['which_data'] == 'R':
    data = genfromtxt('Rdata/data.csv', delimiter=',', skip_header=1) - 1
    cond_probs = genfromtxt('Rdata/cond_probs.csv', delimiter=',', skip_header=1)
    true_class = genfromtxt('Rdata/true_class.csv', delimiter=',', skip_header=1)
    true_class_ix = np.argmax(true_class, 1)
else:
    raise ValueError('Unknown value for which_data')


# create pytorch dataset
dataset = MemoryDataset(data)
train_loader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True)
test_loader = DataLoader(dataset, batch_size=cfg['N'], shuffle=False)

# repeat process cfg['n_rep'] times to prevent local minima
best = -float('inf')
for i in range(cfg['n_rep']):
    if cfg['model'] == 'vq':
        # initiate model
        model = VQVAE(dataloader=train_loader,
                      nitems=cfg['nitems'],
                      learning_rate=cfg['learning_rate'],
                      latent_dims=cfg['nclass'],
                      hidden_layer_size=50,
                      emb_dim=10,
                      beta=1)
    elif cfg['model'] == 'gs':
        model = GSVAE(dataloader=train_loader,
                      nitems=cfg['nitems'],
                      learning_rate=cfg['learning_rate'],
                      latent_dims = cfg['nclass'],
                      hidden_layer_size=50,
                      beta=1,
                      temperature=cfg['temperature'],
                      temperature_decay=cfg['temperature_decay'],
                      n_iw_samples=cfg['n_iw_samples'])
    elif cfg['model'] == 'rbm':
        nnodes = np.log2(cfg['nclass'])
        if not nnodes.is_integer():
            raise ValueError('RBM only implemented for 2, 4, 8, ... classes')
        model = RestrictedBoltzmannMachine(
            dataloader=train_loader,
            n_visible=cfg['nitems'],
            n_hidden=int(nnodes),
            learning_rate=cfg['learning_rate'],
            n_gibbs=cfg['gibbs_samples']
        )
    elif cfg['model'] == 'dvae':
        nnodes = np.log2(cfg['nclass'])
        if not nnodes.is_integer():
            raise ValueError('DVAE only implemented for 2, 4, 8, ... classes')
        model = DiscreteVAE(dataloader=train_loader,
                            nitems=cfg['nitems'],
                            latent_dims=int(nnodes),
                            hidden_layer_size=50,
                            learning_rate=cfg['learning_rate'])
    else:
        raise ValueError('Invalid model type')
    # initialise model and optimizer

    if os.path.exists('logs/all/version_0/metrics.csv'):
        os.remove('logs/all/version_0/metrics.csv')

    logger = CSVLogger("logs", name='all', version=0)
    trainer = Trainer(fast_dev_run=cfg['single_epoch_test_run'],
                      max_epochs=cfg['max_epochs'],
                      logger=logger,
                      callbacks=[
                      EarlyStopping(monitor='train_loss', min_delta=cfg['min_delta'], patience=cfg['patience'],
                                    mode='min')])

    trainer.fit(model)

    # compute the estimated conditional probabilities and the posterior probabilities on the test data
    test_data = next(iter(test_loader))
    if cfg['model'] == 'gs':
        log_pi = model.encoder(test_data)
        pred_class = model.GumbelSoftmax(log_pi).detach().numpy()
        pred_class_ix = np.argmax(pred_class, 1)

        # get the conditional probabilities
        est_probs = model.decoder(torch.eye(cfg['nclass'])).detach().numpy()
    elif cfg['model'] == 'dvae':
        # log
        log_pi = model.encoder(test_data)
        p1 =  model.SaE(log_pi.exp()).detach().numpy()

        # Get the number of hidden nodes
        num_hidden_nodes = p1.shape[1]

        # Initialize an array to store the probabilities for each configuration
        probabilities = []
        est_probs = np.zeros((cfg['nclass'], model.nitems))
        # Iterate over all possible binary configurations
        for i in range(cfg['nclass']):
            binary_config = np.array([int(x) for x in bin(i)[2:].zfill(num_hidden_nodes)])
            # Compute the probability for the current configuration
            probability = np.prod(p1 ** binary_config * (1 - p1) ** (1 - binary_config), axis=1)
            probabilities.append(probability)

            est_probs[i, :] = model.decoder(torch.Tensor(binary_config)).detach().numpy()

        # Stack the probabilities to create a Nx2^M tensor
        pred_class = np.column_stack(probabilities)
        pred_class_ix = np.argmax(pred_class, 1)
        print(pred_class)
    elif cfg['model'] == 'vq':
        ze = model.encoder(test_data)
        _, pred_class_ix, _ = model.VQSampler(ze)
        pred_class_ix = pred_class_ix.detach().numpy()
        pred_class = np.zeros_like(true_class)
        pred_class[np.arange(cfg['N']), pred_class_ix] = 1

        embs = model.VQSampler.embeddings.weight
        est_probs = model.decoder(embs).detach().numpy()

    elif cfg['model'] == 'rbm':
        _, p1 = model.sample_h(next(iter(test_loader)))
        p1 = p1.detach().numpy()
        # Get the number of hidden nodes
        num_hidden_nodes = p1.shape[1]

        # Initialize an array to store the probabilities for each configuration
        probabilities = []
        binary_configs = []
        # Iterate over all possible binary configurations
        for i in range(cfg['nclass']):
            binary_config = np.array([int(x) for x in bin(i)[2:].zfill(num_hidden_nodes)])
            binary_configs.append(binary_config)
            # Compute the probability for the current configuration
            # TODO: make this more readable! and extend to multiple nodes
            probability = np.prod(p1 ** binary_config * (1 - p1) ** (1 - binary_config), axis=1)
            probabilities.append(probability)

        # Stack the probabilities to create a Nx2^M tensor
        pred_class = np.column_stack(probabilities)
        pred_class_ix = np.argmax(pred_class, 1)

        est_probs = np.zeros((cfg['nclass'], cfg['nitems']))
        for i in range(cfg['nclass']):
            _, probs = model.sample_v(torch.Tensor(binary_configs[i]))
            est_probs[i, :] = probs.detach().numpy()

    # plot training loss
    logs = pd.read_csv(f'logs/all/version_0/metrics.csv')
    plt.plot(logs['epoch'], logs['train_loss'])
    plt.title('Training loss')
    plt.savefig(f'./figures/all/training_loss.png')
    # true_probs = pd.read_csv('data/true_probs.csv', index_col=0).to_numpy()

    # Compute log likelihood
    log_likelihood = np.sum(data * np.log(pred_class.dot(est_probs) + 1e-10) +
                            (1 - data) * np.log(1 - pred_class.dot(est_probs) + 1e-10))

    # save class predictions and conditional probanilities only for the model with the highest likelihood
    if log_likelihood > best:
        best = log_likelihood
        best_class_ix = pred_class_ix
        best_cond_probs = est_probs

# match estimated latent classes to the correct true class
_, new_order = linear_sum_assignment(-Cor(best_cond_probs, cond_probs))
cond_probs = cond_probs[new_order, :]
true_class = true_class[:, new_order]
true_class_ix = np.argmax(true_class, 1)

# flatten probs for plotting
true_probs = cond_probs.flatten()
est_probs = best_cond_probs.flatten()


plt.figure()
mse = MSE(est_probs, true_probs)
plt.scatter(y=est_probs, x=true_probs)
plt.plot(true_probs, true_probs)
plt.title(f'Probability estimation plot:, MSE={round(mse, 4)}')
plt.xlabel('True values')
plt.ylabel('Estimates')
plt.savefig(f'./figures/all/probabilities.png')

# print latent class accuracy
print(f'class accuracy: {np.mean(best_class_ix == true_class_ix)}')



