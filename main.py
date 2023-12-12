import torch
from torch.utils.data import DataLoader
from model import *
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from scipy.optimize import linear_sum_assignment
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from data import *
from scipy.stats import pearsonr

#if len(sys.argv) > 1:
#    cfg['nclass'] = int(sys.argv[1])
#    cfg['temperature_decay'] = float(sys.argv[2])

def MSE(est, true):
    """
    Mean square error
    Parameters
    ----------
    est: estimated parameter values
    true: true paremters values

    Returns
    -------
    the MSE
    """
    return np.mean(np.power(est-true,2))


def Cor(x, y):
    """Correlate each n with each m.

    Parameters
    ----------
    x : np.array
      Shape N X T.

    y : np.array
      Shape M X T.

    Returns
    -------
    np.array
      N X M array in which each element is a correlation coefficient.

    """
    mu_x = x.mean(1)
    mu_y = y.mean(1)
    n = x.shape[1]
    if n != y.shape[1]:
        raise ValueError('x and y must ' +
                         'have the same number of timepoints.')
    s_x = x.std(1, ddof=n - 1)
    s_y = y.std(1, ddof=n - 1)
    cov = np.dot(x,
                 y.T) - n * np.dot(mu_x[:, np.newaxis],
                                  mu_y[np.newaxis, :])
    return cov / np.dot(s_x[:, np.newaxis], s_y[np.newaxis, :])


with open("./config.yml", "r") as f:
    cfg = yaml.safe_load(f)
    cfg = cfg['configs']

# initialise model and optimizer
logger = CSVLogger("logs", name=cfg['which_data'], version=0)
trainer = Trainer(fast_dev_run=cfg['single_epoch_test_run'],
                  max_epochs=cfg['max_epochs'],
                  logger=logger,
                  callbacks=[
                      EarlyStopping(monitor='train_loss', min_delta=cfg['min_delta'], patience=cfg['patience'],
                                    mode='min')])

# dataset = CSVDataset('data/data.csv')
# train_loader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=False)
# Number of classes, items, and people
nclass = cfg['nclass']
nitems = cfg['nitems']
N = cfg['N']

# Generate class probabilities
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

# create pytorch dataset
dataset = MemoryDataset(data)
train_loader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=False)
test_loader = DataLoader(dataset, batch_size=cfg['N'], shuffle=False)

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
                temperature_decay=cfg['temperature_decay'])
elif cfg['model'] == 'rbm':
    model = RestrictedBoltzmannMachine(
        dataloader=train_loader,
        n_visible=cfg['nitems'],
        n_hidden=cfg['nclass'],
        learning_rate=cfg['learning_date'],
        n_gibbs=cfg['gibbs_samples']
    )
else:
    raise ValueError('Invalid model type')

trainer.fit(model)

test_data = next(iter(test_loader))
if cfg['model'] == 'gs':
    log_pi = model.encoder(test_data)
    pred_class = model.GumbelSoftmax(log_pi).detach().numpy()
    pred_class_ix = np.argmax(pred_class, 1)

    # get the conditional probabilities
    est_probs = model.decoder(torch.eye(cfg['nclass'])).detach().numpy()

elif cfg['model'] == 'vq':
    ze = model.encoder(test_data)
    _, pred_class_ix, _ = model.VQSampler(ze)
    pred_class_ix = pred_class_ix.detach().numpy()

    embs = model.VQSampler.embeddings.weight
    est_probs = model.decoder(embs).detach().numpy()
elif cfg['model'] == 'rbm':
    _, pred_class = model.sample_h(test_data).detach().numpy()
    pred_class_ix = np.argmax(pred_class, 1)

    est_probs =

# plot training loss
logs = pd.read_csv(f'logs/{cfg["which_data"]}/version_0/metrics.csv')
plt.plot(logs['epoch'], logs['train_loss'])
plt.title('Training loss')
plt.savefig(f'./figures/{cfg["which_data"]}/training_loss.png')
# true_probs = pd.read_csv('data/true_probs.csv', index_col=0).to_numpy()

# match estimated latent classes to the correct true class

_, new_order = linear_sum_assignment(-Cor(est_probs, cond_probs))
cond_probs = cond_probs[new_order, :]
true_class = true_class[:, new_order]
true_class_ix = np.argmax(true_class, 1)

# flatten probs for plotting
true_probs = cond_probs.flatten()
est_probs = est_probs.flatten()

plt.figure()
mse = MSE(est_probs, true_probs)
plt.scatter(y=est_probs, x=true_probs)
plt.plot(true_probs, true_probs)
plt.title(f'Probability estimation plot:, MSE={round(mse, 4)}')
plt.xlabel('True values')
plt.ylabel('Estimates')
plt.savefig(f'./figures/{cfg["which_data"]}/probabilities.png')

# print latent class accuracy
print(f'class accuracy: {np.mean(pred_class_ix == true_class_ix)}')



