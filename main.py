import torch
from torch.utils.data import DataLoader
from model import *
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CSVLogger
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from data import *
from scipy.stats import pearsonr


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

def Cor(A, B):
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(1)[:, None]
    B_mB = B - B.mean(1)[:, None]

    # Sum of squares across rows
    ssA = (A_mA**2).sum(1)
    ssB = (B_mB**2).sum(1)

    # Finally get corr coeff
    return np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None],ssB[None]))

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
                  callbacks=[EarlyStopping(monitor='train_loss', min_delta=cfg['min_delta'], patience=cfg['patience'], mode='min')])

dataset = CSVDataset('data/data.csv')
train_loader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=False)
vae = VAE(dataloader=train_loader,
          nitems=cfg['nitems'],
          learning_rate=cfg['learning_rate'],
          latent_dims=cfg['nclass'],
          hidden_layer_size=50,
          qm=None,
          batch_size=5000,
          beta=1)
trainer.fit(vae)


true_class = torch.squeeze(torch.Tensor(pd.read_csv('./data/true_class.csv', index_col=0).values))
# calculate predicted class labels
data = next(iter(train_loader))
log_p = vae.encoder(data)
cl = torch.argmax(log_p, dim=1)
acc = torch.mean((cl== true_class).float())


print(f'Latent class accuracy: {acc.item():.4f}')

#est_class = F.one_hot(cl, num_classes=2)

data = next(iter(train_loader))
pred = vae(data)

#print(pred)
#print(data)
mse = MSE(pred.detach().numpy(), data.detach().numpy())
print(f'MSE(pred, data): {mse:.4f}')

#print(torch.mean(cl))

#z = vae.GumbelSoftmax(log_p)

# out = vae.decoder(z)
# dist = torch.distributions.binomial.Binomial(probs=out)
# for i in range(10):
#     sample = dist.sample()
#     bce = torch.nn.functional.binary_cross_entropy(out, sample, reduction='none')
#     bce = torch.mean(bce) * 10
#     print(bce)


# get the conditional probabilities for class 1 and class 2
probs = 1-vae.decoder(torch.eye(cfg['nclass'])).detach().numpy()

# plot training loss
logs = pd.read_csv(f'logs/{cfg["which_data"]}/version_0/metrics.csv')
plt.plot(logs['epoch'], logs['train_loss'])
plt.title('Training loss')
plt.savefig(f'./figures/{cfg["which_data"]}/training_loss.png')
true_probs = pd.read_csv('data/true_probs.csv', index_col=0).to_numpy()

# match estimated latent classes to the correct true class
new_order = np.argmax(Cor(probs, true_probs),0)
probs = probs[new_order, :]
print(new_order)
# flatten probs for plotting
true_probs = true_probs.flatten()
est_probs = probs.flatten()


plt.figure()
mse = MSE(est_probs, true_probs)
plt.scatter(y=est_probs, x=true_probs)
plt.plot(true_probs, true_probs)
plt.title(f'Probability estimation plot:, MSE={round(mse,4)}')
plt.xlabel('True values')
plt.ylabel('Estimates')
plt.savefig(f'./figures/{cfg["which_data"]}/probabilities.png')

