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
import sys
import time

# read in configurations
with open("./config.yml", "r") as f:
    cfg = yaml.safe_load(f)
    cfg = cfg['configs']

# potentially overwrite configurations with command line arguments
if len(sys.argv) > 1:
    cfg['model'] = sys.argv[1]
    cfg['nclass'] = int(sys.argv[2])
    cfg['N'] = int(sys.argv[3])
    iteration = int(sys.argv[4])
    cfg['nitems'] = int(sys.argv[5])
    cfg['n_iw_samples'] = int(sys.argv[6])
    cfg['emb_dim'] = int(sys.argv[7])
    cfg['n_rep'] = int(sys.argv[8])
else:
    iteration = 1

# either simulate or read in data depending on configurations
if cfg['which_data'] == 'sim':
    nclass = cfg['nclass']
    nitems = cfg['nitems']
    N = cfg['N']

    # Generate class probabilities
    np.random.seed(iteration)
    class_probs = np.ones(nclass) / nclass

    # Generate conditional probabilities
   # cond_probs = np.random.uniform(.4, .5, size=(nclass, nitems))
    cond_probs = np.expand_dims(np.random.uniform(.3, .7, nitems), 0).repeat(nclass, 0)
    # Iterate over each row
    for i in range(nclass):
        # Randomly choose the number of entries to set to one between 50 and 100
        num_ones = int(nitems * 0.9) #np.random.randint(50, 80)

        # Randomly select num_ones indices to set to one
        indices = np.random.choice(nitems, num_ones, replace=False)

        # Set the selected indices to one
        cond_probs[i, indices] = np.random.uniform(.1,.9, num_ones)#+= .20#

    # Generate true class membership for each person
    true_class_ix = np.random.choice(np.arange(nclass), size=(N,), p=class_probs)
    true_class = np.zeros((N, nclass))
    true_class[np.arange(N), true_class_ix] = 1

    # simulate responses
    prob = true_class @ cond_probs
    data = np.random.binomial(1, prob).astype(float)


    if cfg['save_data']:
        np.savetxt(f'true/data/data_{cfg["nclass"]}_{cfg["N"]}_{iteration}_{cfg["nitems"]}.csv', data,  delimiter=',')
        np.savetxt(f'true/parameters/class_{cfg["nclass"]}_{cfg["N"]}_{iteration}_{cfg["nitems"]}.csv', true_class, delimiter=',')
        np.savetxt(f'true/parameters/probs_{cfg["nclass"]}_{cfg["N"]}_{iteration}_{cfg["nitems"]}.csv', cond_probs, delimiter=',')
        sys.exit()
elif cfg['which_data'] == 'disk':
    data = genfromtxt(f'true/data/data_{cfg["nclass"]}_{cfg["N"]}_{iteration}_{cfg["nitems"]}.csv', delimiter=',')
    cond_probs = genfromtxt(f'true/parameters/probs_{cfg["nclass"]}_{cfg["N"]}_{iteration}_{cfg["nitems"]}.csv', delimiter=',')
    true_class = genfromtxt(f'true/parameters/class_{cfg["nclass"]}_{cfg["N"]}_{iteration}_{cfg["nitems"]}.csv', delimiter=',')

    true_class_ix = np.argmax(true_class, 1)
else:
    raise ValueError('Unknown value for which_data')

np.savetxt('/Users/karel/Downloads/lcadata.csv', data, delimiter=',')
np.savetxt('/Users/karel/Downloads/lcaclass.csv', true_class, delimiter=',')
np.savetxt('/Users/karel/Downloads/lcaprobs.csv', cond_probs, delimiter=',')
# create pytorch dataset
dataset = MemoryDataset(data)
train_loader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True)
test_loader = DataLoader(dataset, batch_size=cfg['N'], shuffle=False)

# repeat process cfg['n_rep'] times to prevent local minima
best = -float('inf')
for i in range(cfg['n_rep']):
    if cfg['model'] in ['dvae', 'gs', 'vq', 'log']:
        model = VAE(dataloader=train_loader,
                    nitems=cfg['nitems'],
                    nclass=cfg['nclass'],
                    hidden_layer_size=50,#(cfg['nitems']+cfg['nclass'])//2,
                    learning_rate=cfg['learning_rate'],
                    emb_dim=cfg['emb_dim'],
                    beta=1,
                    temperature=cfg['temperature'],
                    temperature_decay=cfg['temperature_decay'],
                    sampler_type=cfg['model'],
                    min_temp=cfg['gumbel_min_temp'],
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
    else:
        raise ValueError('Invalid model type')
    # initialise model and optimizer

    if os.path.exists('logs/all/version_0/metrics.csv'):
        os.remove('logs/all/version_0/metrics.csv')

    logger = CSVLogger("logs", name='all', version=0)
    trainer = Trainer(fast_dev_run=cfg['single_epoch_test_run'],
                      max_epochs=cfg['max_epochs'],
                      #min_epochs=1000,
                      logger=logger,
                      callbacks=[
                      EarlyStopping(monitor='train_loss', min_delta=cfg['min_delta'], patience=cfg['patience'],
                                    mode='min')],
                      enable_progress_bar=True,
                      enable_model_summary=False,
                      detect_anomaly=False,
                      accelerator='cpu')

    start = time.time()
    trainer.fit(model)
    runtime = time.time() - start
    print(f'runtime: {runtime}')

    print(model.sampler.temperature)

    # compute the estimated conditional probabilities and the posterior probabilities on the test data
    test_data = next(iter(test_loader))
    if cfg['model'] in ['gs', 'log']:
        log_pi = model.encoder(test_data)
        #pred_class = model.sampler(log_pi.exp()).detach().numpy()
        #pred_class_ix = np.argmax(pred_class, 1)

        pred_class = model.fscores(torch.Tensor(data)).mean(0).detach().numpy()
        pred_class_ix = np.argmax(pred_class, 1)


        # get the conditional probabilities
        est_probs = model.decoder(torch.eye(cfg['nclass'])).detach().numpy()


    elif cfg['model'] == 'dvae':
        # log
        log_pi = model.encoder(test_data)
        p1 =  model.sampler(log_pi.exp()).detach().numpy()

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
    elif cfg['model'] == 'vq':
        ze = model.encoder(test_data)
        pred_class_ix = model.sampler.closest_emb_ix(ze)
        pred_class_ix = pred_class_ix.detach().numpy()
        pred_class = np.zeros_like(true_class)
        pred_class[np.arange(cfg['N']), pred_class_ix] = 1

        embs = model.sampler.embeddings.weight
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

    # only plot when there are no command line arguments
    if len(sys.argv) == 1:
        # plot training loss
        logs = pd.read_csv(f'logs/all/version_0/metrics.csv')
        plt.plot(logs['epoch'], logs['train_loss'])
        plt.title('Training loss')
        plt.savefig(f'./figures/training_loss.png')
        # true_probs = pd.read_csv('data/true_probs.csv', index_col=0).to_numpy()



    # Compute log likelihood
    log_likelihood = np.sum(data * np.log(pred_class.dot(est_probs) + 1e-10) +
                            (1 - data) * np.log(1 - pred_class.dot(est_probs) + 1e-10))


    # save class predictions and conditional probanilities only for the model with the highest likelihood
    if log_likelihood > best:
        best = log_likelihood
        best_class_ix = pred_class_ix
        best_cond_probs = est_probs
        np.savetxt('./tmp/est_probs.csv', est_probs, delimiter=',')
        np.savetxt('./tmp/pred_class.csv', pred_class, delimiter=',')

# match estimated latent classes to the correct true class
_, new_order = linear_sum_assignment(-Cor(best_cond_probs, cond_probs))
cond_probs = cond_probs[new_order, :]
true_class = true_class[:, new_order]
true_class_ix = np.argmax(true_class, 1)



# compute latent class accuracy
lc_acc = np.mean(best_class_ix == true_class_ix)
# compute MSE of conditional probabilities
mse_cond = MSE(best_cond_probs, cond_probs)


# only plot if there are no command line arguments
if len(sys.argv) == 1:
    # flatten probs for plotting
    true_probs = cond_probs.flatten()
    est_probs = best_cond_probs.flatten()

    plt.figure()
    plt.scatter(y=est_probs, x=true_probs)
    plt.plot(true_probs, true_probs)
    plt.title(f'Probability estimation plot:, MSE={round(mse_cond, 4)}')
    plt.xlabel('True values')
    plt.ylabel('Estimates')
    plt.savefig(f'./figures/probabilities.png')

    # print latent class accuracy
    print(f'class accuracy: {lc_acc}')
    print(f'MSE: {mse_cond}')
else:
    with open(f"results/metrics/{'_'.join(sys.argv[1:])}.txt", 'w') as f:
        f.writelines([f'{lc_acc}\n', f'{mse_cond}\n', f'{runtime}\n'])

    par_names = ['conditional', 'class']
    par = []
    value = []
    par_i = []
    par_j = []
    for i, est in enumerate([est_probs, best_cond_probs]):
        for r in range(est.shape[0]):
            for c in range(est.shape[1]):
                par.append(par_names[i])
                value.append(est[r, c])
                par_i.append(r)
                par_j.append(c)

    result = pd.DataFrame({'model': cfg['model'], 'nclass': cfg['nclass'], 'n_rep': cfg['n_rep'],
                           'iteration':iteration, 'parameter': par, 'i': par_i, 'j': par_j, 'value': value})

    result.to_csv(f'results/estimates/est_{"_".join(sys.argv[1:])}.txt')

