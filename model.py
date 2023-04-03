
import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.utils.prune
import pytorch_lightning as pl


class GumbelSoftmax(pl.LightningModule):
    def __init__(self):
        super(GumbelSoftmax, self).__init__()
        # Gumbel distribution
        self.G = torch.distributions.Gumbel(0, 1)
        self.softmax = torch.nn.Softmax(dim=1)
        self.temperature = 10
        self.temperature_decay = .996

    def forward(self, log_pi):
        # sample gumbel variable and move to correct device
        g = self.G.sample(log_pi.shape)
        g = g.to(log_pi)
        # sample from gumbel softmax
        y = self.softmax((log_pi + g)/self.temperature)
        return y


class Encoder(pl.LightningModule):
    """
    Neural network used as encoder
    """
    def __init__(self,
                 nitems: int,
                 nclass: int,
                 hidden_layer_size: int):
        """
        Initialisation
        :param latent_dims: number of latent dimensions of the model
        """
        super(Encoder, self).__init__()

        self.dense1 = nn.Linear(nitems, hidden_layer_size)
        self.dense2 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.dense3 = nn.Linear(hidden_layer_size, nclass)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        A forward pass though the encoder network
        :param x: a tensor representing a batch of response data
        :param m: a mask representing which data is missing
        :return: a sample from the latent dimensions
        """

        # calculate s and mu based on encoder weights
        out = F.elu(self.dense1(x))
        out = F.elu(self.dense2(out))
        pi = self.dense3(out)
        return pi


class Decoder(pl.LightningModule):
    """
    Neural network used as decoder
    """

    def __init__(self, nitems: int, latent_dims: int, qm: torch.Tensor=None):
        """
        Initialisation
        :param latent_dims: the number of latent factors
        :param qm: IxD Q-matrix specifying which items i<I load on which dimensions d<D
        """
        super().__init__()

        # initialise netowrk components
        self.linear = nn.Linear(latent_dims, 5, bias=True)
        self.linear2 = nn.Linear(5, 10, bias=True)
        self.linear3 = nn.Linear(10, nitems, bias=True)
        self.activation = nn.Sigmoid()

        # remove edges between latent dimensions and items that have a zero in the Q-matrix
        if qm is not None:
            msk_wts = torch.ones((nitems, latent_dims), dtype=torch.float32)
            for row in range(qm.shape[0]):
                for col in range(qm.shape[1]):
                    if qm[row, col] == 0:
                        msk_wts[row][col] = 0
            torch.nn.utils.prune.custom_from_mask(self.linear, name='weight', mask=msk_wts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass though the network
        :param x: tensor representing a sample from the latent dimensions
        :param m: mask representing which data is missing
        :return: tensor representing reconstructed item responses
        """
        out = F.relu(self.linear(x))
        out = F.relu(self.linear2(out))
        out = self.linear3(out)
        out = self.activation(out)
        return out


class VAE(pl.LightningModule):
    """
    Neural network for the entire variational autoencoder
    """
    def __init__(self,
                 dataloader,
                 nitems: int,
                 latent_dims: int,
                 hidden_layer_size: int,
                 qm: torch.Tensor,
                 learning_rate: float,
                 batch_size: int,
                 beta: int = 1):
        """
        Initialisaiton
        :param latent_dims: number of latent dimensions
        :param qm: IxD Q-matrix specifying which items i<I load on which dimensions d<D
        """
        super(VAE, self).__init__()
        #self.automatic_optimization = False
        self.nitems = nitems
        self.dataloader = dataloader

        self.encoder = Encoder(nitems,
                               latent_dims,
                               hidden_layer_size
        )

        self.GumbelSoftmax = GumbelSoftmax()

        self.Softmax = nn.Softmax(dim=1)

        self.decoder = Decoder(nitems, latent_dims, qm)

        self.lr = learning_rate
        self.batch_size = batch_size
        self.beta = beta
        self.kl=0

    def forward(self, x: torch.Tensor, m: torch.Tensor=None):
        """
        forward pass though the entire network
        :param x: tensor representing response data
        :param m: mask representing which data is missing
        :return: tensor representing a reconstruction of the input response data
        """
        log_pi = self.encoder(x)
        z = self.GumbelSoftmax(log_pi)
        reco = self.decoder(z)

        # Calcualte the estimated probabilities
        pi = self.Softmax(log_pi)
        # calculate kl divergence
        #self.kl = -torch.sum(pi * torch.log(pi))
        log_ratio = torch.log(pi * 2 + 1e-20)
        self.kl = torch.sum(pi * log_ratio, dim=-1).mean()

        return reco

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, amsgrad=True)

    def training_step(self, batch, batch_idx):
        # forward pass

        data = batch
        X_hat = self(data)

        bce = torch.nn.functional.binary_cross_entropy(X_hat, batch, reduction='none')
        bce = torch.mean(bce) * self.nitems

        loss = bce + self.beta * torch.sum(self.kl)
        self.GumbelSoftmax.temperature *= self.GumbelSoftmax.temperature_decay
        self.log('train_loss',loss)

        return {'loss': loss}

    def train_dataloader(self):
        return self.dataloader