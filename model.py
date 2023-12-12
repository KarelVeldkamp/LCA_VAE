
import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.utils.prune
import pytorch_lightning as pl


class GumbelSoftmax(pl.LightningModule):
    """
    Network module that performs the Gumbel-Softmax operation
    """
    def __init__(self, temperature, temperature_decay):
        """
        initialize
        :param temperature: temperature parameter at the start of training
        :param temperature_decay: factor to multiply temperature by each epoch
        """
        super(GumbelSoftmax, self).__init__()
        # Gumbel distribution
        self.G = torch.distributions.Gumbel(0, 1)
        self.softmax = torch.nn.Softmax(dim=1)
        self.temperature = temperature
        self.temperature_decay = temperature_decay

    def forward(self, log_pi):
        """
        forward pass
        :param log_pi: NxM tensor of log probabilities where N is the batch size and M is the number of classes
        :return: NxM tensor of 'discretized probabilities' the lowe the temperature the more discrete
        """
        # sample gumbel variable and move to correct device
        g = self.G.sample(log_pi.shape)
        g = g.to(log_pi)
        # sample from gumbel softmax
        y = self.softmax((log_pi + g)/self.temperature)
        return y


class STEFunction(torch.autograd.Function):
    """
    Class for straight through estimator. Samples from a tensor of proabilities, but defines gradient as if
    forward is an identity()
    """
    @staticmethod
    def forward(ctx, probs):
        """
        Sample from multinomial distribution given a tensor of probabiltieis
        :param ctx: unused argument for passing class
        :param probs: NxM tensor of probabilties
        :return: output: NxM binary tensor containing smapled values
        """
        sample_ix = torch.multinomial(probs, 1).squeeze()
        output = torch.zeros_like(probs)
        output[torch.arange(output.shape[0]), sample_ix] = 1
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        gradients for straight trough estimator
        :param ctx: unused argument for passing class
        :param grad_output: the gradients passed to this function
        :return: simply the gradients clamped to be fbetween -1 and 1
        """
        return F.hardtanh(grad_output)

class StraightThroughSampler(nn.Module):
    """
    Neural network module for straight though sampler. Samples a value in forward but returns gradients as
    if there was no sampling
    """
    def __init__(self):
        super(StraightThroughSampler, self).__init__()

    def forward(self, x):
            x = STEFunction.apply(x)
            return x

class MultinomialSampler(nn.Module):
    """
    Neural network module for straight though sampler. Samples a value in forward but returns gradients as
    if there was no sampling
    """
    def __init__(self):
        super(MultinomialSampler, self).__init__()

    def forward(self, probs):
        sample_ix =  torch.multinomial(probs, 1).squeeze()
        one_hot = torch.nn.functional.one_hot(sample_ix)

        return one_hot.float()


class VQFunction(torch.autograd.Function):
    """
    Class for vector quantizer function. Backwars is identity
    """
    @staticmethod
    def forward(ctx, ze, embeddings):
        """
        Sample from multinomial distribution given a tensor of probabiltieis
        :param ctx: unused argument for passing class
        :param ze: BxE tensor of proposed embedding vectors where B is the batch size and E is the embedding size
        :return: output: BxE tensor of selected embeddings. Closest embedding to each proposal
        """
        # compute the distances between the encoder outputs and the embeddings
        dist_mat = torch.cdist(ze, embeddings.weight.clone(), p=2)
        # select closest embedding for each person

        emb_ix = torch.argmin(dist_mat, dim=1)

        # Return tensor of selected embeddings
        return embeddings(emb_ix)

    @staticmethod
    def backward(ctx, grad_output):
        """
        gradients for straight trough estimator
        :param ctx: unused argument for passing class
        :param grad_output: the gradients passed to this function
        :return: simply the gradients clamped to be fbetween -1 and 1
        """
        return F.hardtanh(grad_output)


class VectorQuantizeSampler(nn.Module):
    """
    Neural network module for straight though sampler. Samples a value in forward but returns gradients as
    if there was no sampling
    """
    def __init__(self):
        super(VectorQuantizeSampler, self).__init__()

    def forward(self, zq, embeddings):
            x = VQFunction.apply(zq, embeddings)
            return x

class VectorQuantizer(nn.Module):
    def __init__(self, n_emb, emb_dim):
        super().__init__()

        self.embeddings = nn.Embedding(n_emb, emb_dim)


    def forward(self, ze):
        # compute the distances between the encoder outputs and the embeddings
        dist_mat = torch.cdist(ze, self.embeddings.weight.clone(), p=2)
        # select closest embedding for each person

        emb_ix = torch.argmin(dist_mat, dim=1)
        zq = self.embeddings(emb_ix)

        vq_loss = F.mse_loss(zq.detach(), ze) + F.mse_loss(zq, ze.detach())

        zq = ze + (zq-ze).detach()
        return zq, emb_ix, vq_loss


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

    def __init__(self, nitems: int, latent_dims: int):
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

class VQDecoder(pl.LightningModule):
    """
    Neural network used as decoder
    """

    def __init__(self, nitems: int, emb_dim: int):
        """
        Initialisation
        :param latent_dims: the number of latent factors
        :param qm: IxD Q-matrix specifying which items i<I load on which dimensions d<D
        """
        super().__init__()

        # initialise netowrk components
        self.linear = nn.Linear(emb_dim, emb_dim*2, bias=True)
        self.linear2 = nn.Linear(emb_dim*2, emb_dim*2, bias=True)
        self.linear3 = nn.Linear(emb_dim*2, nitems, bias=True)
        self.activation = nn.Sigmoid()

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

class GSVAE(pl.LightningModule):
    """
    Neural network for the entire variational autoencoder
    """
    def __init__(self,
                 dataloader,
                 nitems: int,
                 latent_dims: int,
                 hidden_layer_size: int,
                 learning_rate: float,
                 beta: int = 1,
                 temperature: float = 10,
                 temperature_decay: float = .998):
        """
        Initialisaiton
        :param latent_dims: number of latent dimensions
        :param qm: IxD Q-matrix specifying which items i<I load on which dimensions d<D
        """
        super(GSVAE, self).__init__()
        #self.automatic_optimization = False
        self.nitems = nitems
        self.dataloader = dataloader

        self.encoder = Encoder(nitems,
                               latent_dims,
                               hidden_layer_size
        )
        self.mns = MultinomialSampler()
        self.GumbelSoftmax = GumbelSoftmax(temperature, temperature_decay)
        self.ST = StraightThroughSampler()
        self.Softmax = nn.Softmax(dim=1)

        self.decoder = Decoder(nitems, latent_dims)

        self.lr = learning_rate
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

        # commented out raouls idee
        #z = self.Softmax(log_pi)
        #z = self.mns(z)

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

class VQVAE(pl.LightningModule):
    """
    Neural network for the entire variational autoencoder
    """
    def __init__(self,
                 dataloader,
                 nitems: int,
                 latent_dims: int,
                 hidden_layer_size: int,
                 learning_rate: float,
                 emb_dim: int,
                 beta: int = 1,
):
        """
        Initialisaiton
        :param latent_dims: number of latent dimensions
        :param qm: IxD Q-matrix specifying which items i<I load on which dimensions d<D
        """
        super(VQVAE, self).__init__()
        #self.automatic_optimization = False
        self.nitems = nitems
        self.dataloader = dataloader

        self.encoder = Encoder(nitems,
                               emb_dim,
                               hidden_layer_size
        )
        #self.embeddings = nn.Embedding(latent_dims, emb_dim)
        self.VQSampler = VectorQuantizer(latent_dims, emb_dim)
        self.decoder = VQDecoder(nitems, emb_dim)

        self.lr = learning_rate
        self.vq_loss = 0
        self.beta = beta
        self.kl=0

    def forward(self, x: torch.Tensor):
        """
        forward pass though the entire network
        :param x: tensor representing response data
        :return: tensor representing a reconstruction of the input response data
        """
        # compute proposal sample
        ze = self.encoder(x)

        # find nearest embedding
        zq, _, vq_loss = self.VQSampler(ze)
        self.vq_loss = vq_loss

        # reconstruct input from embedding
        reco = self.decoder(zq)

        return reco

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, amsgrad=True)

    def training_step(self, batch, batch_idx):
        # forward pass

        data = batch
        X_hat = self(data)

        reconstruction_loss = torch.nn.functional.binary_cross_entropy(X_hat, batch, reduction='none')
        reconstruction_loss = torch.mean(reconstruction_loss) * self.nitems

        loss = reconstruction_loss + self.vq_loss

        self.log('train_loss',loss)

        return {'loss': loss}

    def train_dataloader(self):
        return self.dataloader


class RestrictedBoltzmannMachine(pl.LightningModule):

    def __init__(self, dataloader, n_visible, n_hidden, learning_rate, n_gibbs):
        super(RestrictedBoltzmannMachine, self).__init__()
        # parameters
        self.W = torch.nn.Parameter(torch.randn(n_visible, n_hidden))
        self.b_hidden = torch.nn.Parameter(torch.zeros(n_hidden))
        self.b_visible = torch.nn.Parameter(torch.zeros(n_visible))

        # hyperparamters
        self.lr = learning_rate
        self.n_gibbs = n_gibbs

        self.dataloader = dataloader
        self.automatic_optimization = False

    def sample_h(self, v):
        ph = torch.sigmoid(torch.matmul(v, self.W) + self.b_hidden)
        h = torch.bernoulli(ph)

        return h, ph

    def sample_v(self, h):
        pv = torch.sigmoid(torch.matmul(h, self.W.t()) + self.b_visible)
        v = torch.bernoulli(pv)

        return v, pv

    def free_energy(self, v):
        vbias_term = v.mv(self.b_visible)
        h = torch.matmul(v, self.W) + self.b_hidden
        hidden_term = h.exp().add(1).log().sum(1)
        return (-hidden_term - vbias_term).mean()

    def training_step(self, v0):
        v = v0.detach().clone()
        for i in range(self.n_gibbs):
            h, ph = self.sample_hidden(v)
            v = self.sample_v(h)

        loss = self.free_energy(v0) - self.free_energy(v)

        return loss


    def train_dataloader(self):
        return self.dataloader