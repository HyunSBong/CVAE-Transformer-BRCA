import torch
import torch.nn as nn

def idx2onehot(idx, n):
 
    assert torch.max(idx).item() < n
    if idx.dim() == 1:
        idx = idx.unsqueeze(1)
    idx = idx.cuda()
    onehot = torch.zeros(idx.size(0), n).cuda()
    onehot = onehot.scatter(1, idx, 1).cuda()
    """
    tensor([[0.0000, 0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000, 0.000]])
    """

    return onehot
class CVAE(nn.Module):

    def __init__(self, data_dim, compress_dims, latent_size, decompress_dims, conditional=True, num_labels=0, view_size=1000, multivariate=False):

        super().__init__()

        if conditional:
            assert num_labels > 0

        assert type(compress_dims) == list
        assert type(latent_size) == int
        assert type(decompress_dims) == list
        
        self.compress_dims = compress_dims
        self.latent_size = latent_size
        self.decompress_dims = decompress_dims
        self.num_labels = num_labels
        self.view_size = view_size
        self.multivariate = multivariate
        
        self.encoder = Encoder(
            data_dim, compress_dims, latent_size, conditional, num_labels)
        self.decoder = Decoder(
            decompress_dims, latent_size, data_dim, conditional, num_labels, multivariate)
        
    def reparameterize(self, means, logvar, batch_size):
        """Make latent variable z"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn([batch_size, self.latent_size])
        if eps.is_cuda != True:
            eps = eps.cuda()
        z = means + std * eps
        
        return z

    def forward(self, x, c=None):
        if x.dim() > 2:
            x = x.view(-1, self.view_size)

        batch_size = x.size(0)
        
        if x.is_cuda != True:
            x = x.cuda()
        if c.is_cuda != True:
            c = c.cuda()

        means, logvar = self.encoder(x, c)
        
        """Make latent variable z"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        if eps.is_cuda != True:
            eps = eps.cuda()
        z = means + std * eps # latent vector
        
        if self.multivariate:
            # gaussian decoder
            z_means, z_sigmas = self.decoder(z, c)
            return means, logvar, z_means, z_sigmas
        else:
            # bernoulli decoder
            recon_x = self.decoder(z, c)
            return recon_x, means, logvar, z
    

    def inference(self, n=0, c=None):
        """
        Inference gene expression
        Args:
            n (int):
                num of set(tissue).
            c (tensor):
                condition tissues.
                torch.Size([batch_size])
        """
        if n == 0:
            n = self.num_labels
        batch_size = n
        
        if self.multivariate:
            # gaussian decoder
            z_mean = torch.zeros([batch_size, self.latent_size])
            z_std = z_mean + 1
            noise = torch.normal(mean=z_mean, std=z_std).cuda()
            z_means, z_sigmas = self.decoder(noise, c)
            recon_x = z_means

            return recon_x
        else:
            z = torch.randn([batch_size, self.latent_size])
            recon_x = self.decoder(z, c)
            return recon_x

    def embedding(self, x, c=None):

        batch_size = x.size(0)

        means, logvar = self.encoder(x, c)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn([1, self.latent_size])
        
        z = eps * std + means

        return z

class Encoder(nn.Module):
    """
    Encoder of the VAE
    Args:
        data_dim (int):
            Dimensions of the data.
        compress_dims (list of ints):
            Size of each hidden layer.
        latent_size (int):
            Size of the output vector.
        conditional (boolean):
            VAE Condition status.
        num_labels (int):
    """

    def __init__(self, data_dim, compress_dims, latent_size, conditional, num_labels):

        super().__init__()

        self.conditional = conditional
        if self.conditional:
            compress_dims[0] += num_labels
            data_dim += num_labels
        self.num_labels = num_labels
        
        seq = []
        seq += [
            nn.Linear(data_dim, compress_dims[0]),
            nn.BatchNorm1d(num_features=compress_dims[0]),
            nn.ReLU()
        ]
        for i, (in_size, out_size) in enumerate(zip(compress_dims[:-1], compress_dims[1:])):
            seq += [
                nn.Linear(in_size, out_size),
                nn.BatchNorm1d(num_features=out_size),
                nn.ReLU()
            ]
        self.MLP = nn.Sequential(*seq)
        self.linear_means = nn.Linear(compress_dims[-1], latent_size)
        self.linear_logvar = nn.Linear(compress_dims[-1], latent_size)

    def forward(self, x, c=None):
        """Encode the passed x,c
        Args:
            x (tensor):
                datas.
                torch.Size([batch_size, data_dim])
            c (tensor):
                condition tissues.
                torch.Size([batch_size])
        Returns:
            dimensions of the latent space mean and logvar.
            means : (Tensor) Mean of the latent Gaussian
            logvars : (Tensor) Standard deviation of the latent Gaussian
        """
        if self.conditional:
            c = idx2onehot(c, n=self.num_labels)
            x = torch.cat((x, c), dim=-1)
        x = self.MLP(x)
        means = self.linear_means(x)
        logvars = self.linear_logvar(x)

        return means,logvars

class Decoder(nn.Module):
    """
    Decoder of the VAE
    Args:
        decompress_dims (list of ints):
            Size of each hidden layer.
        latent_size (int):
            Size of the output vector.
        data_dim (int):
            Dimensions of the data
        conditional (boolean):
            VAE Condition status.
        num_labels (int):
    """

    def __init__(self, decompress_dims, latent_size, data_dim, conditional, num_labels, multivariate=False):

        super().__init__()
        
        self.num_labels = num_labels
        
        self.conditional = conditional
        if self.conditional:
            input_size = latent_size + num_labels
        else:
            input_size = latent_size
        
        self.multivariate = multivariate
        
        data_dim = data_dim
        out_dim = 0
        seq = []
        for i, (in_size, out_size) in enumerate(zip([input_size]+decompress_dims[:-1], decompress_dims)):
            seq += [
                    nn.Linear(in_size, out_size),
                    nn.BatchNorm1d(num_features=out_size),
                    nn.ReLU()
                ]
            out_dim = out_size
        
        if self.multivariate:
            # gaussian decoder
            self.MLP = nn.Sequential(*seq)
            self.linear_z_means = nn.Linear(out_dim, data_dim)
            # identity covariance
            self.linear_z_sigma = nn.Parameter(torch.ones(data_dim))
        else:
            # bernoulli decoder
            seq += [
                nn.Linear(out_dim, data_dim),
                nn.Sigmoid()
            ]
            self.MLP = nn.Sequential(*seq)

    def forward(self, z, c):
        """Decode the passed x,c
        Args:
            z (tensor):
                datas.
                torch.Size([batch_size, data_dim])
            c (tensor):
                condition tissues.
                torch.Size([batch_size])
        """
        if self.conditional:
            if type(c) != torch.Tensor:
                c = torch.from_numpy(c)
            c = c.cuda()
            z = z.cuda()
            c = idx2onehot(c, n=self.num_labels)
            z = torch.cat((z, c), dim=-1)
        
        x = self.MLP(z)
        if self.multivariate:
            # gaussian decoder
            means = self.linear_z_means(x)
            sigmas = self.linear_z_sigma
            return means, sigmas
        else:
            # bernoulli decoder
            return x