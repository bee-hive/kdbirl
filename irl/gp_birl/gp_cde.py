import torch
import torch.nn as nn
from abc import ABCMeta, abstractmethod
import torch
import torch.distributions as dists
import math
import warnings
import numpy as np

from irl.gp_birl.cvae_mnist.utils import idx2onehot


class VAE(nn.Module):

	def __init__(self, encoder_layer_sizes, latent_size, decoder_layer_sizes, num_labels=0):
		super().__init__()

		assert num_labels > 0

		assert type(encoder_layer_sizes) == list
		assert type(latent_size) == int
		assert type(decoder_layer_sizes) == list

		self.latent_size = latent_size

		self.encoder = Encoder(
			encoder_layer_sizes, latent_size, True, num_labels)
		self.decoder = Decoder(
			decoder_layer_sizes, latent_size, True, num_labels)

	def forward(self, x, c=None):
		if x.dim() > 2:
			x = x.view(-1, 28 * 28)

		means, log_var = self.encoder(x, c)
		z = self.reparameterize(means, log_var)
		recon_x = self.decoder(z, c)

		return recon_x, means, log_var, z

	def reparameterize(self, mu, log_var):
		std = torch.exp(0.5 * log_var)
		eps = torch.randn_like(std)

		return mu + eps * std

	def inference(self, z, c=None):
		recon_x = self.decoder(z, c)

		return recon_x


class Encoder(nn.Module):

	def __init__(self, layer_sizes, latent_size, conditional, num_labels):
		super().__init__()

		self.conditional = conditional
		layer_sizes[0] += num_labels

		self.MLP = nn.Sequential()

		for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
			self.MLP.add_module(
				name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
			self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())

		self.linear_means = nn.Linear(layer_sizes[-1], latent_size)
		self.linear_log_var = nn.Linear(layer_sizes[-1], latent_size)

	def forward(self, x, c=None):
		c = idx2onehot(c, n=10)  # c must be the classes, so outputs. We want this to be continuous
		x = torch.cat((x, c), dim=-1)

		x = self.MLP(x)

		means = self.linear_means(x)
		log_vars = self.linear_log_var(x)

		# Outputs are the parameters of a likelihood. This should be the latent variable weights?
		return means, log_vars


# This should be converted to a GP.
class Decoder(nn.Module):

	def __init__(self, layer_sizes, latent_size, conditional, num_labels):

		super().__init__()

		self.MLP = nn.Sequential()

		self.conditional = conditional
		if self.conditional:
			input_size = latent_size + num_labels
		else:
			input_size = latent_size

		for i, (in_size, out_size) in enumerate(zip([input_size] + layer_sizes[:-1], layer_sizes)):
			self.MLP.add_module(
				name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
			if i + 1 < len(layer_sizes):
				self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
			else:
				self.MLP.add_module(name="sigmoid", module=nn.Sigmoid())

	def forward(self, z, c):

		if self.conditional:
			c = idx2onehot(c, n=10)
			z = torch.cat((z, c), dim=-1)

		x = self.MLP(z)

		return x

class GPCDE(object):
	'''
	Goal is to compute p(y|X) where y is s,a and X is the reward (parameterized as a vector of length number of states)
	'''
	def __init__(self, layer_sizes, latent_size):
		self.encoder = Encoder_GPCDE(layer_sizes, latent_size, 7)
		self.decoder_GPCDE = DecoderGPCDE(layer_sizes, latent_size, 7, num_labels=7)

	def likelihood(self, X, y):
		w = self.encoder.forward(X, y)
		A = self.learn_a(X)
		means, vars = self.decoder(X, A, w) # These are the parameters of f.
		f = dists.Normal(means, vars)

		P = self.learn_p(y.shape[0])
		Diff = np.matmul(y, P) # Linear transform on Y?

		return f.log_prob(Diff)

	def decoder(self, X, A, w):
		z = np.matmul(X, A)
		means, vars = self.decoder_GPCDE.forward(z, w)
		return means, vars

	def learn_a(self, X):
		A = np.random.normal(size=(X.shape[0], 4))
		return A
		pass

	def learn_p(self, dim):
		P = np.random.normal(size=(dim, dim))
		return P


class Encoder_GPCDE(nn.Module):

	def __init__(self, layer_sizes, latent_size, num_labels):
		super().__init__()
		layer_sizes[0] += num_labels

		self.MLP = nn.Sequential()

		# Constructing the MLP (it's just a neural network
		for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
			self.MLP.add_module(
				name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
			self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())

		self.linear_w = nn.Linear(layer_sizes[-1], latent_size)

	def forward(self, x, y):
		x = torch.cat((x, y), dim=-1)
		x = self.MLP(x)
		# I think we just use the means as the weights w. It's just a linear layer after the MLP
		w = self.linear_w(x)

		return w


# This should be converted to a GP.
class DecoderGPCDE(nn.Module):

	def __init__(self, layer_sizes, latent_size, num_labels):

		super().__init__()

		self.MLP = nn.Sequential()

		if self.conditional:
			input_size = latent_size + num_labels
		else:
			input_size = latent_size

		for i, (in_size, out_size) in enumerate(zip([input_size] + layer_sizes[:-1], layer_sizes)):
			self.MLP.add_module(
				name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
			if i + 1 < len(layer_sizes):
				self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
			else:
				self.MLP.add_module(name="sigmoid", module=nn.Sigmoid())
		self.linear_means = nn.Linear(layer_sizes[-1], latent_size)
		self.linear_log_var = nn.Linear(layer_sizes[-1], latent_size)
	def forward(self, z, c):

		c = idx2onehot(c, n=10)
		z = torch.cat((z, c), dim=-1)

		x = self.MLP(z)

		# The GP is defined by the mean and variance. Just use the same strategy as the encoder- means and variance are linear functions..?
		means = self.linear_means(x)
		vars = self.linear_log_var(x)
		return means, vars


# Implemented by Wittawat: https://github.com/wittawatj/kernel-cgof/blob/master/kcgof/cdensity.py
class UnnormalizedCondDensity(object):
	"""
	An abstract class of an unnormalized conditional probability density
	function. This is intended to be used to represent a conditional model of
	the data    for goodness-of-fit testing. Specifically, the class specifies
	p(y|x) where the normalizer may not be known. That is, in fact, it
	specifies p(y,x) since p(y|x) = p(y,x)/p(x), and p(x) is the normalizer.
	The normalizer of the joint density is not assumed to be known.
	The KCSD and FSCD Stein-based tests only require grad_log(..). Subclasses
	can implement either log_den(..) or grad_log(..). If log_den(..) is
	implemented, grad_log(...) will be implemented automatically with
	torch.autograd functions.
	"""

	@abstractmethod
	def log_den(self, X, Y):
		"""
		Evaluate log of the unnormalized density on the n points in (X, Y)
		i.e., log p(y_i, x_i) (up to the normalizer) for i = 1,..., n.
		Y, X are paired.
		X: n x dx Torch tensor
		Y: n x dy Torch tensor
		Return a one-dimensional Torch array of length n.
		The returned array A should be such that A[i] depends on only X[i] and Y[i].
		"""
		expect_dx = self.dx()
		expect_dy = self.dy()
		if X.shape[1] != expect_dx:
			raise ValueError('X must have dimension dx={}. Found {}.'.format(expect_dx, X.shape[1]))
		if Y.shape[1] != expect_dy:
			raise ValueError('Y must have dimension dy={}. Found {}.'.format(expect_dy, Y.shape[1]))

	def log_normalized_den(self, X, Y):
		"""
		Evaluate the exact normalized log density. The difference to log_den()
		is that this method adds the normalizer. This method is not
		compulsory. Subclasses do not need to override.
		"""
		raise NotImplementedError()

	def get_condsource(self):
		"""
		Return a CondSource that allows sampling from this density.
		May return None if no CondSource is implemented.
		Implementation of this method is not enforced in the subclasses.
		"""
		return None

	def grad_log(self, X, Y):
		"""
		Evaluate the gradients (with respect to Y) of the conditional log density at
		each of the n points in X. That is, compute

		grad_yi log p(y_i | x_i)
		and stack all the results for i =1,..., n.
		Assume X.shape[0] = Y.shape[0] = n.

		Given an implementation of log_den(), this method will automatically work.
		Subclasses may override this if a more efficient implementation is
		available.
		X: n x dx Torch tensor
		Y: n x dy Torch tensor
		Return an n x dy Torch array of gradients.
		"""
		if X.shape[0] != Y.shape[0]:
			raise ValueError('X and Y must have the same number of rows. Found: X.shape[0] = {} and Y.shape[0] = {}'.format(X.shape[0], Y.shape[0]))

		# Default implementation with torch.autograd
		Y.requires_grad = True
		logprob = self.log_den(X, Y)
		# sum
		slogprob = torch.sum(logprob)
		Gs = torch.autograd.grad(slogprob, Y, retain_graph=True, only_inputs=True)
		G = Gs[0]

		n, dy = Y.shape
		assert G.shape[0] == n
		assert G.shape[1] == dy
		return G

	@abstractmethod
	def dy(self):
		"""
		Return the dimension of Y.
		"""
		raise NotImplementedError()

	@abstractmethod
	def dx(self):
		"""
		Return the dimension of X.
		"""
		raise NotImplementedError()


class CDGaussianOLS(UnnormalizedCondDensity):
	"""
    Implement p(y|x) = Normal(y - c - slope*x, variance)
    which is the ordinary least-squares model with Gaussian noise N(0, variance).
    * Y is real valued.
    * X is dx dimensional.
    """

	# What we want is p(y|x) = Normal(s, a| Pf([Ar, w]), sigma^2 I)
	# s, a is encapsulated in y

	def __init__(self, slope, c, variance):
		"""
        slope: the slope vector used in slope*x+c as the linear model.
            One dimensional Torch tensor. Length of the slope vector
            determines the matching dimension of x.
        c: a bias (real value)
        variance: the variance of the noise
        """
		self.slope = slope.reshape(-1)
		self.c = c
		if variance <= 0:
			raise ValueError('variance must be positive. Was {}'.format(variance))
		self.variance = variance

	def log_den(self, X, Y):
		"""
        log p(y_i, x_i) (the normalizer is optional) for i = 1,..., n.
        Y, X are paired.
        X: n x dx Torch tensor
        Y: n x dy Torch tensor
        Return a one-dimensional Torch array of length n.
        """
		super().log_den(X, Y)
		return self.log_normalized_den(X, Y)

	def log_normalized_den(self, X, Y):
		super().log_den(X, Y)
		dx = self.dx()
		# https://pytorch.org/docs/stable/distributions.html#normal
		gauss = dists.Normal(0, self.variance ** 0.5)
		S = self.slope.reshape(dx, 1)
		Diff = Y - self.c - X.matmul(S)
		return gauss.log_prob(Diff)

	def get_condsource(self):
		"""
        Return a CondSource that allows sampling from this density.
        """
		cs = CSGaussianOLS(self.slope, self.c, self.variance)
		return cs

	def dy(self):
		"""
        Return the dimension of Y.
        """
		return 1

	def dx(self):
		"""
        Return the dimension of X.
        """
		return self.slope.shape[0]


class CondData(object):
	"""
	Class representing paired data {(y_i, x_i)}_{i=1}^n for conditional
	goodness-of-fit testing. The data are such that y_i is generated from a
	conditional distribution p(y|x_i).
	properties:
	X, Y: Pytorch tensor. X and Y are paired of the same sample size
		(.shape[0]). The dimensions (.shape[1]) are not necessarily the same.
	"""

	def __init__(self, X, Y):
		"""
		:param X: n x dx Pytorch tensor for dataset X
		:param Y: n x dy Pytorch tensor for dataset Y
		"""
		self.X = X.detach()
		self.Y = Y.detach()
		self.X.requires_grad = False
		self.Y.requires_grad = False

		nx, dx = self.X.shape
		ny, dy = self.Y.shape
		if nx != ny:
			raise ValueError('Sample size of the paired sample must be the same.')

	# if not np.all(np.isfinite(X)):
	#     print 'X:'
	#     print util.fullprint(X)
	#     raise ValueError('Not all elements in X are finite.')

	# if not np.all(np.isfinite(Y)):
	#     print 'Y:'
	#     print util.fullprint(Y)
	#     raise ValueError('Not all elements in Y are finite.')

	def dx(self):
		"""Return the dimension of X."""
		dx = self.X.shape[1]
		return dx

	def dy(self):
		"""Return the dimension of Y."""
		dy = self.Y.shape[1]
		return dy

	def n(self):
		return self.X.shape[0]

	def xy(self):
		"""Return (X, Y) as a tuple"""
		return (self.X, self.Y)

	def split_tr_te(self, tr_proportion=0.5, seed=820):
		"""Split the dataset into training and test sets. Assume n is the same
		for both X, Y.

		Return (CondData for tr, CondData for te)"""
		# torch tensors
		X = self.X
		Y = self.Y
		nx, dx = X.shape
		ny, dy = Y.shape
		if nx != ny:
			raise ValueError('Require nx = ny')
		Itr, Ite = util.tr_te_indices(nx, tr_proportion, seed)
		tr_data = CondData(X[Itr].detach(), Y[Itr].detach())
		te_data = CondData(X[Ite].detach(), Y[Ite].detach())
		return (tr_data, te_data)


# def subsample(self, n, seed=87):
#     """Subsample without replacement. Return a new PairedData """
#     if n > self.X.shape[0] or n > self.Y.shape[0]:
#         raise ValueError('n should not be larger than sizes of X, Y.')
#     ind_x = util.subsample_ind( self.X.shape[0], n, seed )
#     ind_y = util.subsample_ind( self.Y.shape[0], n, seed )
#     return PairedData(self.X[ind_x, :], self.Y[ind_y, :], self.label)


# end PairedData class

class CondSource(object):
	"""
	Class representing a data source that allows one to generate data from a
	conditional distribution. This class basically implements a forward
	sampler of a conditional distribution p(y|x).
	No p(x) is implemented.
	Work with Pytorch tensors.
	"""
	__metaclass__ = ABCMeta

	@abstractmethod
	def cond_pair_sample(self, X, seed):
		"""
		Return a Torch tensor Y such that Y.shape[0] = X.shape[0], and
		Y[i, :] ~ p(y | X[i, :]).
		The result should be deterministic given the seed value.
		"""
		raise NotImplementedError()

	def __call__(self, X, seed, *args, **kwargs):
		return self.cond_pair_sample(X, seed, *args, **kwargs)

	@abstractmethod
	def dx(self):
		"""Return the dimension of X"""
		raise NotImplementedError()

	@abstractmethod
	def dy(self):
		"""Return the dimension of Y"""
		raise NotImplementedError()


# end class CondSource


# end CSAdditiveNoiseRegression

class CSGaussianOLS(CondSource):
	"""
	A CondSource for sampling cdensity.CDGaussianOLS.
	p(y|x) = Normal(y - c - slope*x, variance)
	"""

	def __init__(self, slope, c, variance):
		"""
		slope: the slope vector used in slope*x+c as the linear model.
			One dimensional Torch tensor. Length of the slope vector
			determines the matching dimension of x.
		c: a bias (real value)
		variance: the variance of the noise
		"""
		self.slope = slope.reshape(-1)
		self.c = c
		if variance <= 0:
			raise ValueError('variance must be positive. Was {}'.format(variance))
		self.variance = variance

	def cond_pair_sample(self, X, seed):
		if X.shape[1] != self.slope.shape[0]:
			raise ValueError(
				'The dimension of X must be the same as the dimension of slope. Slope dim: {}, X dim: {}'.format(self.slope.shape[0], X.shape[1]))
		n = X.shape[0]
		std = self.variance ** 0.5
		Mean = X.matmul(self.slope.reshape(self.dx(), 1)) + self.c
		with TorchSeedContext(seed=seed):
			sam = torch.randn(n, 1) * std + Mean
		return sam

	def dx(self):
		return self.slope.shape[0]

	def dy(self):
		return 1


class TorchSeedContext(object):
	"""
    A context manager to reset the random seed used by torch.randXXX(...)
    Set the seed back at the end of the block.
    """

	def __init__(self, seed):
		self.seed = seed

	def __enter__(self):
		rstate = torch.get_rng_state()
		self.cur_state = rstate
		torch.manual_seed(self.seed)
		return self

	def __exit__(self, *args):
		torch.set_rng_state(self.cur_state)
