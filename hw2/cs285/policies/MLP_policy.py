import abc
import itertools
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from cs285.infrastructure import pytorch_util as ptu
from cs285.policies.base_policy import BasePolicy


class MLPPolicy(BasePolicy, nn.Module, metaclass=abc.ABCMeta):
    def __init__(
        self,
        ac_dim,
        ob_dim,
        n_layers,
        size,
        discrete=False,
        learning_rate=1e-4,
        training=True,
        nn_baseline=False,
        **kwargs
    ):
        super().__init__(**kwargs)

        # init vars
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.discrete = discrete
        self.size = size
        self.learning_rate = learning_rate
        self.training = training
        self.nn_baseline = nn_baseline

        if self.discrete:
            self.logits_na = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=self.ac_dim,
                n_layers=self.n_layers,
                size=self.size,
            )
            self.logits_na.to(ptu.device)
            self.mean_net = None
            self.logstd = None
            self.optimizer = optim.Adam(self.logits_na.parameters(), self.learning_rate)
            self.loss = nn.CrossEntropyLoss()
        else:
            self.logits_na = None
            self.mean_net = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=self.ac_dim,
                n_layers=self.n_layers,
                size=self.size,
            )
            self.mean_net.to(ptu.device)
            self.logstd = nn.Parameter(
                torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
            )
            self.logstd.to(ptu.device)
            self.optimizer = optim.Adam(
                itertools.chain([self.logstd], self.mean_net.parameters()),
                self.learning_rate,
            )
            self.loss = nn.MSELoss()

    ##################################

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    def serialize(self):
        """
        Returns a dict with everything needed to reconstruct this policy.
        Used for CPU multiprocessing.
        """
        state_dict = self.state_dict()
        state_dict_cpu = {k: v.cpu() for k, v in state_dict.items()}
        return {
            "__class__": self.__class__,
            "ac_dim": self.ac_dim,
            "ob_dim": self.ob_dim,
            "n_layers": self.n_layers,
            "size": self.size,
            "discrete": self.discrete,
            "learning_rate": self.learning_rate,
            "training": self.training,
            "nn_baseline": self.nn_baseline,
            "__state_dict__": state_dict_cpu,
        }

    @classmethod
    def deserialize(cls, state):
        ret = cls(
            state["ac_dim"],
            state["ob_dim"],
            state["n_layers"],
            state["size"],
            discrete=state["discrete"],
            learning_rate=state["learning_rate"],
            training=state["training"],
            nn_baseline=state["nn_baseline"],
        )
        ret.load_state_dict(state["__state_dict__"])
        return ret

    ##################################

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]
        observation = ptu.from_numpy(observation.astype(np.float32))
        action_dist = self(observation)
        action = action_dist.sample()
        return ptu.to_numpy(action)

    # update/train this policy
    def update(self, observations, actions, **kwargs):
        obs = ptu.from_numpy(observations.astype(np.float32))
        truth_acs = ptu.from_numpy(actions.astype(np.float32))
        policy_acs = self(obs)
        loss = self.loss(policy_acs, truth_acs)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor):
        if self.discrete:
            logits = self.logits_na(observation)
            dist = distributions.Categorical(logits=logits)
            return dist

        mean = self.mean_net(observation)
        dist = distributions.Normal(mean, torch.exp(self.logstd))
        return dist


#####################################################
#####################################################


class MLPPolicyPG(MLPPolicy):
    def __init__(self, ac_dim, ob_dim, n_layers, size, **kwargs):

        super().__init__(ac_dim, ob_dim, n_layers, size, **kwargs)
        self.baseline_loss = nn.MSELoss()

    def update(self, observations, actions, advantages, q_values=None):
        observations = ptu.from_numpy(observations)
        actions = ptu.from_numpy(actions)
        advantages = ptu.from_numpy(advantages)

        # compute the loss that should be optimized when training with policy gradient
        # HINT1: Recall that the expression that we want to MAXIMIZE
        # is the expectation over collected trajectories of:
        # sum_{t=0}^{T-1} [grad [log pi(a_t|s_t) * (Q_t - b_t)]]
        # HINT2: you will want to use the `log_prob` method on the distribution returned
        # by the `forward` method
        # HINT3: don't forget that `optimizer.step()` MINIMIZES a loss

        action_dists = self(observations)

        log_probs = action_dists.log_prob(actions)
        loss = -torch.sum(log_probs * advantages)

        # optimize `loss` using `self.optimizer`
        # HINT: remember to `zero_grad` first
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.nn_baseline:
            ## TODO: normalize the q_values to have a mean of zero and a standard deviation of one
            ## HINT: there is a `normalize` function in `infrastructure.utils`
            targets = TODO
            targets = ptu.from_numpy(targets)

            ## TODO: use the `forward` method of `self.baseline` to get baseline predictions
            baseline_predictions = TODO

            ## avoid any subtle broadcasting bugs that can arise when dealing with arrays of shape
            ## [ N ] versus shape [ N x 1 ]
            ## HINT: you can use `squeeze` on torch tensors to remove dimensions of size 1
            assert baseline_predictions.shape == targets.shape

            # TODO: compute the loss that should be optimized for training the baseline MLP (`self.baseline`)
            # HINT: use `F.mse_loss`
            baseline_loss = TODO

            # TODO: optimize `baseline_loss` using `self.baseline_optimizer`
            # HINT: remember to `zero_grad` first
            TODO

        train_log = {
            "Training Loss": ptu.to_numpy(loss),
        }
        return train_log

    def run_baseline_prediction(self, obs):
        """
        Helper function that converts `obs` to a tensor,
        calls the forward method of the baseline MLP,
        and returns a np array

        Input: `obs`: np.ndarray of size [N, 1]
        Output: np.ndarray of size [N]

        """
        obs = ptu.from_numpy(obs)
        predictions = self.baseline(obs)
        return ptu.to_numpy(predictions)[:, 0]
