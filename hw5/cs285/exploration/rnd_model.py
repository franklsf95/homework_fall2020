from cs285.infrastructure import pytorch_util as ptu
from .base_exploration_model import BaseExplorationModel
import torch.optim as optim
from torch import nn
import torch


def init_uniform(model):
    model.weight.data.uniform_()
    model.bias.data.uniform_()


def init_normal(model):
    model.weight.data.normal_()
    model.bias.data.normal_()


class RNDModel(nn.Module, BaseExplorationModel):
    def __init__(self, hparams, optimizer_spec, **kwargs):
        super().__init__(**kwargs)
        self.ob_dim = hparams["ob_dim"]
        self.output_size = hparams["rnd_output_size"]
        self.n_layers = hparams["rnd_n_layers"]
        self.size = hparams["rnd_size"]
        self.optimizer_spec = optimizer_spec

        # Create two neural networks:
        # 1) f, the random function we are trying to learn
        # 2) f_hat, the function we are using to learn f
        # WARNING: Make sure you use different types of weight
        #          initializations for these two functions

        # HINT 1) Check out the method ptu.build_mlp
        # HINT 2) There are two weight init methods defined above

        self.f = ptu.build_mlp(
            input_size=self.ob_dim,
            output_size=self.output_size,
            n_layers=self.n_layers,
            size=self.size,
            init_method=init_uniform,
        )
        self.f_hat = ptu.build_mlp(
            input_size=self.ob_dim,
            output_size=self.output_size,
            n_layers=self.n_layers,
            size=self.size,
            init_method=init_normal,
        )

        self.optimizer = self.optimizer_spec.constructor(
            self.f_hat.parameters(), **self.optimizer_spec.optim_kwargs
        )
        self.learning_rate_scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            self.optimizer_spec.learning_rate_schedule,
        )

        self.f.to(ptu.device)
        self.f_hat.to(ptu.device)

    def forward(self, ob_no):
        # Get the prediction error for ob_no
        # HINT: Remember to detach the output of self.f!
        f_out = self.f(ob_no).detach()
        f_hat_out = self.f_hat(ob_no)
        error = nn.functional.mse_loss(f_out, f_hat_out, reduction="none").mean(dim=1)
        return error

    def forward_np(self, ob_no):
        ob_no = ptu.from_numpy(ob_no)
        error = self(ob_no)
        return ptu.to_numpy(error)

    def update(self, ob_no):
        # Update f_hat using ob_no
        # Hint: Take the mean prediction error across the batch
        error = self.forward_np(ob_no)
        loss = error.mean()
        return loss.item()
