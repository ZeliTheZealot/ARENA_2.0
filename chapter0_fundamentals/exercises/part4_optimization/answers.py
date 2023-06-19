#%%
import os; os.environ["ACCELERATE_DISABLE_RICH"] = "1"
import sys
import pandas as pd
import torch as t
from torch import optim
import torch.nn.functional as F
from torchvision import datasets
from torch.utils.data import DataLoader, Subset
from typing import Callable, Iterable, Tuple, Optional
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from IPython.display import display, HTML

# Make sure exercises are in the path
chapter = r"chapter0_fundamentals"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part4_optimization"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))
os.chdir(section_dir)

from plotly_utils import bar, imshow
from part3_resnets.solutions import IMAGENET_TRANSFORM, get_resnet_for_feature_extraction, plot_train_loss_and_test_accuracy_from_metrics
from part4_optimization.utils import plot_fn, plot_fn_with_points
import part4_optimization.tests as tests

device = t.device("cuda" if t.cuda.is_available() else "cpu")

MAIN = __name__ == "__main__"

# %%
1
# %%
def pathological_curve_loss(x: t.Tensor, y: t.Tensor):
    # Example of a pathological curvature. There are many more possible, feel free to experiment here!
    x_loss = t.tanh(x) ** 2 + 0.01 * t.abs(x)
    y_loss = t.sigmoid(y)
    return x_loss + y_loss


if MAIN:
    plot_fn(pathological_curve_loss)
# %%
def opt_fn_with_sgd(fn: Callable, xy: t.Tensor, lr=0.001, momentum=0.98, n_iters: int = 1000):
    '''
    Optimize the a given function starting from the specified point.

    xy: shape (2,). The (x, y) starting point.
    n_iters: number of steps.
    lr, momentum: parameters passed to the torch.optim.SGD optimizer.

    Return: (n_iters, 2). The (x,y) BEFORE each step. So out[0] is the starting point.
    '''
    optimizer = t.optim.SGD([xy], lr=lr, momentum=momentum)
    result = t.zeros((n_iters, 2))
    for i in range(n_iters):
        result[i] = xy.detach()
        output = fn(*xy)
        output.backward()
        optimizer.step()
        optimizer.zero_grad()
    return result
if MAIN:
    points = []

    optimizer_list = [
        (optim.SGD, {"lr": 0.1, "momentum": 0.0}),
        (optim.SGD, {"lr": 0.02, "momentum": 0.99}),
    ]

    for optimizer_class, params in optimizer_list:
        xy = t.tensor([2.5, 2.5], requires_grad=True)
        xys = opt_fn_with_sgd(pathological_curve_loss, xy=xy, lr=params['lr'], momentum=params['momentum'])

        points.append((xys, optimizer_class, params))

    plot_fn_with_points(pathological_curve_loss, points=points)
# %%
class SGD:
    def __init__(
        self, 
        params: Iterable[t.nn.parameter.Parameter], 
        lr: float, 
        momentum: float = 0.0, 
        weight_decay: float = 0.0
    ):
        '''Implements SGD with momentum.

        Like the PyTorch version, but assume nesterov=False, maximize=False, and dampening=0
            https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD

        '''
        params = list(params) # turn params into a list (because it might be a generator)
        # SOLUTION
        self.params = params
        self.lr = lr
        self.mu = momentum
        self.lmda = weight_decay
        self.t = 0

        self.gs = [t.zeros_like(p) for p in self.params]

    def zero_grad(self) -> None:
        # SOLUTION
        for param in self.params:
            param.grad = None

    @t.inference_mode()
    def step(self) -> None:
        # SOLUTION
        for i, (g, param) in enumerate(zip(self.gs, self.params)):
            # Implement the algorithm from the pseudocode to get new values of params and g
            new_g = param.grad
            if self.lmda != 0:
                new_g = new_g + (self.lmda * param)
            if self.mu != 0 and self.t > 0:
                new_g = (self.mu * g) + new_g
            # Update params (remember, this must be inplace)
            self.params[i] -= self.lr * new_g
            # Update g
            self.gs[i] = new_g
        self.t += 1

    def __repr__(self) -> str:
        return f"SGD(lr={self.lr}, momentum={self.mu}, weight_decay={self.lmda})"
# %%
if MAIN:
    tests.test_sgd(SGD)
# %%
class RMSprop:
    def __init__(
        self,
        params: Iterable[t.nn.parameter.Parameter],
        lr: float = 0.01,
        alpha: float = 0.99,
        eps: float = 1e-08,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
    ):
        '''Implements RMSprop.

        Like the PyTorch version, but assumes centered=False
            https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html

        '''
        params = list(params) # turn params into a list (because it might be a generator)
        self.params = params
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.lmda = weight_decay
        self.mu = momentum
        self.v = [t.zeros(1) for _ in self.params]
        if self.mu > 0:
            self.b = [t.zeros(1) for _ in self.params]

    def zero_grad(self) -> None:
        for param in self.params:
            param.grad = None

    @t.inference_mode()
    def step(self) -> None:
        for i, param in enumerate(self.params):
            g = param.grad
            if self.lmda != 0:
                g += self.lmda * param
            self.v[i] = self.v[i] * self.alpha + (1 - self.alpha) * g ** 2
            if self.mu > 0:
                self.b[i] = self.mu * self.b[i] + g / ((self.v[i]).sqrt() + self.eps)
                self.params[i] -= self.lr * self.b[i]
            else:
                self.params[i] -= self.lr * g / ((self.v[i]).sqrt() + self.eps)

    def __repr__(self) -> str:
        return f"RMSprop(lr={self.lr}, eps={self.eps}, momentum={self.mu}, weight_decay={self.lmda}, alpha={self.alpha})"



if MAIN:
    tests.test_rmsprop(RMSprop)
# %%
class Adam:
    def __init__(
        self,
        params: Iterable[t.nn.parameter.Parameter],
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-08,
        weight_decay: float = 0.0,
    ):
        '''Implements Adam.

        Like the PyTorch version, but assumes amsgrad=False and maximize=False
            https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
        '''
        params = list(params) # turn params into a list (because it might be a generator)
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.beta1 = t.tensor(self.beta1)
        self.beta2 = t.tensor(self.beta2)
        self.eps = eps
        self.lmda = weight_decay
        self.m = [t.zeros(1) for _ in params]
        self.v = [t.zeros(1) for _ in params]
        self.t = t.tensor(1)

    def zero_grad(self) -> None:
        for param in self.params:
            param.grad = None

    @t.inference_mode()
    def step(self) -> None:
        for i, (param, m, v) in enumerate(zip(self.params, self.m, self.v)):
            g = param.grad
            if self.lmda != 0:
                g += self.lmda * param
            self.m[i] = self.beta1 * m + (1 - self.beta1) * g
            self.v[i] = self.beta2 * v + (1 - self.beta2) * g ** 2
            m_hat = self.m[i] / (1 - t.pow(self.beta1, self.t))
            v_hat = self.v[i] / (1 - t.pow(self.beta2, self.t))
            self.params[i] -= self.lr * m_hat / (v_hat.sqrt() + self.eps)
        self.t += 1

    def __repr__(self) -> str:
        return f"Adam(lr={self.lr}, beta1={self.beta1}, beta2={self.beta2}, eps={self.eps}, weight_decay={self.lmda})"



if MAIN:
    tests.test_adam(Adam)
# %%
def get_cifar(subset: int = 1):
    cifar_trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=IMAGENET_TRANSFORM)
    cifar_testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=IMAGENET_TRANSFORM)
    if subset > 1:
        cifar_trainset = Subset(cifar_trainset, indices=range(0, len(cifar_trainset), subset))
        cifar_testset = Subset(cifar_testset, indices=range(0, len(cifar_testset), subset))
    return cifar_trainset, cifar_testset


if MAIN:
    cifar_trainset, cifar_testset = get_cifar()

    imshow(
        cifar_trainset.data[:15],
        facet_col=0,
        facet_col_wrap=5,
        facet_labels=[cifar_trainset.classes[i] for i in cifar_trainset.targets[:15]],
        title="CIFAR-10 images",
        height=600
    )
# %%
@dataclass
class ResNetTrainingArgs():
    batch_size: int = 64
    max_epochs: int = 3
    max_steps: int = 500
    optimizer: t.optim.Optimizer = t.optim.Adam
    learning_rate: float = 1e-3
    log_dir: str = os.getcwd() + "/logs"
    log_name: str = "day4-resnet"
    log_every_n_steps: int = 1
    n_classes: int = 10
    subset: int = 10
# %%
class LitResNet(pl.LightningModule):
    def __init__(self, args: ResNetTrainingArgs):
        super().__init__()
        self.args = args
        self.resnet = get_resnet_for_feature_extraction(self.args.n_classes)
        self.trainset, self.testset = get_cifar(subset=self.args.subset)

    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.resnet(x)

    def _shared_train_val_step(self, batch: Tuple[t.Tensor, t.Tensor]) -> Tuple[t.Tensor, t.Tensor]:
        imgs, labels = batch
        logits = self(imgs)
        return logits, labels

    def training_step(self, batch: Tuple[t.Tensor, t.Tensor], batch_idx: int) -> t.Tensor:
        logits, labels = self._shared_train_val_step(batch)
        loss = F.cross_entropy(logits, labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: Tuple[t.Tensor, t.Tensor], batch_idx: int) -> None:
        logits, labels = self._shared_train_val_step(batch)
        classifications = logits.argmax(dim=1)
        accuracy = t.sum(classifications == labels) / len(classifications)
        self.log("accuracy", accuracy)

    def configure_optimizers(self):
        return self.args.optimizer(self.resnet.out_layers.parameters(), lr=self.args.learning_rate)

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.args.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.testset, batch_size=self.args.batch_size, shuffle=True)
# %%
if MAIN:
    args = ResNetTrainingArgs()
    model = LitResNet(args)
    logger = CSVLogger(save_dir=args.log_dir, name=args.log_name)

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        logger=logger,
        log_every_n_steps=args.log_every_n_steps,
    )
    trainer.fit(model=model)

    metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")

    plot_train_loss_and_test_accuracy_from_metrics(metrics, "Feature extraction with ResNet34")
# %%
def test_resnet_on_random_input(n_inputs: int = 3):
    indices = np.random.choice(len(cifar_trainset), n_inputs).tolist()
    classes = [cifar_trainset.classes[cifar_trainset.targets[i]] for i in indices]
    imgs = cifar_trainset.data[indices]
    with t.inference_mode():
        x = t.stack(list(map(IMAGENET_TRANSFORM, imgs)))
        logits: t.Tensor = model.resnet(x)
    probs = logits.softmax(-1)
    if probs.ndim == 1: probs = probs.unsqueeze(0)
    for img, label, prob in zip(imgs, classes, probs):
        display(HTML(f"<h2>Classification probabilities (true class = {label})</h2>"))
        imshow(
            img, 
            width=200, height=200, margin=0,
            xaxis_visible=False, yaxis_visible=False
        )
        bar(
            prob,
            x=cifar_trainset.classes,
            template="ggplot2", width=600, height=400,
            labels={"x": "Classification", "y": "Probability"}, 
            text_auto='.2f', showlegend=False,
        )


if MAIN:
    test_resnet_on_random_input()
# %%
import wandb
# %%
@dataclass
class ResNetTrainingArgsWandb(ResNetTrainingArgs):
    run_name: Optional[str] = None
# %%
if MAIN:
    args = ResNetTrainingArgsWandb()
    model = LitResNet(args)
    logger = WandbLogger(save_dir=args.log_dir, project=args.log_name, name=args.run_name)

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        logger=logger,
        log_every_n_steps=args.log_every_n_steps,
    )
    trainer.fit(model=model)
    wandb.finish()
# %%
if MAIN:
    sweep_config = dict()
#     Hyperparameters are chosen randomly, according to the distributions given in 
#the dictionary

# Your goal is to maximize the accuracy metric (note that this is one of the
#  metrics we logged in the Lightning training class above)

# The hyperparameters you vary are:
# learning_rate - a log-uniform distribution between 1e-4 and 1e-1
# batch_size - randomly chosen from (32, 64, 128, 256)
# max_epochs - randomly chosen from (1, 2, 3)

    # YOUR CODE HERE - fill `sweep_config`
    sweep_config = {
        'method': 'random',
        'name': 'sweep',
        'metric': {
            'goal': 'maximize',
            'name': 'accuracy',
        },
        'parameters': {
            'learning_rate': {
                'distribution': 'log_uniform_values',
                'min': 1e-4,
                'max': 1e-1,
            },
            'batch_size': {
                "values": [32, 64, 128, 256],
            },
            'max_epochs': {
                "values": [1, 2, 3],
            }
        }
    }

    tests.test_sweep_config(sweep_config)
# %%
