#%%
import os; os.environ["ACCELERATE_DISABLE_RICH"] = "1"
import sys
import torch
import torchvision
from torch.utils import benchmark
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import copy

from collections import namedtuple

import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.profiler import profile, record_function, ProfilerActivity
from pathlib import Path

from typing import List, Optional, Callable, Tuple, Dict, Literal, Set 
# Make sure exercises are in the path
orig_dir = os.getcwd()
chapter = r"chapter3_training_at_scale"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part7_toy_models_of_superposition"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

import part1_gpus.tests as tests

# Add root dir, so we can import from chapter 0 material
root_dir = exercises_dir.parent.parent.resolve()
if str(root_dir) not in sys.path: sys.path.append(str(root_dir))
os.chdir(root_dir)
# from chapter0_fundamentals.exercises.part3_resnets.solutions import ResNet34
from chapter0_fundamentals.exercises.part3_resnets.answers import ResNet34
os.chdir(orig_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAIN = __name__ == "__main__"
# %%
model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
inputs = torch.randn(5, 3, 224, 224)

with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    with record_function("model_inference"):
        model(inputs)

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
# %%
inputs = torch.randn(5, 3, 224, 224)
model = torchvision.models.resnet18(weights='IMAGENET1K_V1')

with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    with record_function("model_inference"):
        with torch.inference_mode():
            model(inputs)

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
# %%
model = torchvision.models.resnet18(weights='IMAGENET1K_V1').cuda()
inputs = torch.randn(5, 3, 224, 224).cuda()

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    with record_function("model_inference"):
        model(inputs)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
# %%
inputs = torch.randn(5, 3, 224, 224).cuda()
model = torchvision.models.resnet18(weights='IMAGENET1K_V1').cuda()

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    with record_function("model_inference"):
        with torch.inference_mode():
            model(inputs)
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
# %%
model = torchvision.models.resnet18(weights='IMAGENET1K_V1').cuda()
inputs = torch.randn(5, 3, 224, 224).cuda()

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    model(inputs)

output = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
print(output)
prof.export_chrome_trace("trace_3.json")
# %%
model = torchvision.models.resnet18(weights='IMAGENET1K_V1').cuda()
inputs = torch.randn(5, 3, 224, 224).cuda()

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    model(inputs)

prof.export_chrome_trace("trace_afterwarmup.json")
# %%
# MY STUFF
model = ResNet34()
inputs = torch.randn(5, 3, 224, 224)

with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    with record_function("model_inference"):
        model(inputs)

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
# %%
model = ResNet34().cuda()
inputs = torch.randn(5, 3, 224, 224).cuda()

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    model(inputs)

prof.export_chrome_trace("trace.json")

output = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
print(output)
# %%
def daxpy(alpha,x,y):
  return alpha*x + y

@torch.jit.script
def fused_daxpy(alpha,x,y):
    return alpha*x + y

def test_daxpy_random_input(fn1, fn2):

    alpha = torch.rand(1, device='cuda')
    x = torch.randn(1823, 1823, device='cuda')
    y = torch.randn(1823, 1823, device='cuda')

    assert torch.allclose(fn1(alpha, x, y), fn2(alpha, x, y), 0, 1e-6), "Implementations are not analogous"
    print('Tests passed')

test_daxpy_random_input(daxpy, fused_daxpy)

print("benching...")
bench_results = []
for contender in [daxpy, fused_daxpy]:
    try:
        name = contender.__name__
    except:
        name = contender.name

    t = benchmark.Timer(
        setup="alpha, x, y = torch.rand(1, device='cuda'),torch.randn(1823, 1823, device='cuda'), torch.randn(1823, 1823, device='cuda') ",
        stmt="function(alpha, x, y)",
        description=f"cuda",
        label="daxpy",
        sub_label=name,
        globals={
            'function': contender
        }
      )
    bench_results.append(t.blocked_autorange(min_run_time=5))


compare = benchmark.Compare(bench_results)
compare.colorize()
compare.print()
# %%
def naive_softmax(x):
    '''Compute row-wise softmax of X using native pytorch

    We subtract the maximum element in order to avoid overflows. Softmax is invariant to
    this shift.
    '''
    # read  MN elements ; write M  elements
    x_max = x.max(dim=1)[0]
    # read MN + M elements ; write MN elements
    z = x - x_max[:, None]
    # read  MN elements ; write MN elements
    numerator = torch.exp(z)
    # read  MN elements ; write M  elements
    denominator = numerator.sum(dim=1)
    # read MN + M elements ; write MN elements
    ret = numerator / denominator[:, None]
    # in total: read 5MN + 2M elements ; wrote 3MN + 2M elements
    return ret

@torch.jit.script
def torchjit_fused_naive_softmax(x):
    '''Compute row-wise softmax of X using native pytorch

    We subtract the maximum element in order to avoid overflows. Softmax is invariant to
    this shift.
    '''
    # read  MN elements ; write M  elements
    x_max = x.max(dim=1)[0]
    # read MN + M elements ; write MN elements
    z = x - x_max[:, None]
    # read  MN elements ; write MN elements
    numerator = torch.exp(z)
    # read  MN elements ; write M  elements
    denominator = numerator.sum(dim=1)
    # read MN + M elements ; write MN elements
    ret = numerator / denominator[:, None]
    # in total: read 5MN + 2M elements ; wrote 3MN + 2M elements
    return ret

print("benching...")
bench_results = []
for contender in [naive_softmax, torchjit_fused_naive_softmax]:
  try:
    name = contender.__name__
  except:
    name = contender.name

  t = benchmark.Timer(
        setup="x = torch.randn(1823, 1823, device='cuda')",
        stmt="function(x)",
        description=f"cuda",
        label="softmax",
        sub_label=name,
        globals={
            'function': contender
        }
    )
  bench_results.append(t.blocked_autorange(min_run_time=5))


compare = benchmark.Compare(bench_results)
compare.colorize()
compare.print()
# %%
class Net(nn.Module):
    def __init__(self, mnist=True):

        super(Net, self).__init__()
        if mnist:
            num_channels = 1
        else:
            num_channels = 3

        self.conv1 = nn.Conv2d(num_channels, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)


    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)   
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % args["log_interval"] == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()
            ))


def test(model, test_loader, device='cuda'):
    model = model.to(device)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():

    batch_size = 64
    test_batch_size = 64
    epochs = 10
    lr = 0.01
    momentum = 0.5
    seed = 1
    log_interval = 500
    save_model = False
    no_cuda = False

    use_cuda = not no_cuda and torch.cuda.is_available()

    torch.manual_seed(seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            '../data', 
            train=True, 
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]
        )),
        batch_size=batch_size,
        shuffle=True,
        **kwargs
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            '../data',
            train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]
        )),
        batch_size=test_batch_size,
        shuffle=True, 
        **kwargs
    )

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    args = {}
    args["log_interval"] = log_interval
    for epoch in range(1, epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, test_loader)

    if (save_model):
        torch.save(model.state_dict(),"mnist_cnn.pt")

    return model


model = main()
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        '../data', 
        train=False, 
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    ),
    batch_size=64, shuffle=True)

test(model, test_loader)
# %%
QTensor = namedtuple('QTensor', ['tensor', 'scale', 'zero_point'])

# Here scale is equal to (max_val — min_val) / (qmax — qmin), where max_val and min_val are maximum and minimum values of 
# X tensor respectively. qmin and qmax represents the range of an 8 bit number (0 and 255 respectively). 
# The scale scales the quantised net and the zero point shifts the number. 

def quantize_tensor(x, min_val=None, max_val=None, num_bits=8) -> QTensor:
    '''
    Calculate the scale and zero-point of the input tensor for quantization.
    '''
    if not min_val and not max_val: 
        min_val = x.min()
        max_val = x.max()
    q_min = 0
    q_max = 2 ** num_bits - 1

    scale = (max_val - min_val) / (q_max - q_min)
    zero_point = torch.clip(- min_val / scale, min=q_min, max=q_max).byte().item()
    tensor = (zero_point + x / scale).round().byte()
    return QTensor(tensor=tensor, scale=scale, zero_point=zero_point)

tests.test_quantize_tensor(quantize_tensor)


def dequantize_tensor(q: QTensor) -> torch.tensor:
    '''
    Dequantize the input QTensor to obtain the float Tensor.
    '''
    # x_Float = scale*(x_Quant -zero_point)
    return q.scale * (q.tensor - q.zero_point)


tests.test_dequantize_tensor(dequantize_tensor)
# %%
def calcScaleZeroPoint(min_val, max_val, num_bits=8) -> Tuple[float, float]:
    '''
    Calculate scale and zero point of
    '''
    q_min = 0
    q_max = 2 ** num_bits - 1

    scale = (max_val - min_val) / (q_max - q_min)
    zero_point = torch.clip(- min_val / scale, min=q_min, max=q_max).byte().item()
    return (scale, zero_point)
    
# %%
# Get Min and max of x tensor, and stores it
def updateStats(x, stats, key) -> Dict[Dict, Dict[str, int]]:
    max_val, _ = torch.max(x, dim=1)
    min_val, _ = torch.min(x, dim=1)


    if key not in stats:
        stats[key] = {"max": max_val.sum().item(), "min": min_val.sum().item(), "total": 1}
    else:
        stats[key]['max'] += max_val.sum().item()
        stats[key]['min'] += min_val.sum().item()
        stats[key]['total'] += 1

    return stats

# Reworked Forward Pass to access activation Stats through updateStats function
def gatherActivationStats(model, x, stats):

    stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv1')
    x = F.relu(model.conv1(x))
    x = F.max_pool2d(x, 2, 2)
    stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv2')
    x = F.relu(model.conv2(x))
    x = F.max_pool2d(x, 2, 2)
    x = x.view(-1, 4*4*50)
    stats = updateStats(x, stats, 'fc1')
    x = F.relu(model.fc1(x))
    stats = updateStats(x, stats, 'fc2')
    x = model.fc2(x)

    return stats

# Entry function to get stats of all functions.
def gatherStats(model, test_loader):
    device = 'cuda'

    model.eval()
    test_loss = 0
    correct = 0
    stats = {}
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            stats = gatherActivationStats(model, data, stats)

    final_stats = {}
    for key, value in stats.items():
      final_stats[key] = { "max" : value["max"] / value["total"], "min" : value["min"] / value["total"] }
    return final_stats
# %%
q_model = copy.deepcopy(model)
stats = gatherStats(q_model, test_loader)
# %%
