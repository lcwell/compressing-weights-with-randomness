import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
from torchvision import datasets, transforms
import tqdm



# Based on https://stackoverflow.com/a/12421820.
# Produces an approximate integer logarithmic scale s.t.
# all numbers are unique, i.e. len(output) = len(set(output)).
# The limit "end" will be included.
def logspace_int(start, end, n):
    end = end + 1
    result = [start + 1]
    if n > 1:
        ratio = (end / result[-1]) ** (1.0 / (n - len(result)))
    while len(result) < n:
        next_value = result[-1] * ratio
        if next_value - result[-1] >= 1:
            result.append(next_value)
        else:
            result.append(result[-1] + 1)
            ratio = (end / result[-1]) ** (1.0 / (n-len(result)))
    return list(map(lambda x: round(x) - 1, result))



def effective_rank(M):
    svals = np.linalg.svd(M, compute_uv=False)
    std_svals = svals / np.sum(svals)
    mask = std_svals > 1e-12
    entropy = -np.sum(std_svals[mask] * np.log(std_svals[mask]))
    return np.exp(entropy)



class MnistFCNet(nn.Module):
    def __init__(self, name, seed=0):
        super(MnistFCNet, self).__init__()
        self.path = f'models/{name}.pt'
        torch.manual_seed(seed)

    def load_from_file(self):
        self.load_state_dict(torch.load(self.path))
        self.eval()

    def save_to_file(self):
        torch.save(self.state_dict(), self.path)
    
    def already_saved(self):
        return os.path.exists(self.path)

    def init_optimizer(self, lr=None, l2_penalty=0, l1_penalty=None, lr_gamma=None):
        assert lr is not None
        self.optimizer = opt.Adam(self.parameters(), lr=lr, weight_decay=l2_penalty)
        if lr_gamma is not None:
            self.scheduler = opt.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=lr_gamma)
        else:
            self.scheduler = None
        self.l1_penalty = l1_penalty
    
    def init_data_loader(self, batch_size=None):
        assert batch_size is not None

        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_data = datasets.MNIST(
            'data', train=True, download=True, transform=transform)
        test_data = datasets.MNIST(
            'data', train=False, download=True, transform=transform)

        self.train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
        self.test_loader = torch.utils.data.DataLoader(test_data, batch_size=256)

    def init_model(self, activation=None, dropout=None):
        assert activation is not None
        assert dropout is not None
        self.flatten = nn.Flatten(1)
        self.l1 = nn.Linear(28*28, 1000)
        self.activation1 = activation
        self.dropout1 = nn.Dropout(dropout) if dropout is not None else nn.Identity()
        self.l2 = nn.Linear(1000,1000)
        self.activation2 = activation
        self.dropout2 = nn.Dropout(dropout) if dropout is not None else nn.Identity()
        self.l3 = nn.Linear(1000, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.l1(x)
        x = self.activation1(x)
        x = self.dropout1(x)
        x = self.l2(x)
        x = self.activation2(x)
        x = self.dropout2(x)
        x = self.l3(x)
        output = F.log_softmax(x, dim=1) # "log" for stability
        return output
    
    def train_one_epoch(self):
        self.train() # set to training mode
        for batch_num, (data, target) in (pbar := tqdm.tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc='')):
            self.optimizer.zero_grad()
            output = self(data)
            loss = F.nll_loss(output, target)
            if self.l1_penalty is not None:
                l1_norm = sum(p.abs().sum() for p in self.parameters())
                loss = loss + self.l1_penalty * l1_norm
            loss.backward()
            self.optimizer.step()
            if batch_num % 10 == 0:
                pbar.set_description(f'  Loss {loss.item():10.6f}   Progress')

    def score(self):
        self.eval()
        loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                output = self(data)
                loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        loss /= len(self.test_loader.dataset)
        acc = 100.0 * correct / len(self.test_loader.dataset)
        return loss, acc
    
    def score_pretty(self):
        print('  Test results:   ', end='')
        loss, acc = self.score()
        print(f'loss({loss:6.4f})   accuracy({acc:6.2f})')
        print(f'  Effective rank of weight matrix: ', end='')
        with torch.no_grad():
            M = self.l2.weight.numpy()
            erank = effective_rank(M)
            print(erank)

    def train_loop(self, num_epochs):
        for epoch in range(num_epochs):
            print(f'Epoch {epoch+1}')
            self.train_one_epoch()
            self.score_pretty()

            if self.scheduler is not None:
                self.scheduler.step()

    def copy_weights_of_interest(self):
        with torch.no_grad():
            return self.l2.weight.numpy().copy()
    
    # woi = "weights of interest", i.e. those of layer 2 (1000x1000)
    def exchange_woi_with_factorization(self, U_numpy, V_numpy):
        U = torch.tensor(U_numpy)
        V = torch.tensor(V_numpy)
        assert len(U.shape) == 2
        assert len(V.shape) == 2
        assert U.size()[0] == 1000
        assert V.size()[1] == 1000
        assert U.size()[1] == V.size()[0]
        l2_approx = torch.matmul(U, V)
        assert len(l2_approx.size()) == 2
        assert l2_approx.size()[0] == 1000
        assert l2_approx.size()[1] == 1000

        with torch.no_grad():
            self.l2.weight.copy_(l2_approx)

