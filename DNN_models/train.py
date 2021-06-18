
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.utils import shuffle

import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable

# For GPU, change accordingly
torch.set_num_threads(2)
print('Running on %i threads!'%torch.get_num_threads())

def encode(geneLst):
    '''One-hot encoder: from a gene representation to a vector'''
    vec = []
    for i in geneLst[:8]:
        tmp = [0 for i in range(42)]
        tmp[i] = 1
        vec += tmp
    return vec

def getEquiv(geneLst):
    '''Rotate the molecule and include the rotated equivalent representations'''
    shiftRight = lambda lst: [lst[-2]] + lst[:-2]
    eqSet = []
    for i in [geneLst]:
        for j in range(4):
            eqSet.append(shiftRight(i))
    return eqSet

def dataAug(dataSet):
    '''Call the equivalent representation generator to augment the dataset'''
    augSet = []
    for g in dataSet:
        for e in getEquiv(g[0]):
            tmp = g.copy()
            tmp[0] = e
            augSet.append(tmp)
    return augSet

def curate(Xdata, Ydata):
    '''Romove bad data points'''
    Xtmp, Ytmp = [], []
    for i in range(len(Xdata)):
        if -10 < Ydata[i] < 10:
            Xtmp.append(Xdata[i])
            Ytmp.append(Ydata[i])
    return np.array(Xtmp), np.array(Ytmp)


data = [l.split() for l in open('history.dat', 'r').readlines()]
data = [
    [[eval(i) for i in l[0].split('-')],
    eval(l[1]), eval(l[2]), eval(l[3]), eval(l[4]),
    eval(l[5]), eval(l[6]), eval(l[7]), eval(l[8])]
    for l in data
]

data = dataAug(data)

X = np.array([encode(l[0]) for l in data])
Y = np.array([l[8] for l in data])
X, Y = curate(X, Y)

ymax, ymin = Y.max(), Y.min()

# # Uncomment to enable normalization
# Y = (Y - ymin) / (ymax - ymin)

print(' > Training set size: %i\n > Representation dimension: %i'%(len(X), len(X[0])))
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

N = len(X_train)
D_in = 336
H1 = 512
H2 = 512
H3 = 256
H4 = 64
H5 = 16
D_out = 1
batch_size = 128

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H1),
    torch.nn.ReLU(),
    torch.nn.Linear(H1, H2),
    torch.nn.ReLU(),
    torch.nn.Linear(H2, H3),
    torch.nn.ReLU(),
    torch.nn.Linear(H3, H4),
    torch.nn.ReLU(),
    torch.nn.Linear(H4, H5),
    torch.nn.ReLU(),
    torch.nn.Linear(H5, D_out)
)
print(model)

learning_rate = 0.01

loss_func = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# # Uncomment to enable SGD as optimizer
# loss_func = torch.nn.MSELoss()
# optimizer = torch.optim.SGD(neu.parameters(), lr = 0.01)

# Train for 100 epoches
for epoch in range(100):
    x, y = shuffle(X_train, Y_train)
    for start in range(0, len(x), batch_size):
        end = start + batch_size if start + batch_size < len(x) else len(x)
        inp = torch.tensor(x[start: end], dtype=torch.float, requires_grad = True)
        out = torch.tensor(y[start: end], dtype=torch.float, requires_grad = True)
        optimizer.zero_grad()
        y_pred = model(inp)
        loss = loss_func(y_pred, out.unsqueeze(1))
        loss.backward()
        optimizer.step()
        myLoss = loss.item()/(end-start)
    myR2 = r2_score(
        model(Variable(torch.tensor(X_train.astype(np.float32)))).detach().numpy(),
        Y_train)
    Y_pred = model(Variable(torch.tensor(X_test.astype(np.float32))))
    R2_test = r2_score(Y_pred.detach().numpy(), Y_test)
    print('EPOCH\t%i\t| LOSS\t%.9f\t| R2-train\t%.9f\t| R2-test\t%.9f'\
        %(epoch, myLoss, myR2, R2_test))
    if R2_test > 0.999:
        print('Converge!')
        break

# Plot the validation results
Y_pred = model(Variable(torch.tensor(X_test.astype(np.float32))))
print('R square on test set: %.6f'%\
    r2_score(Y_pred.detach().numpy(), Y_test))
plt.scatter(Y_test, Y_pred.detach().numpy(), alpha = 0.5, label='Prediction')
plt.plot(Y_test, Y_test, 'r-', label='Test set target')
plt.title('R-square = %.6f'%R2_test)
plt.grid()
plt.axis('scaled')
plt.tight_layout()
plt.legend()
plt.margins(0,0)
plt.savefig('Model_validation')

torch.save(model, 'model.pkl')

# # Model loader
#model = torch.load('model.pkl')
