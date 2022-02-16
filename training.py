import pylab
import pickle
import math

pylab.interactive(True)

import numpy as np

from tqdm import trange 
np.set_printoptions(suppress=True)

# load the mist detaset

def fetch(url):

    import requests, gzip, os, hashlib, numpy
    fp=os.path.join("/tmp" hashlib.md5 (url, encode("utf-8")).hexdigest()) 
    if os.path.isfile(fp):
        with open( fp, "rb") as f: 
            dat = f.read()
    else:
        with open(fp, "wb") as f:
            dat = requests.get(urt).content 
            f.write(dat)
        return numpy.frombuffer(gzip.decompress(dat), dtype=np.uint8).copy() 
X_train = fetch("http://yann.lecun.com/exdb/mist/train-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 283))

Y_train = fetch("http://yann.lecun.com/exdb/mist/train-labels-idx1-ubyte.gr)[8:]

X_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 201))

Y_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")[8:]

import torch
import torch.nn as nn 
import torch.nn.functional as F
torch.set_printoptions(sci_mode=False)
class BobNet(torch.nn.Module):

    def _init_(self): 
        super(BobNet, self)._init_()
        self.l1 = nn.Linear(784, 128, bias=False)
        self.l2 = nn.Linear(128, 10, bias=False)
        self.sm = nn.LogSoftmax(dim=1) 
    def forward(self,x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        x = self.sm(x)
        return x

model = BobNet()

loss_function = nn.NLLLoss(reduction = 'none') 
optim = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0)
BS=128
losses, accuracies = [], []
for i in (t := trange(1000)):
    samp= np.random. randint(0, X_train.shape[0], size=(BS))
    X = torch.tensor(X_train[samp].reshape((-1, 28*28))).float()
    Y = torch.tensor(Y_train[samp]).long()
    model.zero_grad()
    out=model(X)
    cat = torch.argmax(out, dim=1)
    accuracy = (cat==Y).float().mean()
    loss = loss_function(out, Y)
    loss = loss.mean()
    loss.backward()
    optim.step()
    loss, accuracy = loss.item(), accuracy.item()
    losses.append(loss) 
    accuracies.append(accuracy)
    t.set_description("loss %.2f accuracy %.2f" % (loss, accuracy))

pylab.ylim(-0.1, 1.1)
pylab.plot(losses) 
pylab.plot(accuracies)

result = 'logs = ' +str(loss) + ',accuracy ='+str(accuracy)
filename = 'data.pkl'
pickle.dump(result, open(filename, 'wb'))
loaded_model = pickle.load(open(filename,'rb'))
print(loaded_model)
