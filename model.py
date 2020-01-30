import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np
from sa import simulated_annealing

class GaussianSampler:
  def __init__(self, mu=0., std=1.):
    self.mu = mu
    self.std = std

  def sample(self, shape):
    return torch.FloatTensor(*shape).normal_(self.mu, self.std)


class UniformSampler:
  def __init__(self, minval=-0.5, maxval=0.5):
    self.minval = minval
    self.maxval = maxval

  def sample(self, shape):
    return torch.FloatTensor(*shape).uniform_(self.minval, self.maxval)


class IrisNet(nn.Module):
  def __init__(self):
    super(IrisNet, self).__init__()
    self.fc1 = nn.Linear(4, 50)
    self.fc2 = nn.Linear(50, 50)
    self.fc3 = nn.Linear(50, 3)
    self.softmax = nn.Softmax(dim=1)
    self.apply(self.init_weights)

  def forward(self, X):
    X = F.relu(self.fc1(X))
    X = self.fc2(X)
    X = self.fc3(X)
    X = self.softmax(X)

    return X

  def fit(self, X_train, X_test, Y_train, Y_test, sampler=GaussianSampler(0., 1.),
          T0=5.9, alpha=1e-2, T_decay=lambda T, a: T - a, alpha_decay=None):
    params = [
      self.fc1.weight,
      self.fc2.weight,
      self.fc3.weight,
      self.fc1.bias,
      self.fc2.bias,
      self.fc3.bias
    ]

    X_train, X_test = map(lambda x: torch.tensor(x).type(torch.FloatTensor),
                          (X_train.values, X_test.values))
    Y_train, Y_test = map(lambda x: torch.tensor(x).type(torch.LongTensor).squeeze(),
                          (Y_train.values, Y_test.values))

    loss = nn.CrossEntropyLoss()

    def energy_function_train(X):
      train_output = self(X_train)
      train_loss = loss(train_output, Y_train)
      return train_loss

    def g(X, *args):
      for i in range(len(X)):
        X[i].data += sampler.sample(X[i].data.shape)
      return X

    def p(x, E, T):
      return torch.exp(-E(x) / T)

    # train
    simulated_annealing(params, energy_function_train, g_function=g, p=p, T0=T0,
                        alpha=alpha, T_decay=T_decay, alpha_decay=alpha_decay)

    # test
    test_output = self(X_test)
    test_loss = loss(test_output, Y_test)
    print(f'Test loss : {test_loss:.4f}')

    _, predict_y = torch.max(test_output, 1)
    print(f'prediction accuracy: {accuracy_score(Y_test.data, predict_y.data)}')

    print(f'macro precision: {precision_score(Y_test.data, predict_y.data, average="macro")}')
    print(f'micro precision: {precision_score(Y_test.data, predict_y.data, average="micro")}')
    print(f'macro recall: {recall_score(Y_test.data, predict_y.data, average="macro")}')
    print(f'micro recall: {recall_score(Y_test.data, predict_y.data, average="micro")}')

  def init_weights(self, m):
    if type(m) == nn.Linear:
      torch.nn.init.xavier_uniform(m.weight)
      m.bias.data.fill_(np.random.uniform())

  def fit_sgd(self, X_train, X_test, Y_train, Y_test):
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(self.parameters(), lr=0.01)

    X_train, X_test = map(lambda x: torch.tensor(x).type(torch.FloatTensor),
                          (X_train.values, X_test.values))
    Y_train, Y_test = map(lambda x: torch.tensor(x).type(torch.LongTensor).squeeze(),
                          (Y_train.values, Y_test.values))

    for epoch in range(1000):
      optimizer.zero_grad()
      out = self(X_train)
      _loss = loss(out, Y_train)
      _loss.backward()
      optimizer.step()

      if epoch % 100 == 0:
        print(f'number of epoch: {epoch}, loss: {_loss}')

    test_output = self(X_test)
    test_loss = loss(test_output, Y_test)
    print(f'Test loss : {test_loss:.4f}')

    _, predict_y = torch.max(test_output, 1)
    _, predict_y = torch.max(test_output, 1)
    print(f'prediction accuracy: {accuracy_score(Y_test.data, predict_y.data)}')

    print(f'macro precision: {precision_score(Y_test.data, predict_y.data, average="macro")}')
    print(f'micro precision: {precision_score(Y_test.data, predict_y.data, average="micro")}')
    print(f'macro recall: {recall_score(Y_test.data, predict_y.data, average="macro")}')
    print(f'micro recall: {recall_score(Y_test.data, predict_y.data, average="micro")}')


