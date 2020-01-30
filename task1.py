import pandas as pd
import sklearn
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from time import time

from model import *


def alpha_decay(a, iteration):
  if iteration % 500 == 0:
    return a * 0.95
  return a

def sa_uniform_sampler(X_train, X_test, Y_train, Y_test, minval=-1, maxval=1):
  print("\n\t Evaluating IrisNet with Uniform Sampler")
  net = IrisNet()
  t1 = time()
  net.fit(X_train, X_test, Y_train, Y_test,
          T0=20000.0, alpha=4.5,
          alpha_decay=alpha_decay,
          T_decay=lambda T, a: T - a,
          sampler=UniformSampler(minval, maxval))
  t2 = time() - t1
  print(f"SA with uniform sampling took {t2:.3f} seconds")


def sa_gaussian_sampler(X_train, X_test, Y_train, Y_test, mu=0, std=3):
  print("\n\t Evaluating IrisNet with Gaussian Sampler")
  net = IrisNet()
  t1 = time()
  net.fit(X_train, X_test, Y_train, Y_test,
          T0=4000.0, alpha=1.5,
          alpha_decay=alpha_decay,
          T_decay=lambda T, a: T - a,
          sampler=GaussianSampler(mu, std))
  t2 = time() - t1
  print(f"SA with gaussian sampling took {t2:.3f} seconds")

def iris_sgd(X_train, X_test, Y_train, Y_test):
  print("\n\t Evaluating IrisNet with SGD")
  net = IrisNet()
  t1 = time()
  net.fit_sgd(X_train, X_test, Y_train, Y_test)
  t2 = time() - t1
  print(f"Adam took {t2:.3f} seconds")

def main():
  iris = sklearn.datasets.load_iris()
  X, Y = iris.data, iris.target

  scaler = preprocessing.StandardScaler()

  X = pd.DataFrame(scaler.fit_transform(X))
  Y = pd.DataFrame(Y.reshape(-1))
  X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

  sa_uniform_sampler(X_train, X_test, Y_train, Y_test)
  sa_gaussian_sampler(X_train, X_test, Y_train, Y_test)
  iris_sgd(X_train, X_test, Y_train, Y_test)


if __name__ == '__main__':
  main()