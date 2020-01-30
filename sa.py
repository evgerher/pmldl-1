import numpy as np

def simulated_annealing(X, E,
                        T0=5.9, T_decay=lambda T, a: T - a,
                        alpha=0.01, max_time=15000,
                        g='random', p=None,
                        g_function=None, alpha_decay=None):
  """
  X - sample space
  E - energy function (computable function), also can be understood as Y
  -----------------
  'predefined':
  T0 - initial temperature
  T_decay - temperature anneal rule (alpha < 1), works well if small,
    thus the difference between epochs of P function will be higher. (for rule `T - a`)
  alpha - decay of temperature,
    works well when small (if rule is `T - a`)
  alpha_decay: f(alpha, t) - function defines decay rule for alpha depending on iteration
  max_time - maximum amount of iterations (t)
  -----------------
  Hyper features
  g - sampling strategy:
    - random: random sample from sample space
    - close: neighbour sample
  g_function: user-defined sampling function
    - should take a form of `lambda X, sample: f(X, sample)` - return item of sample shape
  -----------------
  p - energy distribution function, needs to be manually defined
    - None -> test distribution will be used
    - should take a form of `lambda x, E, T: f(E(x), T)`
  """

  def _g_radnom(X, x_old):
    return np.random.choice(X)

  def default_proportional_function(x, E, T):
    return np.exp(-E(x) / T)

  if p is None:
    p = default_proportional_function

  if g_function is not None:
    g = g_function
  elif g == 'random':
    g = _g_radnom

  T = T0
  distributions = []
  # xt \equiv $x_t$, xt_ \equiv $x_{t+1}$
  xt = g(X, X[0])
  positions = [xt]

  p1 = p(xt, E, T)
  for t in range(max_time):
    p2 = p1
    xt_ = g(X, xt)
    p1 = p(xt_, E, T)

    ratio = p1 / p2

    u = np.random.uniform()
    if t % 100 == 0:
      distributions += [p(X, E, T)]

    if u <= ratio:  # accept x'
      xt = xt_
      T = T_decay(T, alpha)

    positions += [xt]

    if alpha_decay is not None:
      alpha = alpha_decay(alpha, t)

    if T < 1e-6 or T < 0:
      break

  print(f'SA took {t} steps, final temperature = {T}')
  return xt_, np.array(positions), np.array(distributions)