import numpy as np
import matplotlib.pyplot as plt
from kmeans import plot_k_means, get_simple_data, cost


def main():
  X = get_simple_data()

  plt.scatter(X[:,0], X[:,1])
  plt.show()

  costs = np.empty(10)
  costs[0] = None
  for k in range(1, 10):
    M, R = plot_k_means(X, k, show_plots=False)
    c = cost(X, R, M)
    costs[k] = c

  plt.plot(costs)
  plt.title("Cost vs K")
  plt.show()


if __name__ == '__main__':
  main()


"""the steep drop ends at k=3===> k=3 is the natural number of clusters"""