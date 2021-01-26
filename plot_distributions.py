import pickle
from scipy.stats import beta
import matplotlib.pyplot as plt
import numpy as np

bandits = pickle.load(open('bandits_0128_2.pkl', 'rb'))

numbBandits = 10

best_accuracys = []
for i in range(len(bandits)):
    best_accuracys.append(bandits[i].best_accuracy)

vector = []
fig, axs = plt.subplots(numbBandits, 3)
for k in range(3):
    # vector for best accuracy regarding finger k
    vector.append(np.array(best_accuracys)[:, k])
    index = [np.random.choice(np.where(vector[k] == vector[k].max())[0])]
    print(np.where(vector[k] == vector[k].max())[0], index)
    print(bandits[index[0]])

    for _ in range(numbBandits -1):
        index.append(np.random.choice(np.where(vector[k] != 0)[0]))

    # Plot
    a, b, x, rv = [], [], [], []
    for i in range(len(index)):
        if k == 0:
            a.append(bandits[index[i]].prior_success_ind)
            b.append(bandits[index[i]].prior_failure_ind)
        elif k == 1:
            a.append(bandits[index[i]].prior_success_mid)
            b.append(bandits[index[i]].prior_failure_mid)
        elif k == 2:
            a.append(bandits[index[i]].prior_success_thumb)
            b.append(bandits[index[i]].prior_failure_thumb)

        x.append(np.linspace(beta.ppf(0.01, a[i], b[i]), beta.ppf(0.99, a[i], b[i]), 100))
        rv.append(beta(a[i], b[i]))

        axs[i, k].plot(x[i], rv[i].pdf(x[i]), 'k-', lw=2, label='frozen pdf')
        print('Accuracy: %s     Posterior Mean: %s     a: %s       b: %s' % (bandits[index[i]].best_accuracy[k], bandits[index[i]].get_posterior_mean()[k], a[i], b[i]))
    print('\n')

plt.show()
