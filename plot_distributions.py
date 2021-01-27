import pickle
from scipy.stats import beta
import matplotlib.pyplot as plt
import numpy as np

bandits = pickle.load(open('bandits_0128.pkl', 'rb'))

numbBandits = 5

best_accuracys = []
for i in range(len(bandits)):
    best_accuracys.append(bandits[i].best_accuracy)

vector = []
fig1, axs = plt.subplots(3)
cmap = plt.get_cmap('brg')
colors = [cmap(i) for i in np.linspace(0, 1, numbBandits)]
titles = ['index', 'middle', 'thumb']

for k in range(3):
    # vector for best accuracy regarding finger k
    vector.append(np.array(best_accuracys)[:, k])
    index = [np.random.choice(np.where(vector[k] == vector[k].max())[0])]
    print(np.where(vector[k] == vector[k].max())[0], index)
    print(bandits[index[0]])

    while len(index) < numbBandits:
        new_index = np.random.choice(np.where(vector[k] != 0)[0])
        if new_index not in index:
            index.append(new_index)

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

        axs[k].plot(x[i], rv[i].pdf(x[i]), color = colors[i], lw=2, label='frozen pdf')
        axs[k].set_xlabel('')
        axs[k].set_ylabel('PDF')
        axs[k].grid()
        axs[k].title.set_text(titles[k])
        print('Accuracy: %s     Posterior Mean: %s     a: %s       b: %s' % (bandits[index[i]].best_accuracy[k], bandits[index[i]].get_posterior_mean()[k], a[i], b[i]))
    print('\n')



best_accuracys = []
posterior_means = []
numb_of_stim = []
for i in range(len(bandits)):
    best_accuracys.append(bandits[i].best_accuracy)
    posterior_means.append(bandits[i].get_posterior_mean())
    numb_of_stim.append([bandits[i].prior_success_ind + bandits[i].prior_failure_ind])


fig2, axs = plt.subplots(1, 3)

for k in range(3):
    x = np.array(best_accuracys)[:, k]
    y = np.array(posterior_means)[:, k]
    z = np.array(numb_of_stim)[:]
    scatter = axs[k].scatter(x, y, c = z)
    axs[k].set_xlabel('best accuracy')
    axs[k].set_ylabel('posterior mean')
    axs[k].grid()
    axs[k].title.set_text(titles[k])
    legend = axs[k].legend(*scatter.legend_elements(), loc="upper left", title="Number of\nStimulations")

plt.show()
