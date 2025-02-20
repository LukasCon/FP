import pickle
from scipy.stats import beta
import matplotlib.pyplot as plt
import numpy as np

# Choose file to load the bandits
bandits = pickle.load(open('bandits_0215_1.pkl', 'rb'))

# How many bandits for each finger should be plotted
numbBandits = 10

# Define figure
fig1, axs = plt.subplots(3)
cmap = plt.get_cmap('brg')
colors = [cmap(i) for i in np.linspace(0, 1, numbBandits)]
titles = ['index', 'middle', 'thumb']

for k in range(3):

    # Extract the best accuracys and the posterior means of all bandits
    best_accuracys = []
    posterior_means = []
    for i in range(len(bandits)):
        best_accuracys.append(bandits[i].best_accuracy[k])
        posterior_means.append(bandits[i].get_posterior_mean()[k])

    # Sort bandits by best accuracy
    index = [best_accuracys.index(x) for x in sorted(best_accuracys, reverse=True)[:numbBandits]]
    print('Best accuracy:', bandits[index[0]])
    print('2nd best accuracy:', bandits[index[1]])
    print('3rd best accuracy:', bandits[index[2]])
    index_best_pos = [posterior_means.index(x) for x in sorted(posterior_means, reverse=True)]
    print('Highest posterior mean:', bandits[index_best_pos[0]])

    # Plot
    a, b, x, rv = [], [], [], []

    for i in range(len(index)):

        a.append(bandits[index[i]].priors[k][0])
        b.append(bandits[index[i]].priors[k][1])
        x.append(np.linspace(beta.ppf(0.01, a[i], b[i]), beta.ppf(0.99, a[i], b[i]), 100))
        rv.append(beta(a[i], b[i]))

        axs[k].plot(x[i], rv[i].pdf(x[i]), color = colors[i], lw=2, label='frozen pdf')
        axs[k].set_xlabel('')
        axs[k].set_ylabel('PDF')
        axs[k].grid()
        axs[k].title.set_text(titles[k])
        print('Accuracy: %s     Posterior Mean: %s     a: %s       b: %s' % (bandits[index[i]].best_accuracy[k], bandits[index[i]].get_posterior_mean()[k], a[i], b[i]))
    print('\n')


# Show how many stimulations were applied for this whole set of bandits (including stimulations of the priors)
numb_of_stim = []
for i in range(len(bandits)):
    numb_of_stim.append((bandits[i].prior_success_ind + bandits[i].prior_failure_ind - 2)/2)

print('Number of Stimulations:', sum(numb_of_stim))

# Second figure
fig2, axs = plt.subplots(1, 3)

for k in range(3):

    # Extract the best accuracys and the posterior means of all bandits
    best_accuracys = []
    posterior_means = []
    for i in range(len(bandits)):
        best_accuracys.append(bandits[i].best_accuracy[k])
        posterior_means.append(bandits[i].get_posterior_mean()[k])

    x = np.array(best_accuracys)[:]
    y = np.array(posterior_means)[:]
    z = np.array(numb_of_stim)[:]
    scatter = axs[k].scatter(x, y, c = z)
    axs[k].set_xlabel('best accuracy')
    axs[k].set_ylabel('posterior mean')
    axs[k].grid()
    axs[k].title.set_text(titles[k])
    legend = axs[k].legend(*scatter.legend_elements(), loc="upper left", title="Number of\nStimulations")

plt.show()
