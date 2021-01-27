
from velec import Velec
import numpy as np

class Bandit():

    def __init__(self, electrode, amplitude, prior_param_ind = None, prior_param_mid = None, prior_param_thumb = None):
        self.electrode = electrode
        self.amplitude = amplitude
        [self.prior_success_ind, self.prior_failure_ind] = prior_param_ind or [1,1]
        [self.prior_success_mid, self.prior_failure_mid] = prior_param_mid or [1,1]
        [self.prior_success_thumb, self.prior_failure_thumb] = prior_param_thumb or [1,1]
        self.priors = [[self.prior_success_ind, self.prior_failure_ind],
                       [self.prior_success_mid, self.prior_failure_mid],
                       [self.prior_success_thumb, self.prior_failure_thumb]]
        self.best_accuracy = [0, 0, 0]
        self.undesired_mov = [[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0]]

    def define_velec(self, ser):
        name = 'velec' + str(self.electrode)
        velec = Velec(ser, 8, name)
        velec.cathodes(self.electrode)
        velec.amplitudes(self.amplitude)
        velec.widths([300])
        velec.anodes([2])
        # print(velec)
        velec.define()
        return velec

    def get_posterior_mean(self):
        posterior_means = [self.prior_success_ind / (self.prior_success_ind + self.prior_failure_ind),
                           self.prior_success_mid / (self.prior_success_mid + self.prior_failure_mid),
                           self.prior_success_thumb / (self.prior_success_thumb + self.prior_failure_thumb)]
        return posterior_means

    def get_posterior_sample(self):
        posterior_samples = [np.random.beta(self.prior_success_ind, self.prior_failure_ind),
                             np.random.beta(self.prior_success_mid, self.prior_failure_mid),
                             np.random.beta(self.prior_success_thumb, self.prior_failure_thumb)]
        return posterior_samples

    def update_observation(self, rewards, accuracys, undesired_movs):
        # index finger
        if np.isclose(rewards[0], 1):
            self.prior_success_ind += 1
        elif np.isclose(rewards[0], 0):
            self.prior_failure_ind += 1
        else:
            raise ValueError('Rewards should be 0 or 1 in Bernoulli Bandit')

        if accuracys[0] > self.best_accuracy[0]:
            self.best_accuracy[0] = accuracys[0]
            self.undesired_mov[0] = undesired_movs[0]


        # middle finger
        if np.isclose(rewards[1], 1):
            self.prior_success_mid += 1
        elif np.isclose(rewards[1], 0):
            self.prior_failure_mid += 1
        else:
            raise ValueError('Rewards should be 0 or 1 in Bernoulli Bandit')

        if accuracys[1] > self.best_accuracy[1]:
            self.best_accuracy[1] = accuracys[1]
            self.undesired_mov[1] = undesired_movs[1]


        # thumb
        if np.isclose(rewards[2], 1):
            self.prior_success_thumb += 1
        elif np.isclose(rewards[2], 0):
            self.prior_failure_thumb += 1
        else:
            raise ValueError('Rewards should be 0 or 1 in Bernoulli Bandit')

        if accuracys[2] > self.best_accuracy[2]:
            self.best_accuracy[2] = accuracys[2]
            self.undesired_mov[2] = undesired_movs[2]

    def __repr__(self):
        string = 'Electrodes %s with Amplitude %s mA \n' % (self.electrode, self.amplitude)

        return string




