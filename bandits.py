
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
        self.accuracys = []
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

    def update_observation(self, accuracys, undesired_movs):
        # Add currently observed accuracys to bandit's history
        self.accuracys.append(accuracys)

        for k in range(3):
            # Update alpha and beta with merged accuracy in the interval [0,1]
            self.priors[k][0] += accuracys[k] * 2
            self.priors[k][1] += (1 - accuracys[k]) * 2

            # Update best accuracy
            if accuracys[k] > self.best_accuracy[k]:
                self.best_accuracy[k] = accuracys[k]
                self.undesired_mov[k] = undesired_movs[k]

        # index finger
        # Update alpha and beta NOT with reward (0 OR 1), BUT with merged accuracy in the interval [0,1]
        self.prior_success_ind += accuracys[0] * 2
        self.prior_failure_ind += (1 - accuracys[0]) * 2

        # middle finger
        # update alpha and beta NOT with reward (0 OR 1), BUT with merged accuracy in the interval [0,1]
        self.prior_success_mid += accuracys[1] * 2
        self.prior_failure_mid += (1 - accuracys[1]) * 2

        # thumb
        # update alpha and beta NOT with reward (0 OR 1), BUT with merged accuracy in the interval [0,1]
        self.prior_success_thumb += accuracys[2] * 2
        self.prior_failure_thumb += (1 - accuracys[2]) * 2

    def __repr__(self):
        string = 'Electrodes %s with Amplitude %s mA' % (self.electrode, self.amplitude)

        return string




