import numpy as np
import pandas as pd

# agent ###############################################################################################################

class FiniteBernoulliBanditEpsilonGreedy(Agent):
  """Simple agent made for finite armed bandit problems."""

  def __init__(self, n_arm, a0=1, b0=1, epsilon=0.0):
    self.n_arm = n_arm
    self.epsilon = epsilon
    self.prior_success = np.array([a0 for arm in range(n_arm)])
    self.prior_failure = np.array([b0 for arm in range(n_arm)])

  def set_prior(self, prior_success, prior_failure):
    # Overwrite the default prior
    self.prior_success = np.array(prior_success)
    self.prior_failure = np.array(prior_failure)

  def get_posterior_mean(self):
    return self.prior_success / (self.prior_success + self.prior_failure)

  def get_posterior_sample(self):
    return np.random.beta(self.prior_success, self.prior_failure)

  def update_observation(self, observation, action, reward):
    # Naive error checking for compatibility with environment
    assert observation == self.n_arm

    if np.isclose(reward, 1):
      self.prior_success[action] += 1
    elif np.isclose(reward, 0):
      self.prior_failure[action] += 1
    else:
      raise ValueError('Rewards should be 0 or 1 in Bernoulli Bandit')

class FiniteBernoulliBanditTS(FiniteBernoulliBanditEpsilonGreedy):
  """Thompson sampling on finite armed bandit."""

  def random_argmax(vector):
      """Helper function to select argmax at random... not just first one."""
      index = np.random.choice(np.where(vector == vector.max())[0])
      return index

  def pick_action(self, observation):
    """Thompson sampling with Beta posterior for action selection."""
    sampled_means = self.get_posterior_sample()
    action = self.random_argmax(sampled_means)
    return action

# environment #########################################################################################################
# is given by the real experiment setup
# apply selected action
# make observation
# calculate reward

# experiment ##########################################################################################################
class BaseExperiment(object):
  """Simple experiment that logs regret and action taken.

  If you want to do something more fancy then you should extend this class.
  """

  def __init__(self, agent, environment, n_steps,
               seed=0, rec_freq=1, unique_id='NULL'):
    """Setting up the experiment.

    Note that unique_id should be used to identify the job later for analysis.
    """
    self.agent = agent
    self.environment = environment
    self.n_steps = n_steps
    self.seed = seed
    self.unique_id = unique_id

    self.results = []
    self.data_dict = {}
    self.rec_freq = rec_freq


  def run_step_maybe_log(self, t):
    # Evolve the bandit (potentially contextual) for one step and pick action
    observation = self.environment.get_observation()
    action = self.agent.pick_action(observation)

    # Compute useful stuff for regret calculations
    optimal_reward = self.environment.get_optimal_reward()
    expected_reward = self.environment.get_expected_reward(action)
    reward = self.environment.get_stochastic_reward(action)

    # Update the agent using realized rewards + bandit learing
    self.agent.update_observation(observation, action, reward)

    # Log whatever we need for the plots we will want to use.
    instant_regret = optimal_reward - expected_reward
    self.cum_regret += instant_regret

    # Advance the environment (used in nonstationary experiment)
    self.environment.advance(action, reward)

    if (t + 1) % self.rec_freq == 0:
      self.data_dict = {'t': (t + 1),
                        'instant_regret': instant_regret,
                        'cum_regret': self.cum_regret,
                        'action': action,
                        'unique_id': self.unique_id}
      self.results.append(self.data_dict)


  def run_experiment(self):
    """Run the experiment for n_steps and collect data."""
    np.random.seed(self.seed)
    self.cum_regret = 0
    self.cum_optimal = 0

    for t in range(self.n_steps):
      self.run_step_maybe_log(t)

    self.results = pd.DataFrame(self.results)