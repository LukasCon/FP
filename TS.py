import numpy as np
import pandas as pd
import random
from scipy import integrate
from bandits import Bandit
import pickle

ideal_movements = []
for _ in range(6):
    ideal_movements.append(random.sample(range(0, 120), 100))

def pick_action(basedOn, bandits):

    if basedOn == 'ind':
        k = 0
    elif basedOn == 'mid':
        k = 1
    elif basedOn == 'thumb':
        k = 2
    else:
        raise ValueError('Select between ind, mid or thumb to determine from which posterior distribution the next action should be picked')

    sampled_means = []
    for i in range(len(bandits)):
        sampled_means.append(bandits[i].get_posterior_sample()[k])

    def random_argmax(vector):
        """Helper function to select argmax at random... not just first one."""
        index = np.random.choice(np.where(vector == vector.max())[0])
        return index

    action = random_argmax(np.array(sampled_means))
    return action

def calc_reward(basedOn, flexions, ideal_movements):

        if basedOn == 'ind':
            flexions = np.array(flexions[0:2])
            ref_flexion = np.array(ideal_movements[0:2])
        elif basedOn == 'mid':
            flexions = np.array(flexions[2:4])
            ref_flexion = np.array(ideal_movements[2:4])
        elif basedOn == 'thumb':
            flexions = np.array(flexions[4:6])
            ref_flexion = np.array(ideal_movements[4:6])
        else:
            raise ValueError(
                'Select between ind, mid or thumb to determine which flexion to use for reward calculation')

        relative_res = []
        accuracy = []
        # Extract features and calculate rewards
        for m in range(len(flexions)):
            flexion_sequence = flexions[m]
            max_meas = max(flexion_sequence)
            max_ideal = max(ref_flexion[m])
            res_max = abs(max_ideal - max_meas)
            min_meas = min(flexion_sequence)
            min_ideal = min(ref_flexion[m])
            res_min = abs(min_ideal - min_meas)

            # calculation of integrals
            def rel_integral(flexion_sequence):
                integral = integrate.simps(flexion_sequence)  # full integral above x-axis
                x = np.arange(flexion_sequence.shape[0])
                y = np.full_like(x, flexion_sequence[0])
                relative_integral = integral - integrate.simps(y, x)  # relative integral regarding the initial flexion value of the current section
                return relative_integral

            # residual relative integral
            res_rel_int = abs(rel_integral(ref_flexion[m]) - rel_integral(flexion_sequence))

            # relative residuals/ deviation in percentage
            res_max_norm = (res_max / abs(max_ideal))
            res_rel_int_norm = (res_rel_int / abs(rel_integral(ref_flexion[m])))
            relative_res.append([res_max_norm, res_rel_int_norm])

            # Reward function
            accuracy.append(((1 - res_max_norm) + (1 - res_rel_int_norm))/2)

        if accuracy[0] >= 0.5 and accuracy[1] >= 0.5:  # weighting for mcp and pip flexion residual possible
            reward = 1
        else:
            reward = 0

        return reward, accuracy, relative_res

def neighbor_combinations(elec_number):
    A = [[2, 1, 9, 13],
         [4, 5, 10, 14],
         [6, 7, 11, 15],
         [3, 8, 12, 16]]

    def neighbors(matrix, rowNumber, columnNumber):
        radius = 1
        neighbors = []
        if rowNumber > len(matrix) or columnNumber > len(matrix[0]):
            raise ValueError('Requested matrix entry out of bounds')
        else:
            for j in range(columnNumber - radius, columnNumber + 1 + radius):
                for i in range(rowNumber - radius, rowNumber + 1 + radius):
                    if i >= 0 and i < len(matrix) and j >= 0 and j < len(matrix[0]):
                        neighbors.append(matrix[i][j])
            # remove center electrode and 2 because it works as anode
            neighbors.remove(matrix[rowNumber][columnNumber])
            if 2 in neighbors:
                neighbors.remove(2)
        return neighbors

    def index(matrix, val):
        matrix_dim = len(matrix[0])
        item_index = 0
        for row in matrix:
            for i in row:
                if i == val:
                    break
                item_index += 1
            if i == val:
                break
        index = [int(item_index / matrix_dim), item_index % matrix_dim]
        return index

    combinations = []
    index_active_electrode = index(A, elec_number)
    neighbor_electrodes = neighbors(A, index_active_electrode[0], index_active_electrode[1])
    for neighbor in neighbor_electrodes:
        combinations.append(sorted([elec_number, neighbor]))

    return combinations


# Use uniform priors or posterior distribution from last experiment?
use_uniform_priors = True
last_experiment = 'bandits_0120.pkl'

# Overwrite posterior distributions from last experiment?
overwrite = True
new_file = 'test.pkl'


if use_uniform_priors:
    # Define initial bandits/actionspace
    bandits = []
    electrodes = [4, 6, 3, 1, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    amplitudes = [6, 8, 10, 12, 14]
    all_combinations = []

    # Add two combinations to list of electrodes
    for i in electrodes:
        combinations = neighbor_combinations(i)

        # Check if combination is already in electrodes list
        for j in combinations:
            if j not in all_combinations:
                all_combinations.append(j)

    for k in electrodes:
        for l in amplitudes:
            bandits.append(Bandit(k, l))

    for m in all_combinations:
        for n in amplitudes:
            bandits.append(Bandit(m, n))

else:
    bandits = pickle.load(open(last_experiment, 'rb'))

if overwrite:
    save_name = last_experiment
else:
    save_name = new_file

# Initialize parameters
n = 100
n_deeper = 10
aim_options = ['ind', 'mid', 'thumb']
start_bandits = [x for x in bandits if x.electrode in [4, 6, 3, 1, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16] and x.amplitude == 6]

##########################################################################################################################################################################
for t in range(n):
    deeper_search = False
    print('t:', t)
    # aim defines which finger/ posterior distribution is used to pick the following action
    aim = aim_options[0]
    print(aim_options)
    print(aim)

    # Choose action based on maximum of success probability which comes from random sample of the posterior distributions of the bandits
    action = pick_action(aim, start_bandits)
    selected_bandit = start_bandits[action]
    selected_bandit.define_velec()

    flexions = []
    for _ in range(9):
        flexions.append(random.sample(range(0, 120), 100))

    # Stream marker coordinates and calculate flexion
    #flexions = [flexion_ind1, flexion_ind2, flexion_mid1, flexion_mid2, flexion_thumb1, flexion_thumb2, roll, pitch, yaw]

    #Calculate rewards for each finger
    rewards = []
    accuracys = []
    for finger in ['ind', 'mid', 'thumb']:
        reward, accuracy, relative_res = (calc_reward(finger, flexions, ideal_movements))
        rewards.append(reward)

        merged_accuracy = (accuracy[0] + accuracy[1]) /2
        accuracys.append(merged_accuracy)

        # Check if observation is good enough for deeper search
        if merged_accuracy >=0.5 and finger in aim_options:
            deeper_search = True
            aim = finger
            # Initial aim_accuracy
            aim_accuracy = merged_accuracy

    # Update each distribution for selected bandit
    selected_bandit.update_observation(rewards, accuracys)

    # save bandits with posterior distribution
    pickle.dump(bandits, open(save_name, 'wb'))


############################################################################################################################################################################
    if deeper_search:

        print('Deep search was entered: With the %s an accuracy of %s for the movement of %s was achieved' % (selected_bandit, aim_accuracy, aim))

        # Define new actionspace/bandits for deeper search
        combinations = neighbor_combinations(selected_bandit.electrode)
        new_bandits = [x for x in bandits if x.electrode in combinations and x.amplitude in [6, 8, 10]]
        iter = 0
        time_exceeded = False
        # Stay in deeper search for n_deeper steps
        while aim_accuracy <= 0.8:
            iter += 1
            print(iter)
            if iter == n_deeper:
                time_exceeded = True
                break

            new_action = pick_action(aim, new_bandits)
            selected_bandit = new_bandits[new_action]
            selected_bandit.define_velec()

            '''# Pause between stimulations
            time.sleep(1)
            # Stimulate predefined velecs and set event markers
            await connection.set_qtm_event(name)
            velec.stim(2)'''

            # Stream marker coordinates and calculate flexion
            #flexions = [flexion_ind1, flexion_ind2, flexion_mid1, flexion_mid2, flexion_thumb1, flexion_thumb2, roll,pitch, yaw]

            flexions = []
            for _ in range(9):
                flexions.append(random.sample(range(0, 120), 100))

            # Calculate rewards for each finger
            for finger in ['ind', 'mid', 'thumb']:
                reward, accuracy, relative_res = (calc_reward(finger, flexions, ideal_movements))
                rewards.append(reward)

                merged_accuracy = (accuracy[0] + accuracy[1]) / 2
                accuracys.append(merged_accuracy)

                if finger == aim:
                    aim_accuracy = merged_accuracy

            # Update each distribution for selected bandit
            selected_bandit.update_observation(rewards, accuracys)
            # save bandits with posterior distribution
            pickle.dump(bandits, open(save_name, 'wb'))

        if time_exceeded:

            print('Failure: No good bandit found in given time!')

        else:
            print('Success: Good bandit found! With the %s an accuracy of %s for the movement of %s was achieved' % (selected_bandit, aim_accuracy, aim))
            deeper_search = False
            if len(aim_options) != 0:
                aim_options.remove(aim)
            if len(aim_options) == 0:
                print('Finished! For each movement a good bandit has been found.')
                break












