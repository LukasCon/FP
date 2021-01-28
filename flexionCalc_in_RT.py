import asyncio
import qtm
import pandas as pd
import numpy as np
from scipy import integrate
import math
import matplotlib.pyplot as plt
import time
import serial
import pickle
from bandits import Bandit
import random

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

def calc_reward(basedOn, flexions, ideal_flexions):

        if basedOn == 'ind':
            flexions = np.array(flexions[0:2])
            ref_flexion = np.array(ideal_flexions[0:2])
        elif basedOn == 'mid':
            flexions = np.array(flexions[2:4])
            ref_flexion = np.array(ideal_flexions[2:4])
        elif basedOn == 'thumb':
            flexions = np.array(flexions[4:6])
            ref_flexion = np.array(ideal_flexions[4:6])
        else:
            raise ValueError(
                'Select between ind, mid or thumb to determine which flexion to use for reward calculation')

        relative_res = []
        accuracy = []
        # Extract features and calculate rewards
        for m in range(len(flexions)):
            flexion_sequence = flexions[m]
            max_meas = max(flexion_sequence) - flexion_sequence[0]
            max_ideal = ref_flexion[m]
            res_max = abs(max_ideal - max_meas)

            '''# calculation of integrals
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
            accuracy.append(((1 - res_max_norm) + (1 - res_rel_int_norm))/2)'''

            res_max_norm = (res_max / abs(max_ideal))
            accuracy.append((1 - res_max_norm))

        if accuracy[0] >= 0.5 and accuracy[1] >= 0.5:  # weighting for mcp and pip flexion residual possible
            reward = 1
        else:
            reward = 0

        return reward, accuracy #, relative_res

def calc_undesired_mov (basedOn, flexions):

    if basedOn == 'ind':
        other_flexs = np.array(flexions[2:])

    elif basedOn == 'mid':
        other_flexs = np.delete(np.array(flexions), [2, 3], 0)

    elif basedOn == 'thumb':
        other_flexs = np.delete(np.array(flexions), [4, 5], 0)

    else:
        raise ValueError('Select between ind, mid or thumb')


    biggest_deviation = []
    # Extract features and calculate rewards
    for m in range(len(flexions)):
        flexion_sequence = flexions[m]
        initial_flex = flexion_sequence[0]
        max_flex = max(flexion_sequence)
        min_flex = min(flexion_sequence)

        biggest_deviation.append(max((max_flex - initial_flex), (initial_flex - min_flex)))

    undesired_mov = [(biggest_deviation[0] + biggest_deviation[1])/2,
                     (biggest_deviation[2] + biggest_deviation[3])/2,
                     (biggest_deviation[4] + biggest_deviation[5] + biggest_deviation[6])/3]

    return undesired_mov

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
last_experiment = 'bandits_0130.pkl'

# Overwrite posterior distributions from last experiment?
overwrite = False
new_file = 'bandits_0131_2.pkl'
###########################################################################################################################################################################################
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
n = 30
n_deeper = 10
aim_options = ['ind', 'mid', 'thumb']
start_bandits = [x for x in bandits if x.electrode in [4, 6, 3, 1, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16] and x.amplitude in [6, 8]]
active_bandits = []

###############################################################################################################################################################################################
async def main():
    # Delay to get in position for realtime measurement
    await asyncio.sleep(5)

    # Connect to qtm
    connection = await qtm.connect("127.0.0.1")

    # Connection failed?
    if connection is None:
        print("Failed to connect")
        return

    # Take control of qtm, context manager will automatically release control after scope end
    async with qtm.TakeControl(connection, "password"):

        # New measurement
        await connection.new()
        try:
            # Start capture
            await connection.await_event(qtm.QRTEvent.EventConnected, timeout=10)
            await connection.start()
            await connection.await_event(qtm.QRTEvent.EventCaptureStarted, timeout=10)
            print("Capture started")
        except:
            print("Failed to start new measurement")

        framesOfPositions = []
        framesOfRotations = []
        flexion_ind1 = []
        flexion_ind2 = []
        flexion_mid1 = []
        flexion_mid2 = []
        flexion_thumb1 = []
        flexion_thumb2 = []
        roll = []
        pitch = []
        yaw = []

        # Labels need to be in the same order as in the AIM model used in QTM
        labels = ['wrist1', 'wrist2', 'wrist3', 'wrist4', 'ind1', 'ind2', 'ind3', 'mid1', 'mid2', 'mid3', 'thumb1', 'thumb2', 'thumb3']

        ###################################################################################################################################################################################
        # on_packet defines what happens for every streamed frame
        def on_packet(packet):
            info, bodies = packet.get_6d_euler()
            position, rotation = bodies[0]
            header, markers = packet.get_3d_markers()

            # print("Framenumber: {}".format(packet.framenumber))

            df_pos = pd.DataFrame(markers, index=labels)
            framesOfPositions.append(df_pos)
            df_rot = pd.DataFrame(rotation,columns= ["wrist"], index=['roll', 'pitch', 'yaw'])
            framesOfRotations.append(df_rot)

            roll.append(rotation[0])
            pitch.append(rotation[1])
            yaw.append(rotation[2])

            def vector(point1, point2):
                vector = np.array([df_pos.loc[point2, 'x'] - df_pos.loc[point1, 'x'],
                                   df_pos.loc[point2, 'y'] - df_pos.loc[point1, 'y'],
                                   df_pos.loc[point2, 'z'] - df_pos.loc[point1, 'z']])
                return vector

            def normalize(x):
                return np.array([x[i] / np.linalg.norm(x) for i in range(len(x))])

            def plane(point1, point2, point3):

                p0, p1, p2 = [df_pos.loc[point1, 'x':'z'],
                              df_pos.loc[point2, 'x':'z'],
                              df_pos.loc[point3, 'x':'z']]
                x0, y0, z0 = p0
                x1, y1, z1 = p1
                x2, y2, z2 = p2

                ux, uy, uz = u = [x1 - x0, y1 - y0, z1 - z0]
                vx, vy, vz = v = [x2 - x0, y2 - y0, z2 - z0]
                u_cross_v = [uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx]

                # point = np.array(p0)
                normal = np.array(u_cross_v)
                normal = normalize(normal)
                # d = -point.dot(normal)

                return normal

            def flexion_mcp(finger):
                r_norm = normalize(vector(finger + '1', finger + '2'))
                n_frontal = plane('wrist1', 'wrist2', 'wrist3')

                if finger == 'thumb':
                    r_aux = vector('mid1', 'mid2')
                    ux, uy, uz = r_aux
                    vx, vy, vz = n_frontal
                    n_sagittal = normalize([uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx])
                    n = n_sagittal
                else:
                    n = n_frontal

                flexion = math.acos(
                    (np.dot(r_norm, n)) / (np.linalg.norm(r_norm) * np.linalg.norm(n))) - math.pi / 2
                flexion = math.degrees(flexion)
                return flexion

            def flexion_pip(finger):

                r1 = normalize(vector(finger + '1', finger + '2'))
                r2 = normalize(vector(finger + '2', finger + '3'))

                flexion = math.acos((np.dot(r1, r2)) / (np.linalg.norm(r1) * np.linalg.norm(r2)))

                if finger == 'thumb':
                    flexion = math.degrees(flexion)
                else:
                    flexion = math.degrees(flexion)

                return flexion

            flexion_ind1.append(flexion_mcp('ind'))
            flexion_ind2.append(flexion_pip('ind'))
            flexion_mid1.append(flexion_mcp('mid'))
            flexion_mid2.append(flexion_pip('mid'))
            flexion_thumb1.append(flexion_mcp('thumb'))
            flexion_thumb2.append(flexion_pip('thumb'))


        # Start streaming frames
        await connection.stream_frames(components=["6deuler", "3d"], on_packet=on_packet)

        await asyncio.sleep(8)

        flexions = [flexion_ind1, flexion_ind2,
                    flexion_mid1, flexion_mid2,
                    flexion_thumb1, flexion_thumb2,
                    roll, pitch, yaw]

        def init_ideal_mov(flexions):
            ideal_flexions = []
            for i in range(len(flexions)):
                first_not_Nan = next(item for item in flexions[i] if math.isnan(item) == False)
                ideal_flexions.append(np.nanmax(np.array(flexions[i])) - first_not_Nan)
            return ideal_flexions

        ideal_flexions = init_ideal_mov(flexions)
        print(ideal_flexions)
        ####################################################################################################################################################################################
        # Open serial port
        ser = serial.Serial()
        ser.baudrate = 115200
        ser.port = 'COM4'
        ser.timeout = 1
        ser.open()
        print(ser)

        ser.write(b"iam DESKTOP\r\n")
        ser.write(b"elec 1 *pads_qty 16\r\n")
        ser.write(b"freq 35\r\n")

        #####################################################################################################################################################################################
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
            velec = selected_bandit.define_velec(ser)

            before_stim = len(framesOfPositions)
            print('before', before_stim)

            # Pause between stimulations
            await asyncio.sleep(0.5)
            # Stimulate predefined velecs and set event markers
            await connection.set_qtm_event(velec.name)
            velec.stim(1)

            after_stim = len(framesOfPositions)
            print('after', after_stim)

            # Stream marker coordinates and calculate flexion
            flexions = [flexion_ind1[before_stim:after_stim], flexion_ind2[before_stim:after_stim],
                        flexion_mid1[before_stim:after_stim], flexion_mid2[before_stim:after_stim],
                        flexion_thumb1[before_stim:after_stim], flexion_thumb2[before_stim:after_stim],
                        roll[before_stim:after_stim], pitch[before_stim:after_stim], yaw[before_stim:after_stim]]

            # Calculate rewards for each finger
            rewards = []
            accuracys = []
            undesired_movs = []
            for finger in ['ind', 'mid', 'thumb']:
                reward, accuracy = (calc_reward(finger, flexions, ideal_flexions))
                rewards.append(reward)

                merged_accuracy = (accuracy[0] + accuracy[1]) / 2
                accuracys.append(merged_accuracy)

                undesired_mov = calc_undesired_mov(finger, flexions)
                undesired_movs.append(undesired_mov)
                undesired_wrist_mov = undesired_mov[2]

                # Check if observation is good enough for deeper search
                if merged_accuracy >= 0.5 and undesired_wrist_mov < (20/3) and finger in aim_options:
                    deeper_search = True
                    aim = finger
                    # Initial aim_accuracy
                    aim_accuracy = merged_accuracy

            # Update each distribution for selected bandit
            selected_bandit.update_observation(rewards, accuracys, undesired_movs)
            print(selected_bandit)
            active_bandits.append([selected_bandit.electrode, selected_bandit.amplitude])
            # save bandits with posterior distribution
            pickle.dump(bandits, open(save_name, 'wb'))

            ################################################################################################################################################################################
            if deeper_search:

                print('Deep search was entered: With the %s an accuracy of %s for the movement of %s was achieved' % (selected_bandit, aim_accuracy, aim))

                # Define new actionspace/bandits for deeper search
                combinations = neighbor_combinations(selected_bandit.electrode)
                new_bandits = [x for x in bandits if x.electrode in combinations and x.amplitude in [8, 10, 12]]
                iter = 0
                time_exceeded = False
                # Stay in deeper search for n_deeper steps
                while aim_accuracy <= 0.8:
                    iter += 1
                    print('iteration', iter)
                    if iter == n_deeper:
                        time_exceeded = True
                        break

                    new_action = pick_action(aim, new_bandits)
                    selected_bandit = new_bandits[new_action]
                    velec = selected_bandit.define_velec(ser)

                    before_stim = len(framesOfPositions)
                    print('before', before_stim)

                    # Pause between stimulations
                    await asyncio.sleep(0.5)
                    # Stimulate predefined velecs and set event markers
                    #await connection.set_qtm_event(velec.name)
                    velec.stim(1)

                    after_stim = len(framesOfPositions)
                    print('after', after_stim)

                    '''flexions = []
                    for _ in range(9):
                        flexions.append(random.sample(range(0, 120), 100))'''

                    # Stream marker coordinates and calculate flexion
                    flexions = [flexion_ind1[before_stim:after_stim], flexion_ind2[before_stim:after_stim],
                                flexion_mid1[before_stim:after_stim], flexion_mid2[before_stim:after_stim],
                                flexion_thumb1[before_stim:after_stim], flexion_thumb2[before_stim:after_stim],
                                roll[before_stim:after_stim], pitch[before_stim:after_stim],
                                yaw[before_stim:after_stim]]

                    # Calculate rewards for each finger
                    for finger in ['ind', 'mid', 'thumb']:
                        reward, accuracy = (calc_reward(finger, flexions, ideal_flexions))
                        rewards.append(reward)

                        merged_accuracy = (accuracy[0] + accuracy[1]) / 2
                        accuracys.append(merged_accuracy)

                        if finger == aim:
                            aim_accuracy = merged_accuracy

                    # Update each distribution for selected bandit
                    selected_bandit.update_observation(rewards, accuracys, undesired_movs)
                    print(selected_bandit)
                    active_bandits.append([selected_bandit.electrode, selected_bandit.amplitude])
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

        pickle.dump(active_bandits, open(('active' + save_name), 'wb'))

        ser.close()
        ###################################################################################################################################################################################
        # Delay needed, otherwise error from connection.stop
        await asyncio.sleep(10)

        # Stop streaming
        await connection.stream_frames_stop()

        await connection.stop()

        fulltime = len(framesOfPositions)
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharey= True)
        ax1.plot(range(fulltime), flexion_ind1, label=('mcp flex ind'))
        ax1.plot(range(fulltime), flexion_ind2, label=('pip flex ind'))
        ax1.set(xlabel='', ylabel='ind flexion (°)')
        ax1.legend()
        ax1.grid()
        ax2.plot(range(fulltime), flexion_mid1, label=('mcp flex mid'))
        ax2.plot(range(fulltime), flexion_mid2, label=('pip flex mid'))
        ax2.set(xlabel='', ylabel='mid flexion (°)')
        ax2.legend()
        ax2.grid()
        ax3.plot(range(fulltime), flexion_thumb1, label=('mcp flex thumb'))
        ax3.plot(range(fulltime), flexion_thumb2, label=('pip flex thumb'))
        ax3.set(xlabel='', ylabel='thumb flexion (°)')
        ax3.legend()
        ax3.grid()
        ax4.plot(range(fulltime), roll[:], label=('roll'))
        ax4.plot(range(fulltime), pitch[:], label=('pitch'))
        ax4.plot(range(fulltime), yaw[:], label=('yaw'))
        ax4.set(xlabel='frames', ylabel='rpy angles(°)')
        ax4.legend()
        ax4.grid()
        plt.show()




if __name__ == "__main__":
    # Run our asynchronous function until complete
    asyncio.get_event_loop().run_until_complete(main())






