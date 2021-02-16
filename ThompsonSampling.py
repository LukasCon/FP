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

# Name the measurement
new_file = 'bandits_0218_5.pkl'

# Use uniform priors or posterior distribution from last experiment?
use_uniform_priors = False
last_experiment = 'bandits_0217_3.pkl'

# Capture new ideal movements before the measurement or use predefined ideal flexions?
capture_new_ideal_mov = False
old_ideal_flexions = [35.946132810686834, 60.51440073878825,                        # ind_mcp, ind_pip
                      44.35013573887927, 59.451238560892094,                        # mid_mcp, mid_pip
                      28.543467439621736, 38.741753732943295,                       # thumb_mcp, thumb_pip
                      1.9215731620788574, 1.9310812950134277, 1.0939593315124512]   # roll, pitch, yaw

# Initialize parameters
n = 30                          # max number of stimulations in 'normal' search
n_deeper = 8                    # max number of stimulations in deep search
pause_between_ds = 3            # min number of stimulations between two deep searches of the same finger
max_numb_of_ds = 2              # max number of deep searches for the same finger
aim_options = ['ind', 'mid', 'thumb']

def vector(point1, point2, df_pos):
    vector = np.array([df_pos.loc[point2, 'x'] - df_pos.loc[point1, 'x'],
                       df_pos.loc[point2, 'y'] - df_pos.loc[point1, 'y'],
                       df_pos.loc[point2, 'z'] - df_pos.loc[point1, 'z']])
    return vector

def normalize(x):
    return np.array([x[i] / np.linalg.norm(x) for i in range(len(x))])

def plane(point1, point2, point3, df_pos):
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

def flexion_mcp(finger, df_pos):
    r_norm = normalize(vector(finger + '1', finger + '2', df_pos))
    n_frontal = plane('wrist1', 'wrist2', 'wrist3', df_pos)

    if finger == 'thumb':
        r_aux = vector('mid1', 'mid2', df_pos)
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

def flexion_pip(finger, df_pos):
    r1 = normalize(vector(finger + '1', finger + '2', df_pos))
    r2 = normalize(vector(finger + '2', finger + '3', df_pos))

    flexion = math.acos((np.dot(r1, r2)) / (np.linalg.norm(r1) * np.linalg.norm(r2)))

    if finger == 'thumb':
        flexion = math.degrees(flexion)
    else:
        flexion = math.degrees(flexion)

    return flexion

def init_ideal_mov(flexions):

    ideal_flexions = []
    for i in range(len(flexions)):
        first_not_Nan = next(item for item in flexions[i] if math.isnan(item) == False)
        ideal_flexions.append((np.nanmax(np.array(flexions[i]))) - first_not_Nan)
    return ideal_flexions

def pick_action(basedOn, bandits):

    if basedOn == 'ind':
        k = 0
    elif basedOn == 'mid':
        k = 1
    elif basedOn == 'thumb':
        k = 2
    else:
        raise ValueError('Select between ind, mid or thumb to determine from which posterior distribution the next action should be picked')

    # Take random samples of posterior distributions
    sampled_means = []
    for i in range(len(bandits)):
        sampled_means.append(bandits[i].get_posterior_sample()[k])

    def random_argmax(vector):
        #Helper function to select argmax at random... not just first one
        index = np.random.choice(np.where(vector == vector.max())[0])
        return index

    action = random_argmax(np.array(sampled_means))
    return action

def calc_reward(basedOn, flexions, ideal_flexions):

        if basedOn == 'ind':
            flexions = np.array(flexions[0:2])
            max_ideal = np.array(ideal_flexions[0:2])
        elif basedOn == 'mid':
            flexions = np.array(flexions[2:4])
            max_ideal = np.array(ideal_flexions[2:4])
        elif basedOn == 'thumb':
            flexions = np.array(flexions[4:6])
            max_ideal = np.array(ideal_flexions[4:6])
        else:
            raise ValueError(
                'Select between ind, mid or thumb to determine which flexion to use for reward calculation')

        accuracy = []
        # Extract max values and calculate rewards
        for m in range(len(flexions)):
            # Measured max flexion in relation to the initial flexion of the sequence
            max_meas = max(flexions[m]) - flexions[m][0]
            # Error between max values
            err_max = abs(max_ideal[m] - max_meas)
            # Normalized error
            err_max_norm = (err_max / abs(max_ideal[m]))

            if err_max_norm > 1:
                err_max_norm = 1
                print('ATTENTION! The measurement yields a normalized error greater than 1. \n'
                      'This would lead to an negative accuracy, thus the normalized error is considered 1 in the following calculation.\n'
                      'HINT: You may check the maximal flexion in your ideal movements.')

            # Percentage accuracy
            accuracy.append((1 - err_max_norm))

        return accuracy

def calc_undesired_mov (basedOn, flexions):

    # other_flexs are the flexions of the other fingers and the wrist
    if basedOn == 'ind':
        other_flexs = np.array(flexions[2:])
    elif basedOn == 'mid':
        other_flexs = np.delete(np.array(flexions), [2, 3], 0)
    elif basedOn == 'thumb':
        other_flexs = np.delete(np.array(flexions), [4, 5], 0)
    else:
        raise ValueError('Select between ind, mid or thumb')

    # Extract max and min values and calculate deviations
    biggest_deviation = []
    for m in range(len(other_flexs)):
        flexion_sequence = other_flexs[m]
        initial_flex = flexion_sequence[0]
        max_flex = max(flexion_sequence)
        min_flex = min(flexion_sequence)

        biggest_deviation.append(max((max_flex - initial_flex), (initial_flex - min_flex)))

    undesired_mov = [(biggest_deviation[0] + biggest_deviation[1])/2,
                     (biggest_deviation[2] + biggest_deviation[3])/2,
                     (biggest_deviation[4] + biggest_deviation[5] + biggest_deviation[6])/3]

    return undesired_mov

def neighbor_combinations(elec_number):

    # Matrix represents the order of electrodes in the array
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

########################################################################################################################
if use_uniform_priors:
    # Define all bandits/ whole action space
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

    # Reset best accuracys and undesired movements
    for i in range(len(bandits)):
        bandits[i].best_accuracy = [0, 0, 0]
        bandits[i].undesired_mov = [[0, 0, 0],
                                    [0, 0, 0],
                                    [0, 0, 0]]


# Define bandits/actionspace for 'normal' search
start_bandits = [x for x in bandits if x.electrode in [4, 6, 3, 1, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16] and x.amplitude in [8, 10]]

########################################################################################################################
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

        ################################################################################################################
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

            flexion_ind1.append(flexion_mcp('ind', df_pos))
            flexion_ind2.append(flexion_pip('ind', df_pos))
            flexion_mid1.append(flexion_mcp('mid', df_pos))
            flexion_mid2.append(flexion_pip('mid', df_pos))
            flexion_thumb1.append(flexion_mcp('thumb', df_pos))
            flexion_thumb2.append(flexion_pip('thumb', df_pos))


        # Start streaming frames
        # Frequency is reduced to 10Hz
        await connection.stream_frames(frames='frequency:10', components=["6deuler", "3d"], on_packet=on_packet)

        # Define ideal felxions
        if capture_new_ideal_mov:
            # Time to perform ideal movements
            await asyncio.sleep(8)
            # Initialize ideal movements
            flexions = [flexion_ind1, flexion_ind2, flexion_mid1, flexion_mid2, flexion_thumb1, flexion_thumb2, roll, pitch, yaw]
            ideal_flexions = init_ideal_mov(flexions)
        else:
            ideal_flexions = old_ideal_flexions
        print(ideal_flexions)

        ################################################################################################################
        # Open serial port
        ser = serial.Serial()
        ser.baudrate = 115200
        ser.port = 'COM4'
        ser.timeout = 1
        ser.open()
        print(ser)

        # Initial commands for FES device
        ser.write(b"iam DESKTOP\r\n")
        ser.write(b"elec 1 *pads_qty 16\r\n")
        ser.write(b"freq 35\r\n")

        ################################################################################################################
        await asyncio.sleep(0.5)

        # Archive for deep searches
        deep_searches = pd.DataFrame(data={'time': [], 'finger': []})
        # Initialize history of selected bandits
        active_bandits = []
        # veritcal lines to visualize the stimulation sequences
        vlines = []

        # Start 'normal' search
        for t in range(n):
            deeper_search = False
            print('\nt:', t)
            # aim defines which finger/ posterior distribution is used to pick the following action
            aim = aim_options[0]
            print(aim_options)
            print(aim)

            # Choose action based on maximum of success probability which comes from random sample of the posterior distributions of the bandits
            action = pick_action(aim, start_bandits)
            selected_bandit = start_bandits[action]
            # Define velec corresponding to selected bandit
            velec = selected_bandit.define_velec(ser)

            before_stim = len(framesOfPositions)

            # Pause between stimulations
            await asyncio.sleep(0.5)
            # Stimulate predefined velecs and set event markers
            await connection.set_qtm_event(velec.name)
            velec.stim_on()
            # Stimulation time
            await asyncio.sleep(1)
            velec.stim_off()
            await asyncio.sleep(0.5)

            after_stim = len(framesOfPositions)
            vlines.append(after_stim)
            print('Sequence:', before_stim, '-', after_stim)


            # Get flexions for the recent stimulation section
            flexions = [flexion_ind1[before_stim:after_stim], flexion_ind2[before_stim:after_stim],
                        flexion_mid1[before_stim:after_stim], flexion_mid2[before_stim:after_stim],
                        flexion_thumb1[before_stim:after_stim], flexion_thumb2[before_stim:after_stim],
                        roll[before_stim:after_stim], pitch[before_stim:after_stim], yaw[before_stim:after_stim]]

            # Calculate rewards/accuracys for each finger
            accuracys = []
            undesired_movs = []
            for finger in ['ind', 'mid', 'thumb']:
                accuracy = (calc_reward(finger, flexions, ideal_flexions))

                # Take mean of PIP and MCP joint accuracy
                merged_accuracy = (accuracy[0] + accuracy[1]) / 2
                accuracys.append(merged_accuracy)

                # Undesired movements are the flexions of the other fingers and the wrist
                # especially the wrist flexion is not wanted and therefore the deep search is inhibit if the sum of wrist flexion is > 20degree
                undesired_mov = calc_undesired_mov(finger, flexions)
                undesired_movs.append(undesired_mov)
                undesired_wrist_mov = undesired_mov[2]

                # Check if observation is good enough for deeper search
                if merged_accuracy >= 0.5 and undesired_wrist_mov < (20/3) and finger in aim_options:

                    # Check if deep search for this finger was applied recently without success
                    previous_ds = deep_searches[deep_searches['finger'] == finger]
                    if previous_ds.empty:

                        deeper_search = True
                        # Initial aim_accuracy
                        aim = finger
                        aim_accuracy = merged_accuracy

                    else:
                        numb_of_prev_ds = previous_ds.shape[0]
                        time_since_last_ds = t - int(previous_ds.tail(1)['time'])
                        # too recent deepsearch for the same finger or too many unsuccessful deepsearch attempts inhibit new deepsearch due to impending fatigue
                        if time_since_last_ds > pause_between_ds and numb_of_prev_ds < max_numb_of_ds:

                            deeper_search = True
                            # Initial aim_accuracy
                            aim = finger
                            aim_accuracy = merged_accuracy

        # Update each distribution for selected bandit
            selected_bandit.update_observation(accuracys, undesired_movs)
            print(selected_bandit)
            print('Accuracys:', accuracys)
            print('Wrist movement:', undesired_wrist_mov)
            active_bandits.append([selected_bandit.electrode, selected_bandit.amplitude, accuracys, undesired_movs])
            # save bandits with posterior distribution
            pickle.dump(bandits, open(new_file, 'wb'))

            ############################################################################################################
            if deeper_search:
                print('Deep search was entered: With the %s an accuracy of %s for the movement of %s was achieved' % (selected_bandit, aim_accuracy, aim))

                # Add deep search to deep search archive
                deep_searches = deep_searches.append({'time': t, 'finger': aim}, ignore_index= True)

                # Define new actionspace/bandits for deeper search
                combinations = neighbor_combinations(selected_bandit.electrode)
                new_bandits = [x for x in bandits if x.electrode in combinations and x.amplitude in [8, 10]]

                # Initialize iterator for deeper search
                iter = 0
                time_exceeded = False
                # Stay in deeper search for n_deeper steps
                while aim_accuracy <= 0.75:
                    iter += 1
                    print('iteration', iter)
                    if iter == n_deeper:
                        time_exceeded = True
                        break

                    # Pick action based on random sample of posterior distribution
                    new_action = pick_action(aim, new_bandits)
                    selected_bandit = new_bandits[new_action]
                    # Define velec
                    velec = selected_bandit.define_velec(ser)

                    before_stim = len(framesOfPositions)

                    # Pause between stimulations
                    await asyncio.sleep(0.5)
                    # Stimulate predefined velecs
                    velec.stim_on()
                    # Stimulation time
                    await asyncio.sleep(1)
                    velec.stim_off()
                    await asyncio.sleep(0.5)

                    after_stim = len(framesOfPositions)
                    vlines.append(after_stim)
                    print('Sequence:', before_stim, '-', after_stim)

                    # Get flexions for the recent stimulation section
                    flexions = [flexion_ind1[before_stim:after_stim], flexion_ind2[before_stim:after_stim],
                                flexion_mid1[before_stim:after_stim], flexion_mid2[before_stim:after_stim],
                                flexion_thumb1[before_stim:after_stim], flexion_thumb2[before_stim:after_stim],
                                roll[before_stim:after_stim], pitch[before_stim:after_stim],
                                yaw[before_stim:after_stim]]

                    # Calculate rewards/accuracys for each finger
                    accuracys = []
                    undesired_movs = []
                    for finger in ['ind', 'mid', 'thumb']:
                        accuracy = (calc_reward(finger, flexions, ideal_flexions))

                        # Take mean of PIP and MCP joint accuracy
                        merged_accuracy = (accuracy[0] + accuracy[1]) / 2
                        accuracys.append(merged_accuracy)

                        # Undesired movements are the flexions of the other fingers and the wrist
                        undesired_mov = calc_undesired_mov(finger, flexions)
                        undesired_movs.append(undesired_mov)

                        # Check if other finger (than aim) achieved the desired accuracy
                        if merged_accuracy >= 0.75 and finger != aim and finger in aim_options:
                            print('Success: Good bandit found! With the %s an accuracy of %s for the movement of %s was achieved' % (selected_bandit, merged_accuracy, finger))
                            aim_options.remove(finger)

                        if finger == aim:
                            aim_accuracy = merged_accuracy

                    # Update each distribution for selected bandit
                    selected_bandit.update_observation(accuracys, undesired_movs)
                    print(selected_bandit)
                    print('Accuracys:', accuracys)
                    print('Wrist movement:', undesired_mov[2])

                    active_bandits.append([selected_bandit.electrode, selected_bandit.amplitude, accuracys, undesired_movs])
                    # save bandits with posterior distribution
                    pickle.dump(bandits, open(new_file, 'wb'))

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

        # Save list of active bandits for eventual trace back of stimulation order
        pickle.dump(active_bandits, open(('active' + new_file), 'wb'))

        # Close serial port
        ser.close()

        ################################################################################################################
        # Delay needed, otherwise error from connection.stop
        await asyncio.sleep(10)

        # Stop streaming
        await connection.stream_frames_stop()

        await connection.stop()

        # Save flexions to ease a new plot
        with open(('flexions' + new_file), 'wb') as f:
            pickle.dump(flexions, f)

        # Plot the flexions for the whole measurement
        fulltime = len(framesOfPositions)
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharey= True)
        ax1.plot(range(fulltime), flexion_ind1, label=('mcp flex ind'))
        ax1.plot(range(fulltime), flexion_ind2, label=('pip flex ind'))
        ax1.set(xlabel='', ylabel='ind flexion (°)')
        ax1.legend()
        ax1.grid()
        ax1.vlines(vlines, 0, 75)
        ax2.plot(range(fulltime), flexion_mid1, label=('mcp flex mid'))
        ax2.plot(range(fulltime), flexion_mid2, label=('pip flex mid'))
        ax2.set(xlabel='', ylabel='mid flexion (°)')
        ax2.legend()
        ax2.grid()
        ax2.vlines(vlines, 0, 75)
        ax3.plot(range(fulltime), flexion_thumb1, label=('mcp flex thumb'))
        ax3.plot(range(fulltime), flexion_thumb2, label=('pip flex thumb'))
        ax3.set(xlabel='', ylabel='thumb flexion (°)')
        ax3.legend()
        ax3.grid()
        ax3.vlines(vlines, -20, 60)
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






