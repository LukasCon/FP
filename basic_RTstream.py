import asyncio
import qtm
import pkg_resources
import pandas as pd
import numpy as np
from scipy import interpolate
import math
import matplotlib.pyplot as plt
import serial
import time
import pickle

QTM_FILE = pkg_resources.resource_filename("qtm", "data\Grasp1_20201210.qtm")

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

        realtime = True

        if realtime:
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

        else:
            # Load qtm file
            await connection.load(QTM_FILE)
            # start rtfromfile
            await connection.start(rtfromfile=True)


        framesOfPositions = []
        framesOfRotations = []
        # Labels need to be in the same order as in the AIM model used in QTM
        labels = ['wrist1', 'wrist2', 'wrist3', 'wrist4', 'ind1', 'ind2', 'ind3', 'mid1', 'mid2', 'mid3', 'thumb1', 'thumb2', 'thumb3']

        # on_packet defines what happens for every streamed frame
        def on_packet(packet):
            info, bodies = packet.get_6d_euler()
            position, rotation = bodies[0]
            header, markers = packet.get_3d_markers()

            print("Framenumber: {}".format(packet.framenumber))

            df_pos = pd.DataFrame(markers, index=labels)
            framesOfPositions.append(df_pos)
            df_rot = pd.DataFrame(rotation,columns= ["wrist"], index=['roll', 'pitch', 'yaw'])
            framesOfRotations.append(df_rot)
            # print(df_rot)

        # Start streaming frames
        await connection.stream_frames(components=["6deuler", "3d"], on_packet=on_packet)

        if realtime:
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

            class velec:
                def __init__(self, number, name='test'):
                    self.number = number

                    if number <= 4:
                        print(
                            "Using the first four velecs (1-4) can cause mal functions, thus it is recommended to define new velecs from 5 on.")
                    else:

                        self.name = name
                        self.cathodelist = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                        self.amp = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                        self.width = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                        self.anode = []
                        self.selected = 1

                def cathodes(self, activecathodes):
                    self.cathodelist = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    for x in activecathodes:
                        self.cathodelist[x - 1] = 1

                def amplitudes(self, amplitudes):
                    activecathodesindices = [i + 1 for i in range(len(self.cathodelist)) if self.cathodelist[i] == 1]
                    for x in activecathodesindices:
                        self.amp[x - 1] = amplitudes[activecathodesindices.index(x)]

                def widths(self, widths):
                    activecathodesindices = [i + 1 for i in range(len(self.cathodelist)) if self.cathodelist[i] == 1]
                    for x in activecathodesindices:
                        self.width[x - 1] = widths[activecathodesindices.index(x)]

                def anodes(self, activeanodes):
                    anodelist = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    activecathodes = [i + 1 for i in range(len(self.cathodelist)) if self.cathodelist[i] == 1]

                    def commonelements(x, y):
                        common = 0
                        for value in x:
                            if value in y:
                                common = 1
                        return common

                    if (activeanodes == [2] or activeanodes == [16] or activeanodes == [2, 16]) and commonelements(
                            activeanodes, activecathodes) == 0:
                        for x in activeanodes:
                            anodelist[x - 1] = 1
                        anodelist.reverse()
                        listToStr = ''.join([str(elem) for elem in anodelist])
                        self.anode = int(listToStr, 2)
                    else:
                        print('ERROR: Inadmissible choice of anodes')

                def activate(self):
                    self.selected = 1

                def deactivate(self):
                    self.selected = 0

                def define(self):
                    print(bytes(conv_to_message(self), 'utf-8'))
                    ser.write(bytes(conv_to_message(self), 'utf-8'))

                def stim(self, seconds=2):
                    message_start = "stim %s\r\n" % (self.name)
                    print(bytes(message_start, 'utf-8'))
                    ser.write(bytes(message_start, 'utf-8'))
                    time.sleep(seconds)
                    message_stop = "stim off\r\n"
                    print(bytes(message_stop, 'utf-8'))
                    ser.write(bytes(message_stop, 'utf-8'))

                def __repr__(self):
                    string = 'Name: %s\n' % self.name
                    string = string + 'cathodes: %s\n' % self.cathodelist
                    string = string + 'amp:      %s\n' % self.amp
                    string = string + 'width:    %s\n' % self.width
                    string = string + 'anode: %s\n' % self.anode
                    string = string + 'selected: %s\n' % self.selected

                    return string

            def conv_to_message(velec):
                message = "velec %s *name %s *elec 1 " \
                          "*cathodes 1=%s,2=%s,3=%s,4=%s,5=%s,6=%s,7=%s,8=%s,9=%s,10=%s,11=%s,12=%s,13=%s,14=%s,15=%s,16=%s, " \
                          "*amp 1=%s,2=%s,3=%s,4=%s,5=%s,6=%s,7=%s,8=%s,9=%s,10=%s,11=%s,12=%s,13=%s,14=%s,15=%s,16=%s, " \
                          "*width 1=%s,2=%s,3=%s,4=%s,5=%s,6=%s,7=%s,8=%s,9=%s,10=%s,11=%s,12=%s,13=%s,14=%s,15=%s,16=%s, " \
                          "*anode %s *selected %s *sync 0\r\n" \
                          % (velec.number, velec.name, velec.cathodelist[0], velec.cathodelist[1], velec.cathodelist[2],
                             velec.cathodelist[3],
                             velec.cathodelist[4], velec.cathodelist[5], velec.cathodelist[6], velec.cathodelist[7],
                             velec.cathodelist[8],
                             velec.cathodelist[9], velec.cathodelist[10], velec.cathodelist[11], velec.cathodelist[12],
                             velec.cathodelist[13],
                             velec.cathodelist[14], velec.cathodelist[15],
                             velec.amp[0], velec.amp[1], velec.amp[2], velec.amp[3], velec.amp[4], velec.amp[5],
                             velec.amp[6], velec.amp[7],
                             velec.amp[8], velec.amp[9], velec.amp[10], velec.amp[11], velec.amp[12], velec.amp[13],
                             velec.amp[14], velec.amp[15],
                             velec.width[0], velec.width[1], velec.width[2], velec.width[3], velec.width[4], velec.width[5],
                             velec.width[6], velec.width[7],
                             velec.width[8], velec.width[9], velec.width[10], velec.width[11], velec.width[12],
                             velec.width[13], velec.width[14], velec.width[15],
                             velec.anode, velec.selected)

                return message

            # Define virtual electrodes for twitch stimulation
            velec_order = [4, 6, 3, 1, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
            velecs = []
            for i in range(len(velec_order)):
                name = 'velec' + str(velec_order[i])
                velecs.append(velec(8, name)) #random velec number (here: 8) doesnt matter because redefinition for each velec
                velecs[i].cathodes([velec_order[i]])
                velecs[i].amplitudes([10])
                velecs[i].widths([300])
                velecs[i].anodes([2])
                # print(velecs[i])
                velecs[i].define()

                # Pause between stimulations
                time.sleep(0.5)
                # Stimulate predefined velecs and set event markers
                await connection.set_qtm_event(name)
                velecs[i].stim(1)

            '''velec10 = velec(11, 'close')
            velec10.cathodes([4, 6])
            velec10.amplitudes([10, 10])
            velec10.widths([300, 300])
            velec10.anodes([2])
            print(velec10)
            velec10.define()
            velec10.stim()'''

            # Delay needed, otherwise error from connection.stop
            await asyncio.sleep(5)

        else:
            # Length of measurement for non-realtime
            await asyncio.sleep(15)

        # Stop streaming
        await connection.stream_frames_stop()

        await connection.stop()


        def vector(time, point1, point2):
            vector = np.array([framesOfPositions[time].loc[point2, 'x'] - framesOfPositions[time].loc[point1, 'x'],
                               framesOfPositions[time].loc[point2, 'y'] - framesOfPositions[time].loc[point1, 'y'],
                               framesOfPositions[time].loc[point2, 'z'] - framesOfPositions[time].loc[point1, 'z']])
            return vector

        def normalize(x):
            return np.array([x[i] / np.linalg.norm(x) for i in range(len(x))])

        def plane(time, point1, point2, point3):

            p0, p1, p2 = [framesOfPositions[time].loc[point1, 'x':'z'], framesOfPositions[time].loc[point2, 'x':'z'],
                          framesOfPositions[time].loc[point3, 'x':'z']]
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

        def flexion_mcp(time, finger):
            flexion = np.zeros((len(time)))
            for i in time:
                r_norm = normalize(vector(i, finger + '1', finger + '2'))
                n_frontal = plane(i, 'wrist1', 'wrist2', 'wrist3')

                if finger == 'thumb':
                    r_aux = vector(i, 'mid1', 'mid2')
                    ux, uy, uz = r_aux
                    vx, vy, vz = n_frontal
                    n_sagittal = normalize([uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx])
                    n = n_sagittal
                else:
                    n = n_frontal

                flexion[i] = math.acos((np.dot(r_norm, n)) / (np.linalg.norm(r_norm) * np.linalg.norm(n))) - math.pi / 2
                flexion[i] = math.degrees(flexion[i])

            if np.isnan(flexion).any():

                return fill_nan(flexion)

            else:
                return flexion

        def flexion_pip(time, finger):
            flexion = np.zeros((len(time)))
            for i in time:
                r1 = normalize(vector(i, finger + '1', finger + '2'))
                r2 = normalize(vector(i, finger + '2', finger + '3'))

                flexion[i] = math.acos((np.dot(r1, r2)) / (np.linalg.norm(r1) * np.linalg.norm(r2)))

                if finger == 'thumb':
                    flexion[i] = math.degrees(flexion[i])
                else:
                    flexion[i] = math.degrees(flexion[i])

            if np.isnan(flexion).any():

                return fill_nan(flexion)

            else:
                return flexion

        def fill_nan(A):

            inds = np.arange(A.shape[0])
            good = np.where(np.isfinite(A))
            f = interpolate.interp1d(inds[good], A[good], kind='slinear', bounds_error=False)
            B = np.where(np.isfinite(A), A, f(inds))
            return B

        fulltime = len(framesOfPositions)
        flexion_ind1 = flexion_mcp(range(fulltime), 'ind')
        flexion_ind2 = flexion_pip(range(fulltime), 'ind')
        flexion_mid1 = flexion_mcp(range(fulltime), 'mid')
        flexion_mid2 = flexion_pip(range(fulltime), 'mid')
        flexion_thumb1 = flexion_mcp(range(fulltime), 'thumb')
        flexion_thumb2 = flexion_pip(range(fulltime), 'thumb')

        roll = np.zeros(len(framesOfRotations))
        pitch = np.zeros(len(framesOfRotations))
        yaw = np.zeros(len(framesOfRotations))
        for i in range(len(framesOfRotations)):
            roll[i] = framesOfRotations[i].loc['roll']
            pitch[i] = framesOfRotations[i].loc['pitch']
            yaw[i] = framesOfRotations[i].loc['yaw']

        flexions = [flexion_ind1, flexion_ind2, flexion_mid1, flexion_mid2, flexion_thumb1, flexion_thumb2, roll, pitch, yaw]

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

        with open('flexions_0121_2.pkl', 'wb') as f:
            pickle.dump(flexions, f)

if __name__ == "__main__":
    # Run our asynchronous function until complete
    asyncio.get_event_loop().run_until_complete(main())
