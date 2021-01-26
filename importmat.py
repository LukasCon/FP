from scipy.io import loadmat
from scipy import interpolate
import os
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import re
from scipy import integrate

measurements = [25]#,26,28,30,32,34,38,40,42,46,48,50,52,54,56]
for number in measurements:

    filename = "Measurement" + str(number)
    print(filename)

    username = os.environ["USERPROFILE"]
    filepath = os.path.join("/Users", username, "Desktop","Messungen", filename)

    def importmat(filename):
        #filepath
        username = os.environ["USERPROFILE"]
        filepath = os.path.join("/Users", username, "Desktop", "Messungen", filename)

        data = loadmat(filepath)
        name = filename
        frames = int(data[name][0][0]['Frames'])
        numbOfLabels = len(data[name][0][0]['Trajectories']['Labeled'][0][0]['Labels'][0][0][0][:])
        rigidBody = 'wristL'
        indexRB = list(data[name][0][0]['RigidBodies']['Name'][0][0][0][:]).index(rigidBody)
        coordinates = np.zeros((numbOfLabels,4,frames))
        positions = np.zeros((3,frames))
        rotations = np.zeros((9,frames))
        rpys = np.zeros((3,frames))
        residuals = np.zeros((frames))
        listOfDf = []
        labels = []
        for j in range(numbOfLabels):
            labels.append(data[name][0][0]['Trajectories']['Labeled'][0][0]['Labels'][0][0][0][j][0])

        for i in range(frames):
            coordinates[:,:,i] = (data[name][0][0]['Trajectories']['Labeled'][0][0]['Data'][0][0][:,:,i])
            df = pd.DataFrame(coordinates[:,:,i], columns=['x','y','z','res'], index=labels)
            listOfDf.append(df)

            positions[:,i] = (data[name][0][0]['RigidBodies']['Positions'][0][0][indexRB,:,i])
            rotations[:,i] = (data[name][0][0]['RigidBodies']['Rotations'][0][0][indexRB,:,i])
            rpys[:,i] = (data[name][0][0]['RigidBodies']['RPYs'][0][0][indexRB,:,i])
            residuals[i] = (data[name][0][0]['RigidBodies']['Residual'][0][0][indexRB,:,i])

        numbOfEventLabels = len(data[name][0][0]['Events'])
        events_labels = []
        events_frames = []
        for k in range(numbOfEventLabels):
            events_labels.append(data[name][0][0]['Events'][k][0][0][0])
            events_frames.append(data[name][0][0]['Events'][k][0][1][0][0])

        return coordinates, listOfDf, positions, rotations, rpys, residuals, events_labels, events_frames

    coordinates, listOfDf, positions, rotations, rpys, residuals, events_labels, events_frames = importmat(filename)

    events = [events_labels, events_frames]

    #print(listOfDf[0].loc[:,:])
    #print(coordinates[0,:,0])
    #print(rotations[:,4])

    def vector(time, point1, point2):
        vector = np.array([listOfDf[time].loc[point2, 'x'] - listOfDf[time].loc[point1, 'x'],
                  listOfDf[time].loc[point2, 'y'] - listOfDf[time].loc[point1, 'y'],
                  listOfDf[time].loc[point2, 'z'] - listOfDf[time].loc[point1, 'z']])
        return vector

    def normalize(x):
        return np.array([x[i]/np.linalg.norm(x) for i in range(len(x))])

    def project_onto_plane(x, n):
        d = np.dot(x, n)/np.linalg.norm(n)
        p = [d * normalize(n)[i] for i in range(len(n))]
        return [x[i] - p[i] for i in range(len(x))]

    def plane(time, point1, point2, point3):

        p0, p1, p2 = [listOfDf[time].loc[point1,'x':'z'], listOfDf[time].loc[point2,'x':'z'], listOfDf[time].loc[point3,'x':'z']]
        x0, y0, z0 = p0
        x1, y1, z1 = p1
        x2, y2, z2 = p2

        ux, uy, uz = u = [x1-x0, y1-y0, z1-z0]
        vx, vy, vz = v = [x2-x0, y2-y0, z2-z0]
        u_cross_v = [uy*vz-uz*vy, uz*vx-ux*vz, ux*vy-uy*vx]

        #point = np.array(p0)
        normal = np.array(u_cross_v)
        normal = normalize(normal)
        #d = -point.dot(normal)

        return normal

    def flexion_mcp(time, finger):
        flexion = np.zeros((len(time)))
        for i in time:
            r_norm = normalize(vector(i, finger + '1', finger + '2'))
            n_frontal = plane(i,'wrist1','wrist2','wrist3')

            if finger == 'thumb':
                r_aux = vector(i, 'mid1', 'mid2')
                ux, uy, uz = r_aux
                vx, vy, vz = n_frontal
                n_sagittal = normalize([uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx])
                n = n_sagittal
            else:
                n = n_frontal

            flexion[i] = math.acos((np.dot(r_norm,n))/(np.linalg.norm(r_norm)*np.linalg.norm(n))) - math.pi/2
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

            flexion[i] = math.acos((np.dot(r1,r2))/(np.linalg.norm(r1)*np.linalg.norm(r2)))

            if finger == 'thumb':
                flexion[i] = math.degrees(flexion[i])
            else:
                flexion[i] = math.degrees(flexion[i])

        if np.isnan(flexion).any():

            return fill_nan(flexion)

        else:
            return flexion

    def fill_nan(A):
        '''
        interpolate to fill nan values
        '''
        inds = np.arange(A.shape[0])
        good = np.where(np.isfinite(A))
        f = interpolate.interp1d(inds[good], A[good], kind = 'slinear', bounds_error=False)
        B = np.where(np.isfinite(A),A,f(inds))
        return B

    def quickplot(time,flexion):
        fig, ax = plt.subplots()
        for i in range(len(flexion)):
            ax.plot(time, flexion[i], label = ('flexion %s' %(i+1)))

        ax.set(xlabel='time (s)', ylabel='flexion (°)')
        ax.grid()
        ax.legend()
        plt.show()

    fulltime = coordinates.shape[2]

    flexion_ind1 = flexion_mcp(range(fulltime),'ind')
    flexion_ind2 = flexion_pip(range(fulltime),'ind')
    flexion_mid1 = flexion_mcp(range(fulltime),'mid')
    flexion_mid2 = flexion_pip(range(fulltime),'mid')
    flexion_thumb1 = flexion_mcp(range(fulltime),'thumb')
    flexion_thumb2 = flexion_pip(range(fulltime),'thumb')

    #quickplot(range(fulltime),[flexion_thumb1, flexion_thumb2])

    roll, pitch, yaw = rpys
    #quickplot(range(fulltime),[roll, pitch, yaw])

    flexions = [flexion_ind1, flexion_ind2, flexion_mid1, flexion_mid2, flexion_thumb1, flexion_thumb2, roll, pitch, yaw]

    # Create feature vector for each sequence with max, min, and integral residual
    #last_event_frame = int(events_frames[0])
    all_vectors = []
    for l in range(len(events_labels)):
        vector = [int(re.search(r'\d+', events_labels[l])[0])]
        for m in range(len(flexions)):
            if l == len(events_labels)-1:
                flexion_sequence = flexions[m][int(events_frames[l]):]
            else:
                flexion_sequence = flexions[m][int(events_frames[l]):int(events_frames[l+1])]

            vector.append(max(flexion_sequence))
            vector.append(min(flexion_sequence))

            # calculation of integrals
            integral = integrate.simps(flexion_sequence)    # full integral above x-axis
            x = np.arange(flexion_sequence.shape[0])
            y = np.full_like(x, flexion_sequence[0])
            rel_integral = integral - integrate.simps(y, x) # relative integral regarding the initial flexion value of the current section

            print(integral)
            print(rel_integral)


        #last_event_frame = int(events_frames[l])
        all_vectors.append(vector)

    sample_vec = pd.DataFrame(np.array(all_vectors), columns=['electrode', 'flex_ind1_max', 'flex_ind1_min', 'flex_ind2_max', 'flex_ind2_min',
                                               'flex_mid1_max', 'flex_mid1_min', 'flex_mid2_max', 'flex_mid2_min', 'flex_thumb1_max',
                                               'flex_thumb1_min', 'flex_thumb2_max', 'flex_thumb2_min', 'roll_max', 'roll_min',
                                               'pitch_max', 'pitch_min', 'yaw_max', 'yaw_min'])
    print(sample_vec)

    #sample_vec.to_pickle(filename + '.pkl')


'''measurements = [8,10,14,16,18,25,26,28,30,32,34,38,40,42,46,48,50,52,54,56]

samples_all = pd.DataFrame(columns=['electrode', 'flex_ind1_max', 'flex_ind1_min', 'flex_ind2_max', 'flex_ind2_min',
                                    'flex_mid1_max', 'flex_mid1_min', 'flex_mid2_max', 'flex_mid2_min', 'flex_thumb1_max',
                                    'flex_thumb1_min', 'flex_thumb2_max', 'flex_thumb2_min', 'roll_max', 'roll_min',
                                    'pitch_max', 'pitch_min', 'yaw_max', 'yaw_min'])
for number in measurements:

    filename = "Measurement" + str(number)
    print(filename)
    samples_meas = pd.read_pickle(filename + '.pkl')
    print(samples_meas)
    samples_all = samples_all.append(samples_meas)

print(samples_all)
samples_all.to_pickle('MeasurmentsALL.pkl')'''


fig, (ax1, ax2, ax3, ax4) = plt.subplots(4,1, sharey=True)
ax1.plot(range(fulltime), flexion_ind1, label = ('mcp flex ind' ))
ax1.plot(range(fulltime), flexion_ind2, label = ('pip flex ind' ))
ax1.set(xlabel='', ylabel='ind flexion (°)')
ax1.legend()
ax1.grid()
ax2.plot(range(fulltime), flexion_mid1, label = ('mcp flex mid' ))
ax2.plot(range(fulltime), flexion_mid2, label = ('pip flex mid' ))
ax2.set(xlabel='', ylabel='mid flexion (°)')
ax2.legend()
ax2.grid()
ax3.plot(range(fulltime), flexion_thumb1, label = ('mcp flex thumb' ))
ax3.plot(range(fulltime), flexion_thumb2, label = ('pip flex thumb' ))
ax3.set(xlabel='', ylabel='thumb flexion (°)')
ax3.legend()
ax3.grid()
ax4.plot(range(fulltime), roll, label = ('roll'))
ax4.plot(range(fulltime), pitch, label = ('pitch'))
ax4.plot(range(fulltime), yaw, label = ('yaw'))
ax4.set(xlabel='frames', ylabel='rpy angles(°)')
ax4.legend()
ax4.grid()
plt.show()

