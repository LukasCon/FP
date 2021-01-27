import time
import serial

ser = serial.Serial()
ser.baudrate = 115200
ser.port = 'COM4'
ser.timeout = 1
ser.open()
print(ser)

class velec:
    def __init__(self,number,name='test'):
        self.number = number

        if number <= 4:
            print("Using the first four velecs (1-4) can cause mal functions, thus it is recommended to define new velecs from 5 on.")
        else:

            self.name = name
            self.cathodelist = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            self.amp         = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            self.width       = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            self.anode       = []
            self.selected    = 1

    def cathodes(self,activecathodes):
        self.cathodelist = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for x in activecathodes:
            self.cathodelist[x-1] = 1

    def amplitudes(self,amplitudes):
        activecathodesindices = [i+1 for i in range(len(self.cathodelist)) if self.cathodelist[i] == 1]
        for x in activecathodesindices:
            self.amp[x - 1] = amplitudes[activecathodesindices.index(x)]

    def widths(self,widths):
        activecathodesindices = [i+1 for i in range(len(self.cathodelist)) if self.cathodelist[i] == 1]
        for x in activecathodesindices:
            self.width[x - 1] = widths[activecathodesindices.index(x)]

    def anodes(self,activeanodes):
        anodelist = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        activecathodes= [i + 1 for i in range(len(self.cathodelist)) if self.cathodelist[i] == 1]

        def commonelements(x, y):
            common = 0
            for value in x:
                if value in y:
                    common = 1
            return common

        if (activeanodes == [2] or activeanodes == [16] or activeanodes == [2,16]) and commonelements(activeanodes,activecathodes) == 0:
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

    def stim(self,seconds = 2):
        message_start = "stim %s\r\n" %(self.name)
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
              "*anode %s *selected %s *sync 0\r\n"\
              %(velec.number,velec.name,velec.cathodelist[0],velec.cathodelist[1],velec.cathodelist[2],velec.cathodelist[3],
                velec.cathodelist[4],velec.cathodelist[5],velec.cathodelist[6],velec.cathodelist[7],velec.cathodelist[8],
                velec.cathodelist[9],velec.cathodelist[10],velec.cathodelist[11],velec.cathodelist[12],velec.cathodelist[13],
                velec.cathodelist[14],velec.cathodelist[15],
                velec.amp[0],velec.amp[1],velec.amp[2],velec.amp[3],velec.amp[4],velec.amp[5],velec.amp[6],velec.amp[7],
                velec.amp[8],velec.amp[9],velec.amp[10],velec.amp[11],velec.amp[12],velec.amp[13],velec.amp[14],velec.amp[15],
                velec.width[0],velec.width[1],velec.width[2],velec.width[3],velec.width[4],velec.width[5],velec.width[6],velec.width[7],
                velec.width[8],velec.width[9],velec.width[10],velec.width[11],velec.width[12],velec.width[13],velec.width[14],velec.width[15],
                velec.anode,velec.selected)

    return message

def define_twitch_velecs(velec_order):
    velecs = []
    for i in range(len(velec_order)):
        name = 'velec' + str(velec_order[i])
        # velecs.append(velec(velec_order[i]+4, name))
        velecs.append(velec(8, name))
        velecs[i].cathodes([velec_order[i]])
        velecs[i].amplitudes([1])
        velecs[i].widths([100])
        velecs[i].anodes([2])
        # print(velecs[i])
        velecs[i].define()
        time.sleep(1)
        velecs[i].stim()

    return velecs

ser.write(b"iam DESKTOP\r\n")
ser.write(b"elec 1 *pads_qty 16\r\n")
ser.write(b"freq 35\r\n")

'''velec_order = [4, 6, 3, 1, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15]
twitch_velecs = define_twitch_velecs(velec_order)


velec10 = velec(11, 'close')
velec10.cathodes([3])
velec10.amplitudes([8])
velec10.widths([300])
velec10.anodes([2])
print(velec10)
velec10.define()
velec10.stim()

'''
ser.write(b"battery ?\r\n")
preanswer = ser.read_until(b"battery")
answer = ser.read_until(b"\r\n")
print(preanswer)
print(answer)

ser.close()
