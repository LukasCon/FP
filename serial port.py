import time
import serial
from velec import Velec

ser = serial.Serial()
ser.baudrate = 115200
ser.port = 'COM4'
ser.timeout = 1
ser.open()
print(ser)

ser.write(b"iam DESKTOP\r\n")
ser.write(b"elec 1 *pads_qty 16\r\n")
ser.write(b"freq 35\r\n")


#Example for defining and stimulating a virtual electrode manually
'''velec10 = Velec(11, 'close')
velec10.cathodes([3])
velec10.amplitudes([8])
velec10.widths([300])
velec10.anodes([2])
print(velec10)
velec10.define()
velec10.stim_on()
time.sleep(2)
velec10.stim_off()'''


# Check battery status
ser.write(b"battery ?\r\n")
preanswer = ser.read_until(b"battery")
answer = ser.read_until(b"\r\n")
print(preanswer)
print(answer)

ser.close()
