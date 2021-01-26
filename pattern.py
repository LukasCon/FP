class pattern:
    def __init__(self, name):
        self.name = name
        self.form = []

    def ramp(self,amp_perc, width_perc, duration):
        self.form.append(['RAMP',amp_perc, width_perc, duration])

    def const(self,amp_perc, width_perc, duration):
        self.form.append(['CONST',amp_perc, width_perc, duration])

    def define(self):
        for i in range(len(self.form)):
            message = "sdcard ed default/test/%s.ptn %s %s R %s %s %s\r\n" %(self.name, self.form[i][0], self.form[i][0], self.form[i][1], self.form[i][2], self.form[i][3])
            print(bytes(message, 'utf-8'))
            #ser.write(bytes(message, 'utf-8'))

    def create(self):
        message = "sdcard cat > default/test/%s.ptn\r\n" %self.name
        print(bytes(message, 'utf-8'))
        #ser.write(bytes(message, 'utf-8'))

    def remove(self):
        message = "sdcard rm default/test/%s.ptn\r\n" %self.name
        print(bytes(message, 'utf-8'))
        #ser.write(bytes(message, 'utf-8'))

    def __repr__(self):
        string = 'Name: %s\n' % self.name
        for i in range(len(self.form)):
            string = string + '%s\n' %self.form[i]

        return string

ve6 = pattern('ve6')
ve6.ramp(100,100,500)
ve6.const(100,100,5000)
ve6.ramp(0,0,500)
print(ve6)
ve6.remove() # first remove pattern, otherwise previously defined ramps and constants remain and new chunks are only appended to the end
ve6.create() # create a fresh pattern, without any previously defined ramps and constants
ve6.define() # adds newly defined ramps and constants to EXISTING pattern
