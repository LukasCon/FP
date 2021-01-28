import numpy as np
import pandas as pd

deepsearchs = pd.DataFrame(data = {'time': [], 'finger': []})

for finger in ['ind', 'mid', 'thumb']:
    previous_deepsearchs = deepsearchs[deepsearchs['finger'] == finger]
    if previous_deepsearchs.empty:
        numb_of_prev_deepsearchs = 0

    else:
        numb_of_prev_deepsearchs = previous_deepsearchs.shape[0]
        time_since_last_deepsearch = 2 - int(previous_deepsearchs.tail(1)['time'])
        print(time_since_last_deepsearch)


