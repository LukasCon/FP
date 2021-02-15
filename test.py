import numpy as np
import pandas as pd
import pickle

bandits = pickle.load(open('activebandits_0217_5.pkl','rb'))
print(len(bandits))

