# Forschungspraxis 
In this repository there are two scripts to perform a calibration session for the FES device of Technalia with the Qualisys motion capture system (Miqus).
During the calibration session electrodes are stimulated, then the resulting finger movements are observed and evaluated in comparison to predefined ideal movements.

The ```Twitch Response``` script selects the electrodes in a predetermined consecutive order.\
In the ```Thompson Sampling``` script the electrodes are considered as one arm of a multi-armed bandit problem and a modified [Thompson Sampling](https://web.stanford.edu/~bvr/pubs/TS_Tutorial.pdf)
algorithm is used to solve the online decision making problem of selecting the next electrode to stimulate.

The ```plot_distributions``` script should help to analyse the results.

The ```serial port``` script establishes a Bluetooth connection to the FES device. Here you can communicate with the FES device separately to check the battery level 
or define and stimulate virtual electrodes manually.

## Installation

1.  Clone the repo.

2.  You will also need the [Qualisys SDK for Python](https://github.com/qualisys/qualisys_python_sdk).
    Just go to the repo, clone it and copy the qtm folder into you project folder.

## Usage

### Preparations:

1. Open QTM. 
    - DO NOT start a new measurement, this is already done in the code.
    - Make sure 'continuous capture' is enabled in the capture settings.
 
2. Attach the markers to your fingers.
    - Make sure the AIM model fits and Qualisys recognizes your fingers and wrist properly.

3. Attach the electrode pad to your arm.

### Initialization:

For ```Twitch Response``` and ```Thompson Sampling```:

```new_file =```\
  The virtual electrodes will be saved as an instance of the Bandit class, which includes their achieved accuracys and other important information. 
  Choose a name for the pickle file to save them in. 

```capture_new_ideal_mov =```
  - ```True```, you will have a time slot of 8 seconds right after the capture start to perform the ideal movements
    for each finger. The maximal flexion angles in reference to your initial position will be extracted and used to evaluate the induced movements during the calibration session.
  
  - ```False```, you need to predefine ```old_ideal_flexions``` with a 1x9 array that includes the desired maximal flexion angle of the index finger mcp joint,
    the index finger pip joint, the middle finger mcp joint, the middle finger pip joint, the thumb mcp joint, the thumb pip joint as well as the desired maximal 
    roll, pitch and yaw angle. This predefined angles will be used as reference for the induced movements during the calibration session.
    
Further self-explanatory parameters can be changed in the block of 'Initialize parameters'.
    
You can also decide which virtual electrodes should be considered during the 'normal' search. Therefore, you have to define the electrode pads and the amplitudes of the ```start_bandits```.
This also applies for the amplitudes of the virtual electrodes (```new_bandits```) you want to be considered during the 'deeper' search.



For ```Thompson Sampling``` only:

```use_uniform_priors =```
  - ```True```, the virtual electrodes will all start with an uniform prior distribution (alpha = 1, beta = 1).
  
  - ```False```, the virtual electrodes will start with the beta distributions they had at the end of the previous experiment you defined as ```last_experiment =```.
  
### Results

After each stimulation the main characterisitcs of the selected virtual electrode (electrode pads and amplitude) will be printed out as well as the resulting accuracies
for each finger.
In an optimal run, you will find at least one virtual electrode for each finger which achieves the desired ideal flexion with at least 75% accuracy.

All virtual electrodes are saved with their updated distributions and accuracies. 
Additionally, a chronological list of all stimulated virtual electrodes is saved.
The resulting flexions are also saved in a pickle file to ease new plots.

In the ```plot_distributions``` script you can print out the top 10 virtual electrodes with the best accuracy for each finger, as well as their according posterior 
distribution. Furthermore, a figure shows all virtual electrodes' best accuracy of the recent run plotted against the posterior mean. 


  


