# Continuous Control for High-Dimensional State Spaces: An Interactive Learning Approach
Code of the paper "Continuous Control for High-Dimensional State Spaces: An Interactive Learning Approach" submitted to ICRA 2019.

This repository is an extension of [Interactive Learning with Corrective Feedback for Policies based on Deep Neural Networks](https://github.com/rperezdattari/Interactive-Learning-with-Corrective-Feedback-for-Policies-based-on-Deep-Neural-Networks). The *enhanced* version of D-COACH was added in this project; the version presented in "Interactive Learning with Corrective Feedback for Policies based on Deep Neural Networks" is now called *basic*.

This code is based on the following publication:
1. Not available yet.

**Authors:** Rodrigo Pérez-Dattari, Carlos Celemin, Javier Ruiz-del-Solar, Jens Kober.

Link to paper video:

Not available yet.

## Installation

To use the code, it is necessary to first install the gym toolkit (release v0.9.6): https://github.com/openai/gym

Then, the files in the `gym` folder of this repository should be replaced/added in the installed gym folder on your PC. There are modifications of two gym environments:

1. **Duckie Racing:** Duckietown car simulation available at https://github.com/duckietown/gym-duckietown. *TODO: add Duckie Racing config files and specifications*

2. **Car Racing:** the same CarRacing environment of Gym with some bug fixes and modifications in the main loop for database generation.

### Requirements
* setuptools==38.5.1
* numpy==1.13.3
* opencv_python==3.4.0.12
* matplotlib==2.2.2
* tensorflow==1.4.0
* pyglet==1.3.2
* gym==0.9.6

## Usage

1. To run the main program type in the terminal (inside the folder `D-COACH`):

```bash 
python main.py --config-file <environment>
```
The default configuration files are **car_racing** and **duckie_racing**. The version of D-COACH (Enhanced/Basic) is selected in these configuration files.

To be able to give feedback to the agent, the environment rendering window must be selected/clicked.

## Comments

This code has been tested in `Ubuntu 16.04` and `python >= 3.5`.

## Troubleshooting

If you run into problems of any kind, don't hesitate to [open an issue](https://github.com/rperezdattari/Continuous-Control-for-High-Dimensional-State-Spaces-An-Interactive-Learning-Approach/issues) on this repository. It is quite possible that you have run into some bug we are not aware of.

