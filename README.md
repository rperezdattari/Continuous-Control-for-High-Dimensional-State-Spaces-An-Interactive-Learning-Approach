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

1. **Continuous-CartPole:** a continuous-action version of the Gym CartPole environment.

2. **CarRacing:** the same CarRacing environment of Gym with some bug fixes and modifications in the main loop for database generation.

```
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
The default configuration files are **car_racing** and **cartpole**.

To be able to give feedback to the agent, the environment rendering window must be selected/clicked.

To train the autoencoder for the high-dimensional state environments run (inside the folder `D-COACH`):

```bash 
python autoencoder.py
```
2. To generate a database for the CarRacing environment run the (replaced) file `car_racing.py` in the downloaded gym repository.

To modify the dimension of the images in the generated database, this database must be in the folder `D-COACH` and from this folder run:

```bash 
python tools/transform_database_dim.py
```

## Comments

This code has been tested in `Ubuntu 16.04` and `python >= 3.5`.

*TODO: add Gym-Duckietown config files*

## Troubleshooting

If you run into problems of any kind, don't hesitate to [open an issue](https://github.com/rperezdattari/Interactive-Learning-with-Corrective-Feedback-for-Policies-based-on-Deep-Neural-Networks/issues) on this repository. It is quite possible that you have run into some bug we are not aware of.

