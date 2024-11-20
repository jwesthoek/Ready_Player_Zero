# Ready_Player_Zero
This is a semester project for Intro the Machine learning were our team trains a RL agent to player Super Mario Bros 1-1

Intructions on running code
--------------------------------------------------------------------------------
1st: You gotta make a Conda environment with python version 3.8.0
conda create --name SuperMario python=3.8
conda activate SuperMario

2nd: You gotta pip install the Mario environment
conda install pip
pip install gym-super-mario-bros

3rd: Go to your Vs code Aand switch to your new Conda environment and run the following code

4th: You will need to locate the default enviornment(smb_env.py) and replace with our verion of smb_env.py
The path to the file should look like C:\Users\'Your Name'\miniconda3\envs\SuperMario2\Lib\site-packages\gym_super_mario_bros or were ever your conda env is

With that Your all set to run Ready_Player_Zero_v4.py
IMPORTANT NOTES! The file has two flags you manually set to True/False
These flags are Train and Load
Train: Set to True if you want to train the agent and False if you just want use a agent for testing purposes.
Load: Set to True if you want to load a perexisting agent and false if you want to create a brand new one.

Change what "save_dir" is equal to to change/create new folders to hold agents parameters.
