from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import random
import os
import json
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

#Set up environment, render mode can be set to 'human' for video game display, 'none' for faster training without display
env = gym_super_mario_bros.make('SuperMarioBros-v3', apply_api_compatibility=True, render_mode='human')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

#Extration action space size and observations size
action_space = env.action_space.n
state = env.reset()
state, reward, done, turncate, info = env.step(env.action_space.sample())
observation_space = 3 # X postion, Y_postion, time

#Define training optimizer
critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
actor_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Directory to save and load model checkpoints
save_dir = "checkpoints_5"
os.makedirs(save_dir, exist_ok=True)

#Flag for loading
Load = False

# Function to create the actor network
def create_actor_network(action_space,observation_space, network_length, network_width):
    model = tf.keras.Sequential()
    model.add(layers.Input(shape=(observation_space,)))  # Inputs of 3 observations
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(128, activation="relu")) #Add layer to agent 1
    model.add(layers.Dense(128, activation="relu")) #Add layer to agent 1
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(128, activation="relu")) #Add layer to agent 1
    model.add(layers.Dense(128, activation="relu")) #Add layer to agent 1
    model.add(layers.Dense(64, activation="relu")) 
    """
    for _ in range(network_length):
        model.add(layers.Dense(network_width, activation="relu"))
    """    
    # Output layer with 7 units for each discrete action, using softmax to output a probability distribution
    model.add(layers.Dense(action_space, activation="softmax"))
    return model

# Function to create the critic network
def create_critic_network(action_space, observation_space, network_length, network_width):
    model = tf.keras.Sequential()
    model.add(layers.Input(shape=(observation_space+action_space,)))#Inputs state-action pair so 2 observations plus 7 action probs
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(128, activation="relu")) #Added layer from agent 1
    model.add(layers.Dense(128, activation="relu")) #Added layer from agent 1
    """
    for _ in range(network_length):
        model.add(layers.Dense(network_width, activation="relu"))
    """    
    model.add(layers.Dense(1, activation="linear")) #Outputs Q-value
    return model

# Function to create the actor target network
def create_target_actor(action_space, observation_space, network_length, network_width):
    model = create_actor_network(action_space, observation_space, network_length, network_width)
    model.set_weights(model.get_weights())  # Set the same initial weights
    return model

# Function to create the critic target network
def create_target_critic(action_space,observation_space, network_length, network_width):
    model = create_critic_network(action_space,observation_space, network_length, network_width)
    model.set_weights(model.get_weights())  # Set the same initial weights
    return model

# Function to select action with actor
def select_action(observations, actor, epsilon):
    prob = random.random()
    observations = np.array([observations])
    action_values = actor.predict(observations, verbose=0)
    if prob < epsilon:
        # Use exploration: select random action
        action = random.randint(0, 6)
    else:
        # Use exploitation: select action from actor
        action = np.argmax(action_values[0])
    return action, action_values

# Updated function to train the neural networks with tf.concat for TensorFlow compatibility
def fit_network(critic, actor, critic_target, actor_target, obs, last_obs, reward, last_reward, action, gamma=0.99, tau=0.005):
    # One-hot encode the action for critic input
    action_one_hot = tf.one_hot(action, depth=action_space)
    action_one_hot = tf.expand_dims(action_one_hot, axis=0)

    # Ensure obs and last_obs are batch-like
    obs = tf.expand_dims(obs, axis=0)
    last_obs = tf.expand_dims(last_obs, axis=0)
    reward = tf.expand_dims(reward, axis=0)

    # Critic Target Update
    next_action = actor_target(last_obs, training=False)
    target_critic_input = tf.concat([last_obs, next_action], axis=-1)
    target_q_value = tf.cast(reward, dtype=tf.float32) + gamma * critic_target(target_critic_input, training=False)
    target_q_value = tf.reshape(target_q_value, (-1, 1))

    # Critic Loss
    with tf.GradientTape() as tape:
        current_critic_input = tf.concat([obs, action_one_hot], axis=-1)
        current_q_value = critic(current_critic_input, training=True)
        critic_loss = tf.keras.losses.MSE(target_q_value, current_q_value)
    critic_grads = tape.gradient(critic_loss, critic.trainable_variables)
    critic_optimizer.apply_gradients(zip(critic_grads, critic.trainable_variables))

    # Actor Loss
    with tf.GradientTape() as tape:
        actor_action = actor(obs, training=True)
        critic_input = tf.concat([obs, actor_action], axis=-1)
        critic_value = critic(critic_input, training=True)
        actor_loss = -tf.reduce_mean(critic_value)
    actor_grads = tape.gradient(actor_loss, actor.trainable_variables)
    actor_optimizer.apply_gradients(zip(actor_grads, actor.trainable_variables))

    # Soft update of target networks
    for target_param, param in zip(critic_target.trainable_variables, critic.trainable_variables):
        target_param.assign(tau * param + (1 - tau) * target_param)
    for target_param, param in zip(actor_target.trainable_variables, actor.trainable_variables):
        target_param.assign(tau * param + (1 - tau) * target_param)


def convert_observations(observation):
    # Numerical features
    x_pos = observation['x_pos']
    y_pos = observation['y_pos']
    time = observation['time']


    # Combine all features into a numpy array
    observation_array = np.array([
    x_pos, y_pos,time,
    ], dtype=np.float32)
    
    return observation_array

# Function to save the model weights and training state
def save_progress(actor, critic, episode, epsilon, save_dir=save_dir):
    # Save model weights
    actor.save_weights(os.path.join(save_dir, f"actor_weights_episode_latest.h5"))
    critic.save_weights(os.path.join(save_dir, f"critic_weights_episode_latest.h5"))
    
    # Save additional data (epsilon, current episode)
    training_state = {"episode": episode, "epsilon": epsilon}
    with open(os.path.join(save_dir, f"training_state.json"), "w") as f:
        json.dump(training_state, f)
    
    print(f"Progress saved at episode {episode}.")

# Function to load the model weights and training state
def load_progress(actor, critic, save_dir=save_dir):
    # Load model weights
    actor_weights = os.path.join(save_dir, "actor_weights_episode_latest.h5")
    critic_weights = os.path.join(save_dir, "critic_weights_episode_latest.h5")
    state_file = os.path.join(save_dir, "training_state.json")
    
    if not os.path.exists(actor_weights) or not os.path.exists(critic_weights):
        print("No saved weights found. Starting fresh.")
        return 0, 0.5  # Default values for episode and epsilon
    
    actor.load_weights(actor_weights)
    critic.load_weights(critic_weights)
    
    # Load additional data (epsilon, current episode)
    with open(state_file, "r") as f:
        training_state = json.load(f)
    
    print(f"Progress loaded from episode {training_state['episode']}.")
    return training_state["episode"], training_state["epsilon"]

#Create DDPG neural networks actor and critic as well as the target networks
network_len = 5
network_width = [64,128,128,128,64]
actor_network = create_actor_network(action_space,observation_space,network_len,network_width)
actor_network_target = create_target_actor(action_space,observation_space,network_len,network_width)
critic_network = create_critic_network(action_space,observation_space,network_len,network_width)
critic_network_target = create_target_critic(action_space,observation_space,network_len,network_width)

#Traing loop and parameters
number_episodes = 1000
number_steps = 1200   #1060 is speed run, 2000 is casual amount of steps
epsilon = 0.5
time = 0
train = True

# Modify the training loop to save progress every 10 episodes
# Load existing progress if available
if Load and train:
    # Load epsilon and the last episode for the file
    current_episode, epsilon = load_progress(actor_network, critic_network)
    if epsilon <= 0.05:
        epsilon = 0.5

    # Training loop: For loaded agent
    for episode in range(current_episode, number_episodes):
        state = env.reset()
        episode_reward = 0
        for step in range(number_steps):
            if step == 0:
                state, reward, done, truncate, obs = env.step(3)
                obs = convert_observations(obs)
            action, action_value = select_action(obs, actor_network, epsilon)
            epsilon = max(0.05, epsilon * 0.9999) # Decay factor, leads to 5% exploration after 25 eps
            last_obs = obs
            last_reward = reward
            state, reward, done, truncate, obs = env.step(action)
            obs = convert_observations(obs)
            fit_network(critic_network, actor_network, critic_network_target, actor_network_target, obs, last_obs, reward, last_reward, action)
            episode_reward += reward
            tf.keras.backend.clear_session() # release unused resources
            if done or truncate:
                break
        print(f"Episode: {episode + 1}, Reward: {episode_reward}")
        
        # Save progress every 10 episodes
        if (episode + 1) % 10 == 0:
            save_progress(actor_network, critic_network, episode + 1, epsilon)
        # Resets exploration back to %35 every 50 episodes
        if (episode + 1) % 50 == 0:
            epsilon = 0.35 
elif train and not Load:  
    # Training loop: For new agent
    for episode in range(number_episodes):
        state= env.reset()
        episode_reward = 0
        for step in range(number_steps):
            if step == 0:
                state, reward, done, truncate, obs = env.step(3)
                obs = convert_observations(obs) 
            action, action_value = select_action(obs, actor_network,epsilon)
            epsilon = max(0.05, epsilon * 0.99992) # Decay factor, leads to 5% exploration after 25 eps
            last_obs = obs
            last_reward = reward
            state, reward, done, truncate, obs = env.step(action)
            obs = convert_observations(obs)
            fit_network(critic_network, actor_network, critic_network_target, actor_network_target, obs, last_obs, reward, last_reward, action)
            episode_reward += reward
            tf.keras.backend.clear_session() # release unused resources
            if done or truncate:
                break
        print(f"Episode: {episode + 1}, Reward: {episode_reward}")
        # Save progress every 10 episodes
        if (episode + 1) % 10 == 0:
            save_progress(actor_network, critic_network, episode + 1, epsilon)
        # Resets exploration back to %35 every 50 episodes
        if (episode + 1) % 50 == 0:
            epsilon = 0.35
    else:
        print("Oops")