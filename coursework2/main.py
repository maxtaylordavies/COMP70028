# Import some modules from other libraries
import torch
import collections
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

# Import local helper modules
from environment import Environment
from q_value_visualiser import QValueVisualiser

# Turn on interactive mode for PyPlot, to prevent the displayed graph from blocking the program flow
plt.ion()

class ReplayBuffer:
    def __init__(self):
        self.buffer = collections.deque(maxlen=5000)
    
    def record_transition(self, transition):
        self.buffer.append(transition)

    def sample_random_transitions(self, n):
        idxs = np.random.randint(len(self.buffer), size=n)
        return np.array([self.buffer[i] for i in idxs])

    def length(self):
        return len(self.buffer)

# The Agent class allows the agent to interact with the environment.
class Agent:

    # The class initialisation function.
    def __init__(self, environment):
        # Set the agent's environment.
        self.environment = environment
        # Create the agent's current state
        self.state = None
        # Initialise the agent's set of possible actions
        self.actions = np.arange(4)
        # Initialise constant step magnitude
        self.step_size = 0.1
        # Create the agent's total reward for the current episode.
        self.total_reward = None
        # Set epsilon
        self.epsilon = 0.1
        # Reset the agent.
        self.reset()

    # Function to reset the environment, and set the agent to its initial state. This should be done at the start of every episode.
    def reset(self):
        # Reset the environment for the start of the new episode, and set the agent's state to the initial state as defined by the environment.
        self.state = self.environment.reset()
        # Set the agent's total reward for this episode to zero.
        self.total_reward = 0.0

    def decrease_epsilon(self, factor):
        self.epsilon /= factor

    # Function to make the agent take one step in the environment.
    def step(self, policy=None):
        # Choose the next action.
        discrete_action = self._choose_random_action() if policy is None else self._choose_epsilon_greedy_action(policy)
        # Convert the discrete action into a continuous action.
        continuous_action = self._discrete_action_to_continuous(discrete_action)
        # Take one step in the environment, using this continuous action, based on the agent's current state. This returns the next state, and the new distance to the goal from this new state. It also draws the environment, if display=True was set when creating the environment object..
        next_state, distance_to_goal = self.environment.step(self.state, continuous_action)
        # Compute the reward for this paction.
        reward = self._compute_reward(distance_to_goal)
        # Create a transition tuple for this step.
        transition = (self.state, discrete_action, reward, next_state)
        # Set the agent's state for the next step, as the next state from this step
        self.state = next_state
        # Update the agent's reward for this episode
        self.total_reward += reward
        # Return the transition
        return transition

    # Function for the agent to choose its next action
    def _choose_random_action(self):
        # Return random discrete action
        return np.random.choice(self.actions)

    def _choose_epsilon_greedy_action(self, greedy_policy):
        if np.random.random() <= self.epsilon:
            return self._choose_random_action()
        return greedy_policy[int(self.state[1]) * 10, int(self.state[0]) * 10]

    # Function to convert discrete action (as used by a DQN) to a continuous action (as used by the environment).
    def _discrete_action_to_continuous(self, discrete_action):
        if discrete_action == 0:
            # Move right
            continuous_action = np.array([self.step_size, 0], dtype=np.float32)
        elif discrete_action == 1:
            # Move up
            continuous_action = np.array([0, self.step_size], dtype=np.float32)
        elif discrete_action == 2:
            # Move left
            continuous_action = np.array([-self.step_size, 0], dtype=np.float32)
        elif discrete_action == 3:
            # Move down
            continuous_action = np.array([0, -self.step_size], dtype=np.float32)
        return continuous_action

    # Function for the agent to compute its reward. In this example, the reward is based on the agent's distance to the goal after the agent takes an action.
    def _compute_reward(self, distance_to_goal):
        reward = float(0.1*(1 - distance_to_goal))
        return reward


# The Network class inherits the torch.nn.Module class, which represents a neural network.
class Network(torch.nn.Module):

    # The class initialisation function. This takes as arguments the dimension of the network's input (i.e. the dimension of the state), and the dimension of the network's output (i.e. the dimension of the action).
    def __init__(self, input_dimension, output_dimension):
        # Call the initialisation function of the parent class.
        super(Network, self).__init__()
        # Define the network layers. This example network has two hidden layers, each with 100 units.
        self.layer_1 = torch.nn.Linear(in_features=input_dimension, out_features=100)
        self.layer_2 = torch.nn.Linear(in_features=100, out_features=100)
        self.output_layer = torch.nn.Linear(in_features=100, out_features=output_dimension)

    # Function which sends some input data through the network and returns the network's output. In this example, a ReLU activation function is used for both hidden layers, but the output layer has no activation function (it is just a linear layer).
    def forward(self, input):
        layer_1_output = torch.nn.functional.relu(self.layer_1(input))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        output = self.output_layer(layer_2_output)
        return output


# The DQN class determines how to train the above neural network.
class DQN:

    # The class initialisation function.
    def __init__(self):
        # Create a Q-network, which predicts the q-value for a particular state.
        self.q_network = Network(input_dimension=2, output_dimension=4)
        # Create a target network, with same architecture as Q-network
        self.target_network = Network(input_dimension=2, output_dimension=4)
        self.copy_weights()
        # Define the optimiser which is used when updating the Q-network. The learning rate determines how big each gradient step is during backpropagation.
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=0.01)

    def copy_weights(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decrease_learning_rate(self, factor):
        for param_group in self.optimiser.param_groups:
            param_group['lr'] /= factor

    # Function that is called whenever we want to train the Q-network. Each call to this function takes in a transition tuple containing the data we use to update the Q-network.
    def train_network(self, transitions, discount_factor=0.9, use_target_network=False):
        # Set all the gradients stored in the optimiser to zero.
        self.optimiser.zero_grad()
        # Calculate the loss for this transition.
        loss = self._calculate_loss(transitions, discount_factor, use_target_network)
        # Compute the gradients based on this loss, i.e. the gradients of the loss with respect to the Q-network parameters.
        loss.backward()
        # Take one gradient step to update the Q-network.
        self.optimiser.step()
        # Return the loss as a scalar
        return loss.item()

    # Function to calculate the loss for a batch of transitions
    def _calculate_loss(self, transitions, discount_factor=0.9, use_target_network=False):
        first_states = np.zeros((len(transitions), 2), dtype=np.float32)
        action_idxs = np.zeros(len(transitions), dtype=np.int64)
        experienced_rewards = np.zeros(len(transitions), dtype=np.float32)
        second_states = np.zeros((len(transitions), 2), dtype=np.float32)

        for i,t in enumerate(transitions):
            first_states[i], action_idxs[i], experienced_rewards[i], second_states[i] = t

        first_state_output = self.q_network.forward(torch.tensor(first_states))
        if use_target_network:
            second_state_output = self.target_network.forward(torch.tensor(second_states))
        else:
            second_state_output = self.q_network.forward(torch.tensor(second_states))

        first_state_values = first_state_output.gather(dim=1, index=torch.tensor(action_idxs).unsqueeze(-1)).squeeze(-1)
        second_state_max_values = torch.max(second_state_output, dim=1)[0]

        bellman_values = torch.tensor(experienced_rewards) + (discount_factor * second_state_max_values)

        return torch.nn.MSELoss()(first_state_values, bellman_values)

    def get_q_values(self):
        # need to generate a tensor of states, where each state represents the center of a square on the grid
        state_grid = np.zeros((10,10,2), dtype=np.float32)
        for row in range(10):
            for col in range(10):
                x = (col / 10.0) + 0.05
                y = (row / 10.0) + 0.05
                state_grid[row,col] = (x,y)

        q_values = self.target_network.forward(torch.tensor(state_grid.reshape((100,2)))).detach().numpy().reshape((10,10,4))
        return q_values

    def get_greedy_policy(self):
        q_values = self.get_q_values()
        policy = np.zeros((10,10))
        for row in range(10):
            for col in range(10):
                action_values = q_values[row,col]
                policy[row,col] = np.argmax(action_values)
        return policy

def train_without_target_network():
    # Initialise some parameters
    num_eps = 100
    ep_length = 20

    # Create an environment.
    # If display is True, then the environment will be displayed after every agent step. This can be set to False to speed up training time. The evaluation in part 2 of the coursework will be done based on the time with display=False.
    # Magnification determines how big the window will be when displaying the environment on your monitor. For desktop monitors, a value of 1000 should be about right. For laptops, a value of 500 should be about right. Note that this value does not affect the underlying state space or the learning, just the visualisation of the environment.
    environment = Environment(display=True, magnification=500)
    q_vis = QValueVisualiser(environment)
    agent = Agent(environment)
    buffer = ReplayBuffer()

    # Initialise Q network and target network
    dqn = DQN()

    losses = []
    iterations = []

    fig, ax = plt.subplots()
    ax.set(xlabel='Iteration', ylabel='Loss', title='Loss Curve for DQN (no target network)')

    # Loop over episodes
    for ep_idx in tqdm(range(num_eps)):
        # Reset the environment for the start of the episode.
        agent.reset()
        # Initialise average loss for this episode
        episode_loss_average = 0
        # Loop over steps within this episode. The episode length here is 20.
        for step_num in range(ep_length):
            # Step the agent once, and record the transition tuple
            transition = agent.step()
            buffer.record_transition(transition)

            # Train the Q NETWORK on a random sample from the replay buffer
            if buffer.length() >= 100:
                training_sample = buffer.sample_random_transitions(200)
                loss = dqn.train_network(training_sample)

                # update the episode loss average
                episode_loss_average += (loss - episode_loss_average)/(step_num + 1)

        if buffer.length() >= 100:
            iterations.append(ep_idx)
            losses.append(episode_loss_average)

        if ep_idx > 0 and ep_idx % 25 == 0:
            dqn.decrease_learning_rate(10)
   
    ax.plot(iterations, losses, color='blue')
    plt.yscale('log')
    plt.show()
    fig.savefig("3a.png")
    
    final_q_values = dqn.get_q_values()
    q_vis.draw_q_values(final_q_values, '4a.png')

    # compute greedy policy based on q values
    policy = np.zeros((10,10,2), dtype=np.float32)
    for row in range(10):
        for col in range(10):
            action_values = final_q_values[row,col]
            best_action_idx = np.argmax(action_values)
            policy[row,col] = agent._discrete_action_to_continuous(best_action_idx)
    environment.draw_greedy_policy(policy, 25, "4b.png")

def train_with_target_network():
    # Initialise some parameters
    num_eps = 100
    ep_length = 20

    # Create an environment.
    # If display is True, then the environment will be displayed after every agent step. This can be set to False to speed up training time. The evaluation in part 2 of the coursework will be done based on the time with display=False.
    # Magnification determines how big the window will be when displaying the environment on your monitor. For desktop monitors, a value of 1000 should be about right. For laptops, a value of 500 should be about right. Note that this value does not affect the underlying state space or the learning, just the visualisation of the environment.
    environment = Environment(display=True, magnification=500)
    q_vis = QValueVisualiser(environment)
    agent = Agent(environment)
    buffer = ReplayBuffer()

    # Initialise Q network and target network
    dqn = DQN()

    losses = []
    iterations = []

    fig, ax = plt.subplots()
    ax.set(xlabel='Iteration', ylabel='Loss', title='Loss Curve for DQN (target network)')

    # Loop over episodes
    for ep_idx in tqdm(range(num_eps)):
        # Reset the environment for the start of the episode.
        agent.reset()
        # Initialise average loss for this episode
        episode_loss_average = 0
        # Loop over steps within this episode. The episode length here is 20.
        for step_num in range(ep_length):
            # Step the agent once, and record the transition tuple
            transition = agent.step()
            buffer.record_transition(transition)

            # Train the TARGET NETWORK on a random sample from the replay buffer
            if buffer.length() >= 100:
                training_sample = buffer.sample_random_transitions(100)
                loss = dqn.train_network(training_sample, use_target_network=True)

                # update the episode loss average
                episode_loss_average += (loss - episode_loss_average)/(step_num + 1)

        if buffer.length() >= 100:
            iterations.append(ep_idx)
            losses.append(episode_loss_average)

        if ep_idx > 0 and ep_idx % 10 == 0:
            dqn.copy_weights()
   
    ax.plot(iterations, losses, color='blue')
    plt.yscale('log')
    plt.show()
    fig.savefig("3b.png")
    
    final_q_values = dqn.get_q_values()
    q_vis.draw_q_values(final_q_values, '4a_target.png')

    # compute greedy policy based on q values
    policy = np.zeros((10,10,2), dtype=np.float32)
    for row in range(10):
        for col in range(10):
            action_values = final_q_values[row,col]
            best_action_idx = np.argmax(action_values)
            policy[row,col] = agent._discrete_action_to_continuous(best_action_idx)
    environment.draw_greedy_policy(policy, 20, "4b_target.png")

def train_epsilon_greedy():
    # Initialise some parameters
    num_eps = 2000
    ep_length = 100

    # Create an environment.
    # If display is True, then the environment will be displayed after every agent step. This can be set to False to speed up training time. The evaluation in part 2 of the coursework will be done based on the time with display=False.
    # Magnification determines how big the window will be when displaying the environment on your monitor. For desktop monitors, a value of 1000 should be about right. For laptops, a value of 500 should be about right. Note that this value does not affect the underlying state space or the learning, just the visualisation of the environment.
    environment = Environment(display=True, magnification=500)
    q_vis = QValueVisualiser(environment)
    agent = Agent(environment)
    buffer = ReplayBuffer()

    # Initialise Q network and target network
    dqn = DQN()

    losses = []
    iterations = []

    fig, ax = plt.subplots()
    ax.set(xlabel='Iteration', ylabel='Loss', title='Loss Curve for DQN (epsilon-greedy)')

    # Loop over episodes
    for ep_idx in tqdm(range(num_eps)):
        # Reset the environment for the start of the episode.
        agent.reset()
        # Initialise average loss for this episode
        episode_loss_average = 0
        # Loop over steps within this episode. The episode length here is 20.
        for step_num in range(ep_length):
            # Step the agent once, and record the transition tuple
            policy = dqn.get_greedy_policy()
            transition = agent.step(policy)
            buffer.record_transition(transition)

            # Train the Q NETWORK on a random sample from the replay buffer
            if buffer.length() >= 200:
                training_sample = buffer.sample_random_transitions(200)
                loss = dqn.train_network(training_sample)

                # update the episode loss average
                episode_loss_average += (loss - episode_loss_average)/(step_num + 1)

        if buffer.length() >= 200:
            iterations.append(ep_idx)
            losses.append(episode_loss_average)

        if ep_idx != 0 and ep_idx % 400 == 0:
            dqn.decrease_learning_rate(5)
   
    ax.plot(iterations, losses, color='blue')
    plt.yscale('log')
    plt.show()
    fig.savefig("epsilon_loss_curve.png")
    
    final_q_values = dqn.get_q_values()
    q_vis.draw_q_values(final_q_values, '4a.png')

    # compute greedy policy based on q values
    policy = np.zeros((10,10,2), dtype=np.float32)
    for row in range(10):
        for col in range(10):
            action_values = final_q_values[row,col]
            best_action_idx = np.argmax(action_values)
            policy[row,col] = agent._discrete_action_to_continuous(best_action_idx)
    environment.draw_greedy_policy(policy, 25, "4b.png")


# Main entry point
if __name__ == "__main__":
    train_epsilon_greedy()