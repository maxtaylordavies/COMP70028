############################################################################
############################################################################
# THIS IS THE ONLY FILE YOU SHOULD EDIT
#
#
# Agent must always have these five functions:
#     __init__(self)
#     has_finished_episode(self)
#     get_next_action(self, state)
#     set_next_state_and_distance(self, next_state, distance_to_goal)
#     get_greedy_action(self, state)
#
#
# You may add any other functions as you wish
############################################################################
############################################################################

import collections
import torch
import numpy as np


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


# The Network class inherits the torch.nn.Module class, which represents a neural network.
class Network(torch.nn.Module):
    # The class initialisation function. This takes as arguments the dimension of the network's input (i.e. the dimension of the state), and the dimension of the network's output (i.e. the dimension of the action).
    def __init__(self, input_dimension, output_dimension):
        # Call the initialisation function of the parent class.
        super(Network, self).__init__()
        # Define the network layers. This example network has two hidden layers, each with 100 units.
        self.layer_1 = torch.nn.Linear(in_features=input_dimension, out_features=100)
        self.layer_2 = torch.nn.Linear(in_features=100, out_features=100)
        self.output_layer = torch.nn.Linear(
            in_features=100, out_features=output_dimension
        )

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
            param_group["lr"] /= factor

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
    def _calculate_loss(
        self, transitions, discount_factor=0.9, use_target_network=False
    ):
        first_states = np.zeros((len(transitions), 2), dtype=np.float32)
        action_idxs = np.zeros(len(transitions), dtype=np.int64)
        experienced_rewards = np.zeros(len(transitions), dtype=np.float32)
        second_states = np.zeros((len(transitions), 2), dtype=np.float32)

        for i, t in enumerate(transitions):
            (
                first_states[i],
                action_idxs[i],
                experienced_rewards[i],
                second_states[i],
            ) = t

        first_state_output = self.q_network.forward(torch.tensor(first_states))
        if use_target_network:
            second_state_output = self.target_network.forward(
                torch.tensor(second_states)
            )
        else:
            second_state_output = self.q_network.forward(torch.tensor(second_states))

        first_state_values = first_state_output.gather(
            dim=1, index=torch.tensor(action_idxs).unsqueeze(-1)
        ).squeeze(-1)
        second_state_max_values = torch.max(second_state_output, dim=1)[0]

        bellman_values = torch.tensor(experienced_rewards) + (
            discount_factor * second_state_max_values
        )

        return torch.nn.MSELoss()(first_state_values, bellman_values)

    def get_q_values(self):
        # need to generate a tensor of states, where each state represents the center of a square on the grid
        state_grid = np.zeros((10, 10, 2), dtype=np.float32)
        for row in range(10):
            for col in range(10):
                x = (col / 10.0) + 0.05
                y = (row / 10.0) + 0.05
                state_grid[row, col] = (x, y)

        q_values = (
            self.target_network.forward(torch.tensor(state_grid.reshape((100, 2))))
            .detach()
            .numpy()
            .reshape((10, 10, 4))
        )
        return q_values

    def get_greedy_policy(self):
        q_values = self.get_q_values()
        policy = np.zeros((10, 10))
        for row in range(10):
            for col in range(10):
                action_values = q_values[row, col]
                policy[row, col] = np.argmax(action_values)
        return policy


class Agent:
    # Function to initialise the agent
    def __init__(self):
        # Set the episode length
        self.episode_length = 1000
        # Reset the total number of steps which the agent has taken
        self.num_steps_taken = 0
        # The state variable stores the latest state of the agent in the environment
        self.state = None
        # The action variable stores the latest action which the agent has applied to the environment
        self.action = None

    # Function to check whether the agent has reached the end of an episode
    def has_finished_episode(self):
        if self.num_steps_taken % self.episode_length == 0:
            return True
        else:
            return False

    # Function to get the next action, using whatever method you like
    def get_next_action(self, state):
        # Here, the action is random, but you can change this
        action = np.random.uniform(low=-0.01, high=0.01, size=2).astype(np.float32)
        # Update the number of steps which the agent has taken
        self.num_steps_taken += 1
        # Store the state; this will be used later, when storing the transition
        self.state = state
        # Store the action; this will be used later, when storing the transition
        self.action = action
        return action

    # Function to set the next state and distance, which resulted from applying action self.action at state self.state
    def set_next_state_and_distance(self, next_state, distance_to_goal):
        # Convert the distance to a reward
        reward = 1 - distance_to_goal
        # Create a transition
        transition = (self.state, self.action, reward, next_state)
        # Now you can do something with this transition ...

    # Function to get the greedy action for a particular state
    def get_greedy_action(self, state):
        # Here, the greedy action is fixed, but you should change it so that it returns the action with the highest Q-value
        action = np.array([0.02, 0.0], dtype=np.float32)
        return action
