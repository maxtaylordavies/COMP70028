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

import torch
import numpy as np
from matplotlib import pyplot as plt


class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = [None] * max_size
        self.probs = [None] * max_size
        self.max_size = max_size
        self.index = 0
        self.size = 0

    def record_transition(self, transition):
        self.buffer[self.index] = transition
        self.probs[self.index] = min(1 - transition[2], 0.2)
        self.size = min(self.size + 1, self.max_size)
        self.index = (self.index + 1) % self.max_size

    def sample_random_transitions(self, n):
        idxs = np.random.choice(np.arange(self.size), size=n)
        return [self.buffer[idx] for idx in idxs]

    def sample_prioritized_transitions(self, n):
        probs = np.array(self.probs[: self.size])
        idxs = np.random.choice(
            np.arange(self.size),
            size=n,
            p=probs / sum(probs),
        )
        return [self.buffer[idx] for idx in idxs]

    def length(self):
        return self.size


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

    def decay_learning_rate(self):
        for param_group in self.optimiser.param_groups:
            param_group["lr"] *= 0.95

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

    def get_q_values_for_state(self, state):
        output = self.q_network.forward(torch.tensor(np.array([state])))
        return output[0].detach().numpy()


class Agent:
    ############################################################################
    #                          REQUIRED FUNCTIONS                              #
    ############################################################################

    # Function to initialise the agent
    def __init__(self):
        self.finished = False
        self.greedy = False
        # Set the episode length
        self.episode_length = 1000
        self.previous_episode_last_step = 0
        # Reset total number of episodes agent has experienced
        self.num_episodes = 0
        # Reset the total number of steps which the agent has taken
        self.num_steps_taken = 0
        # The state variable stores the latest state of the agent in the environment
        self.state = None
        # The action variable stores the latest action which the agent has applied to the environment
        self.action = None
        # Step size
        self.step_size = 0.02
        # Initialise discrete action space
        self.actions = np.array(
            [
                [self.step_size, 0],  # right
                [0, self.step_size],  # up
                [-self.step_size, 0],  # left
                [0, -self.step_size],  # down
            ],
            dtype=np.float32,
        )
        # Exploration
        self.max_epsilon = 0.9
        self.epsilon = 0.9
        # Minibatch size
        self.minibatch_size = 100
        # Losses
        self.average_losses = [0]
        # Initialise an experience replay buffer
        self.buffer = ReplayBuffer(10000)
        # Initialise a DQN
        self.dqn = DQN()

        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.ax.set(
            xlabel="Episodes",
            ylabel="Average Loss",
            title="Loss Curve",
        )

    # Function to check whether the agent has reached the end of an episode
    def has_finished_episode(self):
        return (
            self.num_steps_taken - self.previous_episode_last_step
        ) % self.episode_length == 0

    # Function to get the next action, using whatever method you like
    def get_next_action(self, state):
        action = (
            self.get_greedy_action(state)
            if self.greedy
            else self.choose_epsilon_greedy_action(state)
        )
        # Update the number of steps which the agent has taken
        self.num_steps_taken += 1
        # Store the state; this will be used later, when storing the transition
        self.state = state
        # Store the action; this will be used later, when storing the transition
        self.action = action
        return action

    # Function to set the next state and distance, which resulted from applying action self.action at state self.state
    def set_next_state_and_distance(self, next_state, distance_to_goal):
        if self.greedy and self.episode_length == 100:
            self.max_epsilon = distance_to_goal / 2
            if distance_to_goal < 0.03:
                self.finished = True

        # Convert the distance to a reward
        reward = 1 - distance_to_goal
        if (self.state == next_state).all() and distance_to_goal > 0.25:
            reward /= 2  # discourage going into walls
        # Create and record a transition
        transition = (
            self.state,
            self.get_last_action_idx(),
            reward,
            # 0 if (next_state == self.state).all() else reward,
            next_state,
        )
        self.buffer.record_transition(transition)

        # If we have enough transitions stored, train the network
        if self.buffer.length() >= self.minibatch_size and not self.finished:
            self.take_training_step()

        if self.has_finished_episode():
            self.ax.plot(self.average_losses, color="red")
            plt.yscale("log")
            plt.show()
            self.average_losses.append(0)
            self.num_episodes += 1
            self.decay_epsilon()
            self.reduce_episode_length()
            self.dqn.decay_learning_rate()

            # Every 5 episodes, evaluate the greedy policy
            if self.num_episodes > 0 and self.num_episodes % 5 == 0:
                self.greedy = True
            else:
                self.greedy = False

        self.epsilon = min(self.max_epsilon, distance_to_goal)

        if self.num_steps_taken % 100 == 0:
            self.dqn.copy_weights()

    # Function to get the greedy action for a particular state
    def get_greedy_action(self, state):
        q_values = self.dqn.get_q_values_for_state(state)
        return self.actions[np.argmax(q_values)]

    ############################################################################
    #                             NEW FUNCTIONS                                #
    ############################################################################

    def decay_epsilon(self):
        self.max_epsilon *= 0.95
        # self.max_epsilon = max(self.max_epsilon * 0.95, 0.01)

    def reduce_episode_length(self):
        self.previous_episode_last_step = self.num_steps_taken
        self.episode_length = max(int(self.episode_length * 0.95), 100)

    def get_last_action_idx(self):
        return (self.actions == self.action).all(axis=1).nonzero()[0][0]

    def choose_random_action(self):
        idx = np.random.randint(len(self.actions))
        return self.actions[idx]

    def choose_epsilon_greedy_action(self, state):
        if np.random.random() <= self.epsilon or self.state is None:
            return self.choose_random_action()
        q_values = self.dqn.get_q_values_for_state(state)
        return self.actions[np.argmax(q_values)]

    def take_training_step(self):
        sample = self.buffer.sample_random_transitions(self.minibatch_size)
        loss = self.dqn.train_network(sample, use_target_network=True)

        self.average_losses[-1] += (
            loss - self.average_losses[-1]
        ) / self.num_steps_taken
