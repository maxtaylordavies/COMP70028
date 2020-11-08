import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class GridWorld(object):
    def __init__(self, p):
        
        ### Attributes defining the Gridworld #######
        # Shape of the gridworld
        self.shape = (6,6)
        
        # Locations of the obstacles
        self.obstacle_locs = [(1,1),(2,3),(2,5),(3,1),(4,1),(4,2),(4,4)]
        
        # Locations for the absorbing states
        self.absorbing_locs = [(1,2),(4,3)]
        
        # Rewards for each of the absorbing states 
        self.special_rewards = [10, -100] #corresponds to each of the absorbing_locs
        
        # Reward for all the other states
        self.default_reward = -1
        
        # Starting location
        self.starting_loc = (3,0)
        
        # Action names
        self.action_names = ['N','E','S','W']
        
        # Number of actions
        self.action_size = len(self.action_names)
        
        # Randomizing action results: [1 0 0 0] to no Noise in the action results.
        self.action_randomizing_array = [p, (1-p)/3, (1-p)/3, (1-p)/3]
        
        ############################################
    
        #### Internal State  ####

        # Get attributes defining the world
        state_size, T, R, absorbing, locs = self.build_grid_world()
        
        # Number of valid states in the gridworld (there are 22 of them)
        self.state_size = state_size
        
        # Transition operator (3D tensor)
        self.T = T
        
        # Reward function (3D tensor)
        self.R = R
        
        # Absorbing states
        self.absorbing = absorbing
        
        # The locations of the valid states
        self.locs = locs
        
        # Number of the starting state
        self.starting_state = self.loc_to_state(self.starting_loc, locs);
        
        # Locating the initial state
        self.initial = np.zeros((1,len(locs)))
        self.initial[0,self.starting_state] = 1
        
        
        # Placing the walls on a bitmap
        self.walls = np.zeros(self.shape);
        for ob in self.obstacle_locs:
            self.walls[ob]=1
            
        # Placing the absorbers on a grid for illustration
        self.absorbers = np.zeros(self.shape)
        for ab in self.absorbing_locs:
            self.absorbers[ab] = -1
        
        # Placing the rewarders on a grid for illustration
        self.rewarders = np.zeros(self.shape)
        for i, rew in enumerate(self.absorbing_locs):
            self.rewarders[rew] = self.special_rewards[i]
        
        #Illustrating the grid world
        # self.paint_maps()
        ################################
    
    ####### Getters ###########
    
    def get_transition_matrix(self):
        return self.T
    
    def get_reward_matrix(self):
        return self.R
    
    ########################
    
    ########### Internal Helper Functions #####################
    def paint_maps(self):
        plt.figure()

        plt.subplot(1,3,1)
        plt.axis('off')
        plt.imshow(self.walls)
        plt.title('walls')

        plt.subplot(1,3,2)
        plt.axis('off')
        plt.imshow(self.absorbers)
        plt.title('absorbing states')

        plt.subplot(1,3,3)
        plt.axis('off')
        plt.imshow(self.rewarders)
        plt.title('immediate rewards')
        
        plt.show()
        
    def build_grid_world(self):
        # Get the locations of all the valid states, the neighbours of each state (by state number),
        # and the absorbing states (array of 0's with ones in the absorbing states)
        locations, neighbours, absorbing = self.get_topology()
        
        # Get the number of states
        S = len(locations)
        
        # Initialise the transition matrix
        T = np.zeros((S,S,4))
        
        for action in range(4):
            for effect in range(4):
                
                # Randomize the outcome of taking an action
                outcome = (action+effect+1) % 4
                if outcome == 0:
                    outcome = 3
                else:
                    outcome -= 1
    
                # Fill the transition matrix
                prob = self.action_randomizing_array[effect]
                for prior_state in range(S):
                    post_state = neighbours[prior_state, outcome]
                    post_state = int(post_state)
                    T[post_state,prior_state,action] = T[post_state,prior_state,action]+prob
                    
    
        # Build the reward matrix
        R = self.default_reward*np.ones((S,S,4))
        for i, sr in enumerate(self.special_rewards):
            post_state = self.loc_to_state(self.absorbing_locs[i],locations)
            R[post_state,:,:]= sr
        
        return S, T,R,absorbing,locations
    
    def get_topology(self):
        height = self.shape[0]
        width = self.shape[1]
        
        locs = []
        neighbour_locs = []
        
        for i in range(height):
            for j in range(width):
                # Get the locaiton of each state
                loc = (i,j)
                
                #And append it to the valid state locations if it is a valid state (ie not absorbing)
                if(self.is_location(loc)):
                    locs.append(loc)
                    
                    # Get an array with the neighbours of each state, in terms of locations
                    local_neighbours = [self.get_neighbour(loc,direction) for direction in self.action_names]
                    neighbour_locs.append(local_neighbours)
                
        # translate neighbour lists from locations to states
        num_states = len(locs)
        state_neighbours = np.zeros((num_states,4))
        
        for state in range(num_states):
            for direction in range(4):
                # Find neighbour location
                nloc = neighbour_locs[state][direction]
                
                # Turn location into a state number
                nstate = self.loc_to_state(nloc,locs)
      
                # Insert into neighbour matrix
                state_neighbours[state,direction] = nstate;
                
        # Translate absorbing locations into absorbing state indices
        absorbing = np.zeros((1,num_states))
        for a in self.absorbing_locs:
            absorbing_state = self.loc_to_state(a,locs)
            absorbing[0,absorbing_state] =1
        
        return locs, state_neighbours, absorbing    
    
    def loc_to_state(self,loc,locs):
        #takes list of locations and gives index corresponding to input loc
        return locs.index(tuple(loc))

    def is_location(self, loc):
        # It is a valid location if it is in grid and not obstacle
        if(loc[0]<0 or loc[1]<0 or loc[0]>self.shape[0]-1 or loc[1]>self.shape[1]-1):
            return False
        elif(loc in self.obstacle_locs):
            return False
        else:
             return True
            
    def get_neighbour(self,loc,direction):
        #Find the valid neighbours (ie that are in the grif and not obstacle)
        i = loc[0]
        j = loc[1]
        
        nr = (i-1,j)
        ea = (i,j+1)
        so = (i+1,j)
        we = (i,j-1)
        
        # If the neighbour is a valid location, accept it, otherwise, stay put
        if(direction == 'N' and self.is_location(nr)):
            return nr
        elif(direction == 'E' and self.is_location(ea)):
            return ea
        elif(direction == 'S' and self.is_location(so)):
            return so
        elif(direction == 'W' and self.is_location(we)):
            return we
        else:
            #default is to return to the same location
            return loc

    def generate_episode_with_random_start(self, policy):
        # initialise list to record our episode in format [(state_idx, action_idx, immediate_reward)]
        episode = []

        # select random starting location (sampled from uniform distribution)
        loc = self.locs[np.random.randint(len(self.locs))]

        # continue until termination
        while loc not in self.absorbing_locs:
            state_idx = self.loc_to_state(loc, self.locs)

            # probabilistically select action from policy
            action_probs = policy[state_idx]
            action_idx = np.random.choice(len(action_probs), p=action_probs)

            # determine the outcome of our action accounting for environment randomness
            outcome_idx = np.random.choice([action_idx] + [i for i in range(len(action_probs)) if i != action_idx], p=self.action_randomizing_array)
            
            # determine our next location
            next_loc = self.get_neighbour(loc, self.action_names[outcome_idx])

            # determine our immediate reward
            if next_loc in self.absorbing_locs:
                reward = self.special_rewards[self.absorbing_locs.index(next_loc)]
            else:
                reward = self.default_reward
            
            episode.append((state_idx, action_idx, reward))
            loc = next_loc
        
        return episode

    def compute_undiscounted_return(self, episode, start, gamma):
        return sum([reward for (_, _, reward) in episode[start:]])

    def compute_forward_discounted_return(self, episode, start, gamma):
        return sum([(gamma ** i) * reward for i, (_, _, reward) in enumerate(episode[start:])])

    def compute_backwards_discounted_return(self, episode, start, gamma):
        # compute the discounted return collected in the given episode from the step episode[i] onwards
        # we discount the return backwards to avoid suppressing the information in the terminal state reward
        return sum([(gamma ** i) * reward for i, (_, _, reward) in enumerate(episode[start:][::-1])])

    def rmse(self, data, mean):
        return np.sqrt(np.mean((data - mean) ** 2))

    ########################

####### Methods #########
    def draw_deterministic_policy(self, Policy, p, gamma, new_figure):
        # Draw a deterministic policy
        # The policy needs to be a np array of values between 0 and 3 with
        # 0 -> N, 1->E, 2->S, 3->W
        if new_figure:
            plt.figure()
            plt.axis('off')
        
        plt.imshow(self.walls+self.rewarders +self.absorbers)
        for state, action in enumerate(Policy):
            if(self.absorbing[0,state]):
                continue
            arrows = [r"$\uparrow$",r"$\rightarrow$", r"$\downarrow$", r"$\leftarrow$"]
            action_arrow = arrows[action]
            location = self.locs[state]
            plt.text(location[1], location[0], action_arrow, ha='center', va='center')

        if new_figure:
            plt.title(rf"Optimal policy for p={p}, $\gamma$={gamma}")
            plt.show()

    def draw_grid_values(self, values, p, gamma, new_figure):
        if new_figure:
            plt.figure()
            plt.axis('off')

        plt.imshow(self.walls+self.rewarders +self.absorbers)
        for state, value in enumerate(values):
            if (self.absorbing[0, state]):
                continue
            location = self.locs[state]
            plt.text(location[1], location[0], '{0:.2f}'.format(value), ha='center', va='center')
        
        if new_figure:
            plt.title(rf"State values for p={p}, $\gamma$={gamma}")
            plt.show()

    def policy_evaluation(self, policy, threshold, discount):
        
        # Make sure delta is bigger than the threshold to start with
        delta = 2*threshold
        
        #Get the reward and transition matrices
        R = self.get_reward_matrix()
        T = self.get_transition_matrix()
        
        # The value is initialised at 0
        V = np.zeros(policy.shape[0])
        # Make a deep copy of the value array to hold the update during the evaluation
        Vnew = np.copy(V)
        
        # While the Value has not yet converged do:
        while delta>threshold:
            for state_idx in range(policy.shape[0]):
                # If it is one of the absorbing states, ignore
                if(self.absorbing[0,state_idx]):
                    continue   
                
                # Accumulator variable for the Value of a state
                tmpV = 0
                for action_idx in range(policy.shape[1]):
                    # Accumulator variable for the State-Action Value
                    tmpQ = 0
                    for state_idx_prime in range(policy.shape[0]):
                        tmpQ = tmpQ + T[state_idx_prime,state_idx,action_idx] * (R[state_idx_prime,state_idx, action_idx] + discount * V[state_idx_prime])
                    
                    tmpV += policy[state_idx,action_idx] * tmpQ
                    
                # Update the value of the state
                Vnew[state_idx] = tmpV
            
            # After updating the values of all states, update the delta
            delta =  max(abs(Vnew-V))
            # and save the new value into the old
            V=np.copy(Vnew)
            
        return V

    def policy_iteration(self, policy, threshold, discount):
        T = self.get_transition_matrix()
        R = self.get_reward_matrix()
        policy_stable = False

        while not policy_stable:
            policy_stable = True
            value = self.policy_evaluation(policy, threshold, discount)

            # iterate through all states, finding the optimal action in each
            for state_idx in range(policy.shape[0]):
                # b is the most probable action dictated by the current policy in this state
                b = np.argmax(policy[state_idx])
                # compute the total value of the current state for every possible action
                action_values = np.zeros(policy.shape[1])
                for action_idx in range(policy.shape[1]):
                    for state_idx_prime in range(policy.shape[0]):
                        action_values[action_idx] += T[state_idx_prime,state_idx,action_idx] * (R[state_idx_prime,state_idx,action_idx] + discount * value[state_idx_prime])
                # and thus find the action that yields the highest value for this state
                best_action_idx = np.argmax(action_values)
                policy[state_idx] = np.eye(1, M=policy.shape[1], k=best_action_idx)
                
                # if we performed an update, set policy_stable to false to indicate that we haven't yet converged
                if best_action_idx != b:
                    policy_stable = False

        return policy

    def monte_carlo_iterative_optimisation(self, n, alpha, epsilon, gamma, discounting, show_progress=True, analytical_values=None):
        # initialise arrays to track the returns for our learning curve and our errors
        returns = np.zeros(n)
        errors = np.zeros(n)

        # initialise arbitrary Q-function and policy
        Q = np.zeros((self.state_size, self.action_size))
        policy = create_uniform_policy(self.state_size, self.action_size)

        # generate n episodes, using the experienced returns to iteratively update our Q-function and policy
        if show_progress:
            iterable = tqdm(range(n))
        else:
            iterable = range(n)

        for itr in iterable:
            # generate an episode using our current policy
            episode = self.generate_episode_with_random_start(policy)

            # for each state-action pair in the episode, compute the 
            # return collected following that pair
            for step_idx, (state_idx, action_idx, _) in enumerate(episode):
                if discounting == "forward":
                    R = self.compute_forward_discounted_return(episode, step_idx, gamma)
                elif discounting == "backward":
                    R = self.compute_backwards_discounted_return(episode, step_idx, gamma)
                else:
                    R = self.compute_undiscounted_return(episode, step_idx, gamma)

                # update our Q-value for the pair in the direction of our experienced return
                Q[state_idx, action_idx] += alpha * (R - Q[state_idx, action_idx])

                # record total return for episode
                if step_idx == 0:
                    returns[itr] = R

            # record the rmse of our state-values wrt analytical solution
            if analytical_values is not None:
                errors[itr] = self.rmse(np.amax(Q, axis=1), analytical_values)

            # update our policy to be epsilon-greedy in our current Q-function
            for state_idx in set([state_idx for (state_idx, _, _) in episode]):
                best_action_idx = np.argmax(Q[state_idx])
                policy[state_idx] = [1-epsilon+(epsilon/self.action_size)  if action_idx == best_action_idx else epsilon/self.action_size for action_idx in range(self.action_size)] 

        # finally update policy to be greedy in our final Q-function
        for state_idx in set([state_idx for (state_idx, _, _) in episode]):
                best_action_idx = np.argmax(Q[state_idx])
                policy[state_idx] = [1 if action_idx == best_action_idx else 0 for action_idx in range(self.action_size)] 
        
        return policy, Q, returns, errors

    def sarsa(self, n, alpha, epsilon, gamma, discounting, show_progress=True, analytical_values=None):
        # initialise array to track the returns for our learning curve
        returns = np.zeros(n)
        errors = np.zeros(n)

        # initialise arbitrary Q-function and policy
        Q = np.zeros((self.state_size, self.action_size))
        policy = create_uniform_policy(self.state_size, self.action_size)

        # generate n episodes, using the experienced returns to iteratively update our Q-function and policy
        if show_progress:
            iterable = tqdm(range(n))
        else:
            iterable = range(n)

        for itr in iterable:
            # generate an episode using our current policy
            episode = self.generate_episode_with_random_start(policy)

            # record total return for episode
            if discounting == "forward":
                R = self.compute_forward_discounted_return(episode, 0, gamma)
            elif discounting == "backward":
                R = self.compute_backwards_discounted_return(episode, 0, gamma)
            else:
                R = self.compute_undiscounted_return(episode, 0, gamma)
            returns[itr] = R

            # update our values of Q towards our estimated returns
            for step_idx in range(len(episode)):
                (s1, a1, r) = episode[step_idx]

                if step_idx == len(episode) - 1:
                    td_error = r  - Q[s1, a1]
                else:
                    (s2, a2, _) = episode[step_idx + 1]
                    td_error = r + (gamma * Q[s2, a2]) - Q[s1, a1]

                Q[s1, a1] += (alpha * td_error)
                    
            # update our policy to be epsilon-greedy in our current Q-function
            for state_idx in set([state_idx for (state_idx, _, _) in episode]):
                best_action_idx = np.argmax(Q[state_idx])
                policy[state_idx] = [1-epsilon+(epsilon/self.action_size)  if action_idx == best_action_idx else epsilon/self.action_size for action_idx in range(self.action_size)]

            # record the rmse of our state-values wrt analytical solution
            if analytical_values is not None:
                errors[itr] = self.rmse(np.amax(Q, axis=1), analytical_values) 

        # finally update policy to be greedy in our final Q-function
        for state_idx in set([state_idx for (state_idx, _, _) in episode]):
                best_action_idx = np.argmax(Q[state_idx])
                policy[state_idx] = [1 if action_idx == best_action_idx else 0 for action_idx in range(self.action_size)] 
        
        return policy, Q, returns, errors

    def plot_learning_curve(self, returns_matrix, epsilon, alpha, new_figure, show_error, color, algo_name):
        if new_figure:
            plt.figure()
        
        returns_average = np.average(returns_matrix, axis=0)
        returns_std = np.std(returns_matrix, axis=0)

        plt.plot(returns_average, color=color)

        if show_error:
            plt.fill_between(np.arange(len(returns_average)), returns_average - returns_std, returns_average + returns_std, alpha=0.2)

        plt.ylabel("Total return for episode")
        plt.xlabel("Number of episodes experienced")
        plt.title(rf"{algo_name} learning curve for $\epsilon$={epsilon}, $\alpha$={alpha}")

        if new_figure:
            plt.show()

########################################### 

def create_uniform_policy(state_size, action_size):
    return (1 / action_size) * np.ones((state_size, action_size))

def format_deterministic_policy(policy):
    return [np.argmax(row) for row in policy]

# return the optimal state values and optimal policy for 
# the gridworld given particular choices of p and gamma
def find_optimal_policy_and_values(grid, p, gamma, threshold):
    policy = create_uniform_policy(grid.state_size, grid.action_size)
    policy = grid.policy_iteration(policy, threshold, gamma)
    values = grid.policy_evaluation(policy, threshold, gamma)

    return policy, values

def q2b():
    threshold = 0.01

    # investigate effect of changing p on optimal policy and values
    plt.figure()

    gamma = 0.3
    for i,p in enumerate([0.1, 0.25]):
        grid = GridWorld(p)
        pol, val = find_optimal_policy_and_values(grid, p, gamma, threshold)

        plt.subplot(2,2,(2*i)+1)
        plt.axis("off")
        grid.draw_deterministic_policy(format_deterministic_policy(pol), p, gamma, False)
        plt.title(f"optimal policy for $p$={p}")

        plt.subplot(2,2,(2*i)+2)
        plt.axis("off")
        grid.draw_grid_values(val, p, gamma, False)
        plt.title(f"optimal values for $p$={p}")

    plt.suptitle("Effect of changing $p$ on optimal policy and values")
    plt.show()

    # investigate effect of changing gamma on optimal policy and values
    plt.figure()

    p = 0.75
    grid = GridWorld(p)
    for i,gamma in enumerate([0.1, 0.9]):
        pol, val = find_optimal_policy_and_values(grid, p, gamma, threshold)

        plt.subplot(2,2,(2*i)+1)
        plt.axis("off")
        grid.draw_deterministic_policy(format_deterministic_policy(pol), p, gamma, False)
        plt.title(rf"optimal policy for $\gamma$={gamma}")

        plt.subplot(2,2,(2*i)+2)
        plt.axis("off")
        grid.draw_grid_values(val, p, gamma, False)
        plt.title(rf"optimal values for $\gamma$={gamma}")
    
    plt.suptitle(r"Effect of changing $\gamma$ on optimal policy and values")
    plt.show()

def q2c():
    # initialise parameters and construct gridworld
    num_runs, num_episodes, p, gamma, epsilon, alpha = 200, 500, 0.75, 0.3, 0.1, 0.01
    grid = GridWorld(p)

    # run MC optimisation a bunch of times
    returns_matrix = np.zeros((num_runs, num_episodes))

    for run_idx in tqdm(range(num_runs)):
        policy, Q, returns, _ = grid.monte_carlo_iterative_optimisation(num_episodes, alpha, epsilon, gamma, discounting="backward")
        returns_matrix[run_idx] = returns
        
    # display policy and value function
    grid.draw_deterministic_policy(format_deterministic_policy(policy), p, gamma, new_figure=True)
    grid.draw_grid_values(np.amax(Q, axis=1), p, gamma, new_figure=True)

    # plot learning curve
    grid.plot_learning_curve(returns_matrix, epsilon, alpha, new_figure=True, show_error=True, color="teal", algo_name="Monte-Carlo")

    colors = ["palevioletred", "coral", "mediumseagreen", "darkblue"][::-1]
    
    # investigate effect of epsilon
    epsilons = [0, 0.1, 0.5, 1]
    plt.figure()
    for i in range(len(epsilons)):
        print(f"running with epsilon = {epsilons[i]}...")
        returns_matrix = np.zeros((num_runs, num_episodes))
        for run_idx in tqdm(range(num_runs)):
            _, _, returns, _ = grid.monte_carlo_iterative_optimisation(num_episodes, alpha, epsilons[i], gamma, discounting="backward")
            returns_matrix[run_idx] = returns
        grid.plot_learning_curve(returns_matrix, epsilon, alpha, new_figure=False, show_error=False, color=colors[i], algo_name="Monte-Carlo") 
    plt.title(r"Effect of varying $\epsilon$ on Monte-Carlo learning curves")
    plt.legend((rf"$\epsilon=0$", rf"$\epsilon=0.1$", rf"$\epsilon=0.5$", rf"$\epsilon=1$"), loc="upper right")
    plt.show()

    # investigate effect of alpha
    epsilon = 0.1
    alphas = [0.001, 0.01, 0.1, 0.5]
    plt.figure()
    for i in range(len(alphas)):
        print(f"running with alpha = {alphas[i]}...")
        returns_matrix = np.zeros((num_runs, num_episodes))
        for run_idx in tqdm(range(num_runs)):
            _, _, returns, _ = grid.monte_carlo_iterative_optimisation(num_episodes, alphas[i], epsilon, gamma, discounting="backward")
            returns_matrix[run_idx] = returns
        grid.plot_learning_curve(returns_matrix, epsilon, alpha, new_figure=False, show_error=False, color=colors[i], algo_name="Monte-Carlo") 
    plt.title(r"Effect of varying $\alpha$ on Monte-Carlo learning curves")
    plt.legend((rf"$\alpha=0.001$", rf"$\alpha=0.01$", rf"$\alpha=0.1$", rf"$\alpha=0.5$"), loc="upper right")
    plt.show()       

def q2d():
    # initialise parameters and construct gridworld
    num_runs, num_episodes, p, gamma, epsilon, alpha = 200, 500, 0.75, 0.3, 0.1, 0.01
    grid = GridWorld(p)

    # run TD optimisation once and display policy and value function
    policy, Q, _, _ = grid.sarsa(5000, alpha, epsilon, gamma, discounting="backward")
    grid.draw_deterministic_policy(format_deterministic_policy(policy), p, gamma, new_figure=True)
    grid.draw_grid_values(np.amax(Q, axis=1), p, gamma, new_figure=True)

    # run TD optimisation a bunch of times and plot learning curve with errors
    returns_matrix = np.zeros((num_runs, num_episodes))
    for run_idx in tqdm(range(num_runs)):
        _, Q, returns, _ = grid.sarsa(num_episodes, alpha, epsilon, gamma, discounting="backward", show_progress=False)
        returns_matrix[run_idx] = returns
    grid.plot_learning_curve(returns_matrix, epsilon, alpha, new_figure=True, show_error=True, color="teal", algo_name="Temporal Difference")

    colors = ["palevioletred", "coral", "mediumseagreen", "darkblue"][::-1]
    
    # investigate effect of epsilon
    epsilons = [0, 0.1, 0.5, 1]
    plt.figure()
    for i in range(len(epsilons)):
        print(f"running with epsilon = {epsilons[i]}...")
        returns_matrix = np.zeros((num_runs, num_episodes))
        for run_idx in tqdm(range(num_runs)):
            _, _, returns, _ = grid.sarsa(num_episodes, alpha, epsilons[i], gamma, discounting="backward", show_progress=False)
            returns_matrix[run_idx] = returns
        grid.plot_learning_curve(returns_matrix, epsilon, alpha, new_figure=False, show_error=False, color=colors[i], algo_name="Temporal Difference") 
    plt.title(r"Effect of varying $\epsilon$ on Temporal Difference learning curves")
    plt.legend((rf"$\epsilon=0$", rf"$\epsilon=0.1$", rf"$\epsilon=0.5$", rf"$\epsilon=1$"), loc="upper right")
    plt.show()

    # investigate effect of alpha
    epsilon = 0.1
    alphas = [0.001, 0.01, 0.1, 0.5]
    plt.figure()
    for i in range(len(alphas)):
        print(f"running with alpha = {alphas[i]}...")
        returns_matrix = np.zeros((num_runs, num_episodes))
        for run_idx in tqdm(range(num_runs)):
            _, _, returns, _ = grid.sarsa(num_episodes, alphas[i], epsilon, gamma, discounting="backward", show_progress=False)
            returns_matrix[run_idx] = returns
        grid.plot_learning_curve(returns_matrix, epsilon, alpha, new_figure=False, show_error=False, color=colors[i], algo_name="Temporal Difference") 
    plt.title(r"Effect of varying $\alpha$ on Temporal Difference learning curves")
    plt.legend((rf"$\alpha=0.001$", rf"$\alpha=0.01$", rf"$\alpha=0.1$", rf"$\alpha=0.5$"), loc="upper right")
    plt.show()

def q2e():
    # initialise parameters and construct gridworld
    num_runs, num_episodes, p, gamma, epsilon, alpha, threshold = 100, 1000, 0.75, 0.3, 0, 0.005, 0.001
    grid = GridWorld(p)

    _, dp_values = find_optimal_policy_and_values(grid, p, gamma, threshold)

    # # plot optimal state-value function RMSE against number of episodes for both MC and TD agents
    # _, _, _, mc_errors = grid.monte_carlo_iterative_optimisation(num_episodes, alpha, epsilon, gamma, discounting="forward", analytical_values=dp_values)
    # _, _, _, td_errors = grid.sarsa(num_episodes, alpha, epsilon, gamma, discounting="forward", analytical_values=dp_values)

    # plt.figure()
    # plt.plot(mc_errors, color="mediumseagreen")
    # plt.plot(td_errors, color="darkblue")
    # plt.xlabel("Number of episodes experienced")
    # plt.ylabel(r"RMSE of $V^*$ function")
    # plt.title(rf"Comparison of value RMSE curves for MC and TD ($\alpha={alpha}$)")
    # plt.legend(("MC", "TD"))
    # plt.show()

    # # do the same but this time perform a bunch of runs and display the standard deviation as well
    # mc_error_matrix = np.zeros((num_runs, num_episodes))
    # td_error_matrix = np.zeros((num_runs, num_episodes))

    # for run_idx in tqdm(range(num_runs)):
    #     _, _, _, mc_error_matrix[run_idx] = grid.monte_carlo_iterative_optimisation(num_episodes, alpha, epsilon, gamma, discounting="forward", show_progress=False, analytical_values=dp_values)
    #     _, _, _, td_error_matrix[run_idx] = grid.sarsa(num_episodes, alpha, epsilon, gamma, discounting="forward", show_progress=False, analytical_values=dp_values)
    
    # mc_error_average = np.average(mc_error_matrix, axis=0)
    # td_error_average = np.average(td_error_matrix, axis=0)
    # mc_error_std = np.std(mc_error_matrix, axis=0)
    # td_error_std = np.std(td_error_matrix, axis=0)

    # plt.figure()

    # plt.plot(mc_error_average, color="mediumseagreen")
    # plt.fill_between(np.arange(len(mc_error_average)), mc_error_average - mc_error_std, mc_error_average + mc_error_std, color="mediumseagreen", alpha=0.2)

    # plt.plot(td_error_average, color="darkblue")
    # plt.fill_between(np.arange(len(td_error_average)), td_error_average - td_error_std, td_error_average + td_error_std, color="darkblue", alpha=0.2)

    # plt.xlabel("Number of episodes experienced")
    # plt.ylabel(r"RMSE of $V^*$ function")
    # plt.title(rf"Comparison of value RMSE curves for MC and TD ($\alpha={alpha}$)")
    # plt.legend(("MC", "TD"))
    # plt.show()

    # plot optimal state-value function RMSE against total episode return 
    _, _, mc_returns, mc_errors = grid.monte_carlo_iterative_optimisation(num_episodes, alpha, epsilon, gamma, discounting="forward", analytical_values=dp_values)
    _, _, td_returns, td_errors = grid.sarsa(num_episodes, alpha, epsilon, gamma, discounting="forward", analytical_values=dp_values)
    
    plt.figure()

    plt.subplot(1, 2, 1)
    plt.scatter(mc_returns, mc_errors, color="mediumseagreen")
    plt.xlabel("Total discounted episode return")
    plt.ylabel(r"RMSE of $V^*$ function")
    plt.title("Monte-Carlo")

    plt.subplot(1, 2, 2)
    plt.scatter(td_returns, td_errors, color="darkblue")
    plt.xlabel("Total discounted episode return")
    plt.ylabel(r"RMSE of $V^*$ function")
    plt.title("Temporal Difference")

    plt.suptitle(rf"$V^*$ RMSE against total discounted episode return ($\alpha={alpha}$)")
    plt.show()

def main():
    q2e()

if __name__ == "__main__":
    main()