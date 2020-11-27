import time
import numpy as np

from random_environment import Environment
from agent import Agent


def test_agent():
    display_on = True

    # Create a random seed, which will define the environment
    random_seed = int(time.time())
    np.random.seed(random_seed)

    # Create a random environment
    environment = Environment(magnification=500)

    # Create an agent
    agent = Agent()

    # Get the initial state
    state = environment.init_state

    # Determine the time at which training will stop, i.e. in 10 minutes (600 seconds) time
    start_time = time.time()
    end_time = start_time + 600

    # Train the agent, until the time is up
    while time.time() < end_time:
        # If the action is to start a new episode, then reset the state
        if agent.has_finished_episode():
            state = environment.init_state
        # Get the state and action from the agent
        action = agent.get_next_action(state)
        # Get the next state and the distance to the goal
        next_state, distance_to_goal = environment.step(state, action)
        # Return this to the agent
        agent.set_next_state_and_distance(next_state, distance_to_goal)
        # Set what the new state is
        state = next_state
        # Optionally, show the environment
        if display_on:
            environment.show(state)

    # Test the agent for 100 steps, using its greedy policy
    state = environment.init_state
    has_reached_goal = False
    for step_num in range(100):
        action = agent.get_greedy_action(state)
        next_state, distance_to_goal = environment.step(state, action)
        # The agent must achieve a maximum distance of 0.03 for use to consider it "reaching the goal"
        if distance_to_goal < 0.03:
            has_reached_goal = True
            break
        state = next_state

    # Print out the result
    resultstr = (
        f"Reached goal in {step_num} steps."
        if has_reached_goal
        else f"Did not reach goal. Final distance = {distance_to_goal}"
    )
    print(f"Result for maze {random_seed}: {resultstr}")

    return has_reached_goal


# Main entry point
if __name__ == "__main__":
    score = 0
    num_tests = 20
    for i in range(num_tests):
        if test_agent():
            score += 1
    print(f"achieved score of {score * (100 / num_tests)}%")
