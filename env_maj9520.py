import gymnasium as gym            # gymnasium for creating environment
import numpy as np                 # for numerical operations
import matplotlib.pyplot as plt    # for rendering and visualization
import matplotlib.image as mpimg   # for reading images

class MyGame(gym.Env):
    """
    Custom environment that follows gymnasium interface.
    This environment includes an agent, goal states, hell states, obstacles, and a background image.
    The agent (Naruto) navigating a goal which is Secreat Ninja Code. But Ōtsutsuki Clan and Orochimarut are the enemies of Naruto. 
    If Naruto hits two enemies, he start from original position. Also, there are some walls as a obstacles
    """
    def __init__(self, grid_size=7):
        super(MyGame, self).__init__()
        self.grid_size = grid_size
        self.cell_size = 80
        self.agent_state = np.array([3, 3])  # Initial agent position
        self.goal_state = np.array([3, 5])   # Goal position
        self.hail_states = [np.array([2, 4]), np.array([4, 4])]  # Positions with hail
        self.obstacle_states = [np.array([1, 5]), np.array([2, 5]), np.array([1, 4]),
                                np.array([5, 5]), np.array([4, 5]), np.array([5, 4]),
                                np.array([1, 3]), np.array([5, 3]), np.array([1, 2]), np.array([5, 2])]  # Positions of obstacles

        self.observation_space = gym.spaces.Box(low=0, high=self.grid_size-1, shape=(2,), dtype=np.int32)  # Observation space
        self.action_space = gym.spaces.Discrete(4)  # Action space: 4 possible actions (up, down, left, right)
        
        # Load images
        self.agent_img = mpimg.imread(r"agent.png")
        self.goal_img = mpimg.imread(r"scroll.png")
        self.hail_img1 = mpimg.imread(r"hail1.png")
        self.hail_img2 = mpimg.imread(r"hail3.png")
        self.background_img = mpimg.imread(r"background.jpg")
        self.obstacle_img = mpimg.imread(r"bricks.jpg")
        
        self.fig, self.ax = plt.subplots()
        plt.show(block=False)

    def reset(self):
        """
        Reset the environment to the initial state.
        
        Returns:
            np.ndarray: The initial state.
        """
        self.agent_state = np.array([1, 1])  # Reset agent to initial position
        return self.agent_state
    
    def step(self, action):
        """
        Take an action and update the environment state.
        
        Args:
            action (int): The action to take.
        
        Returns:
            tuple: A tuple of the new state, reward, done flag, and additional info.
        """
        new_state = self.agent_state.copy()  # Copy the current state
        
        # Update the agent's position based on the action
        if action == 0 and self.agent_state[1] < self.grid_size - 1:
            new_state[1] += 1
        if action == 1 and self.agent_state[1] > 0:
            new_state[1] -= 1
        if action == 2 and self.agent_state[0] < self.grid_size - 1:
            new_state[0] += 1
        if action == 3 and self.agent_state[0] > 0:
            new_state[0] -= 1

        # Check if new_state is an obstacle
        if any(np.array_equal(new_state, obs_state) for obs_state in self.obstacle_states):
            reward = 0  # No reward for hitting an obstacle
            done = False  # Episode is not done
            info = "Hit an obstacle, cannot move there"
            return self.agent_state, reward, done, info

        self.agent_state = new_state  # Update the agent's state

        # Check if agent has moved to a hail state
        if any(np.array_equal(self.agent_state, hail_state) for hail_state in self.hail_states):
            self.agent_state = np.array([1, 1])  # Reset agent to initial position
            reward = -1  # Negative reward for hitting hail state
            done = False  # Episode is not done
            info = "Hit hail state, reset to start position"
            return self.agent_state, reward, done, info

        reward = 0  # Default reward
        done = np.array_equal(self.agent_state, self.goal_state)  # Check if goal is reached
        if done:
            reward = 10  # Reward for reaching the goal

        distance_to_goal = np.linalg.norm(self.goal_state - self.agent_state)  # Calculate distance to goal
        info = f"Distance To Goal is = {distance_to_goal}"

        return self.agent_state, reward, done, info
    
    def render(self):
        """
        Render the environment.
        """
        self.ax.clear()  # Clear the previous plot
        
        # Plot the background
        self.ax.imshow(self.background_img, extent=[-0.5, self.grid_size - 0.5, -0.5, self.grid_size - 0.5])
        
        # Plot the agent
        self.ax.imshow(self.agent_img, extent=[self.agent_state[0] - 0.5, self.agent_state[0] + 0.5, self.agent_state[1] - 0.5, self.agent_state[1] + 0.5])
        
        # Plot the goal
        self.ax.imshow(self.goal_img, extent=[self.goal_state[0] - 0.5, self.goal_state[0] + 0.5, self.goal_state[1] - 0.5, self.goal_state[1] + 0.5])
        
        # Plot the hail states with different images
        self.ax.imshow(self.hail_img1, extent=[self.hail_states[0][0] - 0.5, self.hail_states[0][0] + 0.5, self.hail_states[0][1] - 0.5, self.hail_states[0][1] + 0.5])
        self.ax.imshow(self.hail_img2, extent=[self.hail_states[1][0] - 0.5, self.hail_states[1][0] + 0.5, self.hail_states[1][1] - 0.5, self.hail_states[1][1] + 0.5])
        
        # Plot the obstacles
        for obs_state in self.obstacle_states:
            self.ax.imshow(self.obstacle_img, extent=[obs_state[0] - 0.5, obs_state[0] + 0.5, obs_state[1] - 0.5, obs_state[1] + 0.5])
        
        self.ax.set_xlim(-0.5, self.grid_size - 0.5)
        self.ax.set_ylim(-0.5, self.grid_size - 0.5)
        self.ax.set_aspect("equal")
        self.ax.axis('off')  # Remove axis markings
        plt.pause(0.1)  # Pause to create an animation effect

    def close(self):
        """
        Close the rendering window.
        """
        plt.close()

if __name__ == "__main__":
    env = MyGame()
    state = env.reset()
    for _ in range(500):
        action = env.action_space.sample()  # Randomly sample an action
        agent_state, reward, done, info = env.step(action)
        env.render()  # Render the environment
        print(f"State: {agent_state}, Reward: {reward}, Done: {done}, Info: {info}") 
    
        if done:
            print("I reached the goal")
            break
    env.close()
