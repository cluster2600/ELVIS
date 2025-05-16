import os
import torch
import numpy as np

def evaluate_multi_agent(agents, env, episodes=10):
    """
    Evaluate multiple agents in the environment, tracking individual and collective performance.
    Args:
        agents (list): List of agent objects.
        env: The environment to evaluate in.
        episodes (int): Number of episodes to run.
    Returns:
        dict: Dictionary with detailed individual and collective performance metrics.
    """
    total_rewards = {f'agent_{i}': [] for i in range(len(agents))}  # List to track per-episode rewards
    collective_rewards = []  # List to track collective rewards per episode
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        episode_reward = np.zeros(len(agents))  # Individual rewards per episode
        
        while not done:
            actions = [agent.select_action(state) for agent in agents]
            next_state, rewards, done, _ = env.step(actions)  # Assume env handles multi-agent steps
            state = next_state
            
            for i, reward in enumerate(rewards):
                episode_reward[i] += reward
                total_rewards[f'agent_{i}'].append(reward)  # Track per-step rewards for detailed metrics
        
        collective_rewards.append(sum(episode_reward))  # Track collective reward per episode
    
    average_individual_rewards = {key: np.mean(total_rewards[key]) for key in total_rewards}
    average_collective_reward = np.mean(collective_rewards)
    
    return {
        'individual_performance': average_individual_rewards,  # Average per-step rewards
        'collective_performance': average_collective_reward,
        'detailed_individual_rewards': total_rewards,  # Full list for further analysis
        'detailed_collective_rewards': collective_rewards  # Full list per episode
    }

def save_evaluation_results(results, cwd):
    """
    Save the evaluation results to a file, including detailed multi-agent metrics.
    Args:
        results (dict): The evaluation results.
        cwd (str): Current working directory.
    """
    file_path = os.path.join(cwd, 'evaluation_results.json')
    import json
    with open(file_path, 'w') as f:
        json.dump(results, f)
    print(f"Evaluation results saved to {file_path}")
