from functools import partial

import gymnasium as gym
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import slimevolleygym # type: ignore[import]

from src.genome import Genome, phenotype_forward
from src.population import NEATConfig
from src.trainer import evolve


def evaluate_genome_slimevolley(genome: Genome, key: jax.Array, n_episodes: int = 3) -> float:
    """
    Evaluates a genome by playing SlimeVolley.
    Returns the average cumulative reward over n_episodes.
    """
    env = gym.make("SlimeVolley-v0")
    total_rewards = []
    
    for _ in range(n_episodes):
        key, episode_key = jr.split(key)
        seed = int(jr.randint(episode_key, (), 0, 10000))
        obs, info = env.reset(seed=seed)
        
        done = False
        episode_reward = 0.0
        
        while not done:
            # Convert observation to JAX array and get action from genome
            obs_jax = jnp.array(obs, dtype=jnp.float32)
            action_values = phenotype_forward(genome, obs_jax)
            
            # Convert continuous outputs to discrete actions [forward, backward, jump]
            # Use threshold of 0.5 for binary actions
            action = (action_values > 0.5).astype(np.int32)
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += float(reward)
        
        total_rewards.append(episode_reward)
    
    env.close()
    
    # Fitness is the average reward
    fitness = float(np.mean(total_rewards))
    return fitness

def main():
    # SlimeVolley has 12 inputs (state observation) and 3 outputs (actions)
    N_INPUTS = 12
    N_OUTPUTS = 3
    N_GENERATIONS = 100
    
    config = NEATConfig(pop_size=200, delta_threshold=0.6)
    key = jr.PRNGKey(42)
    eval_fn = partial(evaluate_genome_slimevolley, n_episodes=5)
    
    print("Starting SlimeVolley evolution...")
    result = evolve(
        n_inputs=N_INPUTS,
        n_outputs=N_OUTPUTS,
        eval_fn=eval_fn,
        key=key,
        config=config,
        generations=N_GENERATIONS,
        add_bias=True,
        verbose=True,
    )
    
    best_genome_idx = int(jnp.argmax(jnp.array(result.population.fitness)))
    best_genome = result.population.genomes[best_genome_idx]
    best_fit = float(result.population.fitness[best_genome_idx])
    print(f"Best fitness during training: {best_fit:.2f}")
    
    test_score = evaluate_genome_slimevolley(best_genome, jr.PRNGKey(321), n_episodes=5)
    print(f"Average test return over 5 episodes: {test_score:.2f}")
    
    breakpoint()
    
    # Visual demonstration
    env = gym.make("SlimeVolley-v0")
    obs, info = env.reset(seed=42)
    
    done = False
    episode_reward = 0.0
    
    print("\nPlaying demonstration game (rendering)...")
    while not done:
        obs_jax = jnp.array(obs, dtype=jnp.float32)
        action_values = phenotype_forward(best_genome, obs_jax)
        action = (action_values > 0.5).astype(np.int32)
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode_reward += float(reward)
        env.render()
    
    print(f"Demonstration game reward: {episode_reward:.2f}")
    env.close()

if __name__ == "__main__":
    main()