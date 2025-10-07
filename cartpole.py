import jax
import jax.numpy as jnp
import jax.random as jr
import gymnasium as gym

from functools import partial

from src.genome import Genome, phenotype_forward
from src.population import NEATConfig
from src.trainer import evolve
from src.evaluator import SimpleEvaluator

def cartpole_sim(genome: Genome, max_steps: int = 500) -> float:
    """Simulate a genome on Gymnasium's CartPole-v1.

    Returns the total reward over `max_steps` steps.
    """
    env = gym.make("CartPole-v1", render_mode="human")
    obs, _ = env.reset()
    total_reward = 0.0
    for _ in range(max_steps):
        x = jnp.array(obs, dtype=jnp.float32)
        y = phenotype_forward(genome, x)
        action = int(y[0] > 0.0)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        if terminated or truncated:
            break
    env.close()
    return total_reward

def cartpole_eval(genome: Genome, key: jax.Array, episodes: int = 3, max_steps: int = 500) -> float:
    """Evaluate a genome on Gymnasium's CartPole-v1.

    Returns average episodic reward over `episodes` runs.
    """
    total_reward = 0.0

    for _ in range(episodes):
        key, k_seed = jr.split(key)
        seed = int(jr.randint(k_seed, (), 0, 2**31 - 1))
        env = gym.make("CartPole-v1")
        obs, _ = env.reset(seed=seed)

        episode_return = 0.0
        for _ in range(max_steps):
            x = jnp.array(obs, dtype=jnp.float32)
            y = phenotype_forward(genome, x)
            # Single output in [-1, 1]; threshold to choose action {0,1}
            action = int(y[0] > 0.0)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_return += float(reward)
            if terminated or truncated:
                break
        env.close()
        total_reward += episode_return
    
    avg_reward = total_reward / float(episodes)
    return avg_reward


if __name__ == "__main__":
    N_INPUTS = 4
    N_OUTPUTS = 1  # thresholded to 2 actions
    N_GENERATIONS = 20

    config = NEATConfig(delta_threshold=5.0, pop_size=100)

    key = jr.PRNGKey(0)

    eval_fn = partial(cartpole_eval, episodes=3, max_steps=500)
    evaluator = SimpleEvaluator(eval_fn)

    print("Starting CartPole evolution...")
    result = evolve(
        n_inputs=N_INPUTS,
        n_outputs=N_OUTPUTS,
        evaluator=evaluator,
        key=key,
        config=config,
        generations=N_GENERATIONS,
        add_bias=True,
        verbose=True,
    )

    # Evaluate the best genome after evolution
    best_idx = int(jnp.argmax(jnp.array(result.population.fitness)))
    best_genome = result.population.genomes[best_idx]
    best_fit = float(result.population.fitness[best_idx])
    print(f"Best fitness during training: {best_fit:.2f}")

    test_score = cartpole_eval(best_genome, jr.PRNGKey(123), episodes=5, max_steps=500)
    print(f"Average test return over 5 episodes: {test_score:.2f}")
    
    test_score = cartpole_sim(best_genome, max_steps=500)
    print(f"Score: {test_score:.2f}")