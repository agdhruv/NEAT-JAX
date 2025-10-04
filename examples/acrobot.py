import jax
import jax.numpy as jnp
import jax.random as jr

import gymnasium as gym  # type: ignore[import]

from functools import partial

from src.genome import Genome, phenotype_forward
from src.population import NEATConfig
from src.trainer import evolve


def acrobot_eval(
    genome: Genome,
    key: jax.Array,
    episodes: int = 3,
    max_steps: int = 500,
) -> float:
    """Evaluate a genome on Gymnasium's Acrobot-v1.

    Reward is -1 per step until terminal (goal achieved), so higher (less negative) is better.
    We negate the mean steps to failure so that higher is better for the NEAT loop.
    """
    env = gym.make("Acrobot-v1")
    total_return = 0.0

    for _ in range(episodes):
        key, k_seed = jr.split(key)
        seed = int(jr.randint(k_seed, (), 0, 2**31 - 1))
        obs, _ = env.reset(seed=seed)

        ep_ret = 0.0
        for _ in range(max_steps):
            x = jnp.array(obs, dtype=jnp.float32)
            y = phenotype_forward(genome, x)
            action = int((y[0] > 0.0))  # Acrobot has 3 discrete actions {0,1,2}; map sign to {0,2}
            # Try a simple mapping: negative -> 0, positive -> 2, near zero -> 1
            if y[0] < -0.25:
                action = 0
            elif y[0] > 0.25:
                action = 2
            else:
                action = 1
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_ret += float(reward)
            if terminated or truncated:
                break
        total_return += ep_ret
    env.close()

    # Fitness: higher is better. Acrobot gives -1 per step until success; 
    # so average reward is in [-max_steps, 0]. Return it directly; evolve uses higher-is-better.
    return total_return / float(episodes)


def main():
    # Acrobot observation is 6-dim; actions are 3 discrete.
    N_INPUTS = 6
    N_OUTPUTS = 1  # thresholded to 3 actions
    N_GENERATIONS = 100

    config = NEATConfig()
    key = jr.PRNGKey(1)
    eval_fn = partial(acrobot_eval, episodes=3, max_steps=500)

    print("Starting Acrobot evolution...")
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

    # Evaluate the best genome after evolution
    best_idx = int(jnp.argmax(jnp.array(result.population.fitness)))
    best_genome = result.population.genomes[best_idx]
    best_fit = float(result.population.fitness[best_idx])
    print(f"Best fitness during training: {best_fit:.2f}")

    test_score = acrobot_eval(best_genome, jr.PRNGKey(321), episodes=5, max_steps=500)
    print(f"Average test return over 5 episodes: {test_score:.2f}")


if __name__ == "__main__":
    main()


