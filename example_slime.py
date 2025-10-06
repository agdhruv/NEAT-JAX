from functools import partial

from evojax.task.slimevolley import SlimeVolley
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.lax as lax
from src.genome import Genome, phenotype_forward
from src.population import NEATConfig
from src.trainer import evolve


max_steps = 3000
env = SlimeVolley(test=True, max_steps=max_steps)

def evaluate_genome_slimevolley(genome: Genome, key: jax.Array, n_episodes: int) -> float:
    # Split keys for parallel episodes
    episode_keys = jr.split(key, n_episodes)
    states = env.reset(episode_keys)
    
    # Track rewards for each episode
    episode_rewards = jnp.zeros(n_episodes)
    dones = jnp.zeros(n_episodes, dtype=bool)
    step_count = 0
    
    def cond_fn(val):
        states, rewards, dones, step_count = val
        return (~jnp.all(dones)) & (step_count < max_steps)

    def body_fn(val):
        states, rewards, dones, step_count = val
        # Get actions for all active episodes
        action_fn = jax.vmap(lambda obs: phenotype_forward(genome, obs))
        action_values = action_fn(states.obs)
        actions = (action_values - 0.5) * 2.0
        
        # Step all episodes
        new_states, step_rewards, new_dones = env.step(states, actions)
        
        # Accumulate rewards only for active episodes
        new_rewards = jnp.where(~dones, rewards + step_rewards, rewards)
        new_dones = dones | new_dones
        
        return (new_states, new_rewards, new_dones, step_count + 1)
    
    # Run the loop
    final_states, final_rewards, final_dones, final_steps = lax.while_loop(cond_fn, body_fn, (states, episode_rewards, dones, step_count))
    return float(jnp.mean(final_rewards))

def main():
    # SlimeVolley has 12 inputs (state observation) and 3 outputs (actions)
    N_INPUTS = 12
    N_OUTPUTS = 3
    N_GENERATIONS = 100

    config = NEATConfig(pop_size=100, delta_threshold=0.6)
    key = jr.PRNGKey(42)
    n_episodes = 3
    eval_fn = partial(evaluate_genome_slimevolley, n_episodes=n_episodes)
    
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
    
    test_score = evaluate_genome_slimevolley(best_genome, jr.PRNGKey(321), n_episodes=n_episodes)
    print(f"Average test return over {n_episodes} episodes: {test_score:.2f}")

if __name__ == "__main__":
    main()