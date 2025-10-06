from functools import partial
from typing import Tuple, Callable

from evojax.task.slimevolley import SlimeVolley, State
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.lax as lax
from src.genome import Genome
from src.structure import build_plan_and_weights, make_policy, Plan
from src.population import NEATConfig
from src.trainer import evolve
from src.helper import _get_policy, _get_weighted_rollout


def evaluate_genome_slimevolley(genome: Genome, key: jax.Array, n_episodes: int) -> float:
    # 1) Extract static structure (plan) + dynamic weights
    plan, weights = build_plan_and_weights(genome)
    
    # 2) Get/compile the plan-specific policy once; reuse across calls
    policy_apply  = _get_policy(plan)
    
    # 3) Create a rollout that closes over (policy, weights)
    rollout_fn = _get_weighted_rollout(plan, policy_apply, n_episodes)
    
    # 4) Run the rollout
    reward = rollout_fn(key, weights)
    return float(reward)                  # <— weights passed, no capture

def main():
    # SlimeVolley has 12 inputs (state observation) and 3 outputs (actions)
    N_INPUTS = 12
    N_OUTPUTS = 3
    N_GENERATIONS = 500

    config = NEATConfig(pop_size=128, delta_threshold=0.6, w_init_std=0.5)
    key = jr.PRNGKey(42)
    n_episodes = 3
    eval_fn = partial(evaluate_genome_slimevolley, n_episodes=n_episodes)
    
    print("Starting SlimeVolley evolution...")
    import time
    start_time = time.time()
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
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    
    best_genome_idx = int(jnp.argmax(jnp.array(result.population.fitness)))
    best_genome = result.population.genomes[best_genome_idx]
    best_fit = float(result.population.fitness[best_genome_idx])
    print(f"Best fitness during training: {best_fit:.2f}")
    
    test_score = evaluate_genome_slimevolley(best_genome, jr.PRNGKey(321), n_episodes=n_episodes)
    print(f"Average test return over {n_episodes} episodes: {test_score:.2f}")

if __name__ == "__main__":
    main()
