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


max_steps = 3000
env = SlimeVolley(test=True, max_steps=max_steps)

# Cache compiled policies per-plan so we don't rebuild the function each call
_policy_cache = {}

def _get_policy(plan: Plan):
    sig = plan.signature()
    if sig not in _policy_cache:
        _policy_cache[sig] = make_policy(plan)   # jitted fn: (weights, obs[E,Di]) -> actions[E,Do]
    return _policy_cache[sig]

_rollout_cache = {}   # (plan_sig, n_episodes) -> compiled fn

def make_weighted_rollout(policy_apply, n_episodes: int):
    """Create a JIT-compiled rollout function for evaluating a neural network policy.
    
    This function creates a vectorized rollout that runs multiple episodes in parallel
    to evaluate a policy's performance. The policy is parameterized by weights that
    can be changed between calls without recompilation.
    
    Args:
        policy_apply: JIT-compiled policy function (weights, obs) -> actions
        n_episodes: Number of parallel episodes to run for evaluation
        
    Returns:
        A JIT-compiled function with signature (key, weights) -> mean_reward
        where weights are the neural network parameters to evaluate.
    """
    @jax.jit  # n_episodes baked via closure; weights is dynamic
    def _rollout(key: jax.Array, weights: jax.Array) -> jax.Array:
        ep_keys = jr.split(key, n_episodes)
        states  = env.reset(ep_keys)
        dones   = jnp.zeros(n_episodes, dtype=bool)
        rews    = jnp.zeros(n_episodes)

        def step(carry, _):
            states, rews, dones = carry
            actions = policy_apply(weights, states.obs)
            # apply sigmoid to actions
            actions = jax.nn.sigmoid(actions)
            # scale actions from [0, 1] to [-1, 1]
            actions = actions * 2 - 1
            nstates, r, d = env.step(states, actions)
            rews  = rews + jnp.where(dones, 0.0, r)
            dones = dones | d
            return (nstates, rews, dones), None

        (states, rews, _), _ = lax.scan(step, (states, rews, dones), None, length=max_steps)
        return jnp.mean(rews)
    return _rollout

def _get_weighted_rollout(plan: Plan, policy_apply, n_episodes: int):
    plan_sig = plan.signature()
    key = (plan_sig, n_episodes)
    fn = _rollout_cache.get(key)
    if fn is None:
        fn = make_weighted_rollout(policy_apply, n_episodes)
        _rollout_cache[key] = fn
    return fn

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
