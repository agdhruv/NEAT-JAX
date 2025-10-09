from functools import partial
import json

from evojax.task.slimevolley import SlimeVolley
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.lax as lax
from src.genome import Genome
from src.topology import build_topology_and_weights, Topology
from src.population import NEATConfig
from src.trainer import evolve
from src.topology import topology2policy
from src.evaluator import VectorizedEvaluator

max_steps = 500
env = SlimeVolley(test=False, max_steps=max_steps)

_rollout_cache = {}  # Cache compiled rollout functions per (topology_sig, n_episodes)
_policy_cache = {}   # Cache compiled policies per topology

def make_weighted_rollout(policy_apply, n_episodes: int):
    """Create a JIT-compiled rollout function for evaluating a neural network policy.
    
    This function creates a vectorized rollout that runs multiple episodes in parallel
    to evaluate a policy's performance. The policy is parameterized by weights that
    can be changed between calls without recompilation.
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
            actions = jax.nn.tanh(actions)
            nstates, r, d = env.step(states, actions)
            rews  = rews + jnp.where(dones, 0.0, r)
            dones = dones | d
            return (nstates, rews, dones), None

        (states, rews, _), _ = lax.scan(step, (states, rews, dones), None, length=max_steps)
        return jnp.mean(rews)
    return _rollout

def get_slimevolley_rollout(topology: Topology, n_episodes: int):
    """Get or create a cached rollout function for this topology and n_episodes.
    
    This factory function caches compiled rollout functions to avoid recompilation
    when evaluating multiple genomes with the same topology.
    """
    # Get/compile the topology-specific policy once; reuse across calls
    topology_sig = topology.signature()
    policy_apply = _policy_cache.get(topology_sig)
    if policy_apply is None:
        policy_apply = topology2policy(topology)
        _policy_cache[topology_sig] = policy_apply

    # Get/compile the rollout function once; reuse across calls
    key = (topology_sig, n_episodes)
    fn = _rollout_cache.get(key)
    if fn is None:
        fn = make_weighted_rollout(policy_apply, n_episodes)
        _rollout_cache[key] = fn
    return fn

def evaluate_genome_slimevolley(genome: Genome, key: jax.Array, n_episodes: int) -> float:
    # 1) Extract static structure (topology) + dynamic weights
    topology, weights = build_topology_and_weights(genome)
    
    # 2) Create a rollout that closes over (policy, weights)
    rollout_fn = get_slimevolley_rollout(topology, n_episodes)
    
    # 3) Run the rollout
    reward = rollout_fn(key, weights)
    return float(reward)

if __name__ == "__main__":
    # SlimeVolley has 12 inputs (state observation) and 3 outputs (actions)
    N_INPUTS = 12
    N_OUTPUTS = 3
    N_GENERATIONS = 4
    N_EPISODES = 16

    config = NEATConfig(pop_size=128, delta_threshold=1.0, w_init_std=0.5)
    key = jr.PRNGKey(42)
    evaluator = VectorizedEvaluator(get_slimevolley_rollout, n_episodes=N_EPISODES)
    
    print("Starting SlimeVolley evolution...")
    import time
    start_time = time.time()
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
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    
    best_genome_idx = int(jnp.argmax(jnp.array(result.population.fitness)))
    best_genome = result.population.genomes[best_genome_idx]
    best_fit = float(result.population.fitness[best_genome_idx])
    print(f"Best fitness during training: {best_fit:.2f}")
    
    # Test with more episodes for better estimate
    print(f"\nTesting best genome with {N_EPISODES} episodes...")
    test_score = evaluate_genome_slimevolley(
        best_genome, jr.PRNGKey(321), n_episodes=N_EPISODES
    )
    print(f"Average test reward: {test_score:.2f}")
    
    with open('slimevolley_history_2.json', 'w') as f:
        data = {
            "history": [h.to_dict() for h in result.history],
            "time_taken": end_time - start_time,
        }
        json.dump(data, f)