import jax
import jax.numpy as jnp
import jax.random as jr
import jax.lax as lax
from evojax.task.slimevolley import SlimeVolley
from src.structure import Plan
from src.structure import make_policy

max_steps = 500
env = SlimeVolley(test=True, max_steps=max_steps)

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

# Cache compiled policies per-plan so we don't rebuild the function each call
_policy_cache = {}

def _get_policy(plan: Plan):
    sig = plan.signature()
    if sig not in _policy_cache:
        _policy_cache[sig] = make_policy(plan)   # jitted fn: (weights, obs[E,Di]) -> actions[E,Do]
    return _policy_cache[sig]

_rollout_cache = {}   # (plan_sig, n_episodes) -> compiled fn