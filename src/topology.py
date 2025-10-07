from dataclasses import dataclass
from typing import Tuple
import jax
import jax.numpy as jnp

from .genome import INPUT, HIDDEN, OUTPUT, BIAS, Genome

@dataclass(frozen=True)
class Topology:
    input_idx: jnp.ndarray    # [Ni]
    output_idx: jnp.ndarray   # [No]
    bias_idx: int             # -1 if none
    src_idx: jnp.ndarray      # [M] edge sources (0..N-1)
    dst_idx: jnp.ndarray      # [M] edge targets (0..N-1)
    levels: jnp.ndarray       # [N] per-node level (ints)
    level_ids: jnp.ndarray    # [L] unique sorted levels
    n_nodes: int              # total nodes N
    
    def signature(self) -> Tuple:
        return (
            tuple(self.levels.tolist()),
            tuple(self.src_idx.tolist()), tuple(self.dst_idx.tolist()),
            tuple(self.input_idx.tolist()), tuple(self.output_idx.tolist()),
            int(self.bias_idx), int(self.n_nodes)
        )

def build_topology_and_weights(genome: Genome) -> Tuple[Topology, jnp.ndarray]:
    """Extract static network structure and dynamic weights from genome.
    
    Converts a NEAT genome into a Topology (static topology) and weight array
    (dynamic parameters) for efficient JAX compilation and execution.
    
    Returns:
        Topology: Static network structure with node indices and connections
        jnp.ndarray: Dynamic connection weights for enabled edges
    """
    # Map node-id -> 0..N-1
    node_ids = sorted(genome.nodes.keys())
    id2i = {nid: i for i, nid in enumerate(node_ids)}

    input_idx  = [id2i[nid] for nid, nd in genome.nodes.items() if nd.type == INPUT]
    output_idx = [id2i[nid] for nid, nd in genome.nodes.items() if nd.type == OUTPUT]
    bias_idx_l = [id2i[nid] for nid, nd in genome.nodes.items() if nd.type == BIAS]
    bias_idx = bias_idx_l[0] if bias_idx_l else -1

    # Enabled edges
    edges = [c for c in genome.connections.values() if c.enabled]
    src_idx = jnp.array([id2i[c.in_node]  for c in edges], dtype=jnp.int32)
    dst_idx = jnp.array([id2i[c.out_node] for c in edges], dtype=jnp.int32)
    weights = jnp.array([c.weight for c in edges], dtype=jnp.float32) # Dynamic

    # Levels
    levels = jnp.array([genome.nodes[nid].level for nid in node_ids], dtype=jnp.int32)
    level_ids = jnp.unique(levels, sorted=True)

    topology = Topology(
        input_idx=jnp.array(sorted(input_idx), dtype=jnp.int32),
        output_idx=jnp.array(sorted(output_idx), dtype=jnp.int32),
        bias_idx=int(bias_idx),
        src_idx=src_idx,
        dst_idx=dst_idx,
        levels=levels,
        level_ids=level_ids,
        n_nodes=len(node_ids),
    )
    return topology, weights

def topology2policy(topology: Topology):
    """Create a JIT-compiled policy function for a given network topology.
    
    This function compiles once per unique topology (network structure) and returns
    a JIT-compiled function that can be called repeatedly with different weights
    and observations.
    
    Args:
        topology: Network structure specification containing node indices, connections,
              and topology information
              
    Returns:
        A JIT-compiled function with signature:
        (weights: jnp.ndarray, obs: jnp.ndarray) -> jnp.ndarray
        
        Where:
        - weights: [M] connection weights for all enabled edges
        - obs: [E, obs_dim] batch of observations (obs_dim == len(topology.input_idx))
        - returns: [E, act_dim] batch of actions (act_dim == len(topology.output_idx))
    """
    @jax.jit
    def apply(weights: jnp.ndarray, obs: jnp.ndarray) -> jnp.ndarray:
        """
        weights: [M]
        obs:     [E, obs_dim]     (obs_dim == len(topology.input_idx))
        returns: [E, act_dim]     (act_dim == len(topology.output_idx))
        """
        E = obs.shape[0]
        N = topology.n_nodes

        # Node activations
        vals = jnp.zeros((E, N), dtype=obs.dtype)
        vals = vals.at[:, topology.input_idx].set(obs)
        vals = jax.lax.cond(
            topology.bias_idx >= 0,
            lambda v: v.at[:, topology.bias_idx].set(1.0),
            lambda v: v,
            vals,
        )

        # Level-by-level feedforward using scatter-add
        def do_level(v: jnp.ndarray, lvl: jnp.ndarray):
            # Weighted sum into all dst nodes for this step
            weighted_values = v[:, topology.src_idx] * weights  # [E, M]
            pre = v.at[:, topology.dst_idx].add(weighted_values)
            # apply activation function to all hidden nodes
            is_output = jnp.zeros((N,), dtype=bool).at[topology.output_idx].set(True)
            is_input = jnp.zeros((N,), dtype=bool).at[topology.input_idx].set(True)
            is_input = jax.lax.cond(
                topology.bias_idx >= 0,
                lambda v: v.at[topology.bias_idx].set(True),
                lambda v: v,
                is_input,
            )
            activated = jnp.where(is_output[None, :] | is_input[None, :], pre, jnp.tanh(pre))
            # Only update nodes that are at this level; keep others as-is
            mask = (topology.levels == lvl)[None, :]  # [1, N]
            v = jnp.where(mask, activated, v)
            return v, None

        vals, _ = jax.lax.scan(do_level, vals, topology.level_ids)
        assert isinstance(vals, jnp.ndarray)
        actions = vals[:, topology.output_idx]
        return actions

    return apply

if __name__ == "__main__":
    from src.genome import Genome
    from src.draw import draw
    from src.innovation import InnovationTracker
    import jax.random as jr
    genome = Genome.from_initial_feedforward(3, 5, tracker=InnovationTracker(), key=jr.PRNGKey(0), add_bias=True, w_init_std=1.0)
    topology, weights = build_topology_and_weights(genome)
    policy = topology2policy(topology)
    obs = jnp.array([[1.0, 2.0, 3.0]])
    with jax.disable_jit(): # for debugging
        out = policy(weights, obs)
    print(out)
    draw(topology, weights, "topology.png")