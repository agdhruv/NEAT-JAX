from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import jax
import jax.random as jr
import jax.numpy as jnp

from .innovation import InnovationTracker

# Node types
INPUT, HIDDEN, OUTPUT, BIAS = 0, 1, 2, 3


@dataclass
class NodeGene:
    id: int
    type: int  # INPUT, HIDDEN, OUTPUT, BIAS
    level: int  # feed-forward level (0 for input/bias)
    
    def copy(self) -> NodeGene:
        return NodeGene(
            id=self.id,
            type=self.type,
            level=self.level)


@dataclass
class ConnectionGene:
    innovation: int
    in_node: int
    out_node: int
    weight: float
    enabled: bool = True
    
    def copy(self) -> ConnectionGene:
        return ConnectionGene(
            innovation=self.innovation,
            in_node=self.in_node,
            out_node=self.out_node,
            weight=self.weight,
            enabled=self.enabled)


@dataclass
class Genome:
    """
    Represents a NEAT genome containing nodes and connections.

    In NEAT (NeuroEvolution of Augmenting Topologies), a genome encodes the
    structure and weights of a neural network. It consists of:
    - Node genes: defining the neurons in the network
    - Connection genes: defining weighted connections between neurons

    Each connection has an innovation number for historical marking,
    which is crucial for crossover operations in NEAT.
    """
    nodes: Dict[int, NodeGene] = field(default_factory=dict)
    connections: Dict[int, ConnectionGene] = field(default_factory=dict)
    
    def __hash__(self) -> int:
        """
        Computes a hash based on the genome's structure (topology),
        ignoring weights. This allows JAX to cache compiled functions
        for structurally identical genomes.
        """
        # Create a canonical, hashable representation of nodes
        node_items = sorted(self.nodes.items())
        node_tuple = tuple((nid, n.type, n.level) for nid, n in node_items)
        
        # Create a canonical, hashable representation of connections' structure
        conn_items = sorted(self.connections.items())
        conn_tuple = tuple((innov, c.in_node, c.out_node, c.enabled) for innov, c in conn_items)
        
        return hash((node_tuple, conn_tuple))
    
    def copy(self) -> "Genome":
        return Genome(
            nodes={nid: node.copy() for nid, node in self.nodes.items()},
            connections={innov: conn.copy() for innov, conn in self.connections.items()}
        )
    
    @staticmethod
    def from_initial_feedforward(
        n_inputs: int,
        n_outputs: int,
        tracker: InnovationTracker,
        *,
        add_bias: bool = True,
        key: jax.Array,
        w_init_std: float = 1.0,
    ) -> "Genome":
        """
        Create an initial fully-connected feed-forward network for NEAT evolution.

        This creates the starting topology that the NEAT algorithms will
        begin with: a direct connection from each input (and optional bias) to
        each output node. This allows the evolution to start simple and complexify
        through structural mutations.

        Args:
            n_inputs: Number of input nodes (excluding bias)
            n_outputs: Number of output nodes
            add_bias: Whether to include a bias node (recommended for most tasks)
            key: JAX random key for weight initialization
            w_init_std: Standard deviation for random weight initialization

        Returns:
            Tuple containing:
            - genome: The initialized Genome instance
            - input_ids: List of input node IDs
            - output_ids: List of output node IDs
            - bias_id: Bias node ID (None if add_bias=False)

        Notes:
            - Input and bias nodes are assigned level 0 (no computation)
            - Output nodes are assigned level 1 (one computation step)
            - All connections are initially enabled
            - Innovation numbers start from 0 and increment sequentially
        """
        g = Genome()

        # Create input nodes at level 0 (feed-forward level)
        input_node_ids = []
        for _ in range(n_inputs):
            node_id = tracker.allocate_node()
            g.nodes[node_id] = NodeGene(id=node_id, type=INPUT, level=0)
            input_node_ids.append(node_id)

        # Create optional bias node at level 0
        # Bias nodes provide a constant input value (typically 1.0)
        bias_node_id = None
        if add_bias:
            bias_node_id = tracker.allocate_node()
            g.nodes[bias_node_id] = NodeGene(id=bias_node_id, type=BIAS, level=0)

        # Create output nodes at level 1
        output_node_ids = []
        for _ in range(n_outputs):
            node_id = tracker.allocate_node()
            g.nodes[node_id] = NodeGene(id=node_id, type=OUTPUT, level=1)
            output_node_ids.append(node_id)

        # Create fully-connected topology: all inputs/bias -> all outputs
        # This is the standard NEAT starting point before complexification
        srcs = input_node_ids + ([bias_node_id] if bias_node_id is not None else [])
        k1, key = jr.split(key)
        W = jr.normal(k1, (len(srcs), len(output_node_ids))) * w_init_std

        # Create connection genes with innovation tracking
        # Innovation numbers enable proper crossover in NEAT reproduction
        for j, out_node_id in enumerate(output_node_ids):
            for i, in_node_id in enumerate(srcs):
                innovation = tracker.allocate_connection(in_node_id, out_node_id)
                g.connections[innovation] = ConnectionGene(
                    innovation=innovation,
                    in_node=in_node_id,
                    out_node=out_node_id,
                    weight=float(W[i, j]),  # Random initial weight
                    enabled=True,  # All connections start enabled
                )

        return g

    # ----------------- Mutations -----------------
    def mutate_weights(
        self,
        key: jax.Array,
        sigma: float = 0.5,
        p_reset: float = 0.1,
        w_init_std: float = 1.0,
    ) -> None:
        """Mutate connection weights using Gaussian perturbation or reset.

        Each connection weight is either:
        1. Perturbed by Gaussian noise (probability 1-p_reset)
        2. Reset to a new random value (probability p_reset)

        Args:
            key: JAX random key for reproducible randomness
            sigma: Standard deviation for Gaussian weight perturbation
            p_reset: Probability of resetting each weight to random value
            w_init_std: Standard deviation for new random weights when resetting
        """
        if not self.connections:
            return

        innovations = list(self.connections.keys())  # list of innovation numbers (i.e., list of ids for each weight)
        k1, k2 = jr.split(key)
        noise = jr.normal(k1, (len(innovations),)) * sigma
        reset_mask = jr.bernoulli(k2, p=p_reset, shape=(len(innovations),))

        # separate keys for resets to avoid correlation
        reset_keys = jr.split(k2, len(innovations))

        for idx, innovation in enumerate(innovations):
            if reset_mask[idx]:
                self.connections[innovation].weight = float(jr.normal(reset_keys[idx]) * w_init_std)
            else:
                self.connections[innovation].weight += float(noise[idx])

    def mutate_add_connection(
        self, key: jax.Array, tracker: InnovationTracker, w_init_std: float = 1.0
    ):
        """Add a new connection to the genome (only from lower level to higher level - no recurrent connections)

        Args:
            key: JAX random key for reproducible randomness
            tracker: InnovationTracker to provide innovation IDs for new connections
            w_init_std: Standard deviation for new random weights when resetting
        """
        existing_conns = {(c.in_node, c.out_node) for c in self.connections.values()}  # includes disabled connections
        node_ids = list(self.nodes.keys())

        # candidates for new connections
        candidates = []
        for in_node_id in node_ids:
            for out_node_id in node_ids:
                if in_node_id == out_node_id or (in_node_id, out_node_id) in existing_conns:
                    continue
                level_in = self.nodes[in_node_id].level
                level_out = self.nodes[out_node_id].level
                if level_in >= level_out:
                    continue
                candidates.append((in_node_id, out_node_id))

        if not candidates:
            return

        # randomly select a candidate
        k1, k2 = jr.split(key)
        idx = int(jr.randint(k1, (), 0, len(candidates)))
        in_id, out_id = candidates[idx]

        # allocate this connection a new innovation number
        # internally, the tracker will check if this connection has been seen in this generation
        innovation = tracker.allocate_connection(in_id, out_id)
        w = float(jr.normal(k2) * w_init_std)
        self.connections[innovation] = ConnectionGene(
            innovation=innovation,
            in_node=in_id,
            out_node=out_id,
            weight=w,
            enabled=True,
        )

    def _bump_levels_from(self, start_level: int) -> None:
        for n in self.nodes.values():
            if n.level >= start_level:
                n.level += 1

    def mutate_add_node(self, key: jax.Array, tracker: InnovationTracker):
        """
        Add a new hidden node to the genome by splitting an existing connection.

        Args:
            key: JAX random key for reproducible randomness
            tracker: InnovationTracker to provide innovation IDs for new connections
        """
        enabled = [c for c in self.connections.values() if c.enabled]
        if not enabled:
            return

        k_sel, key = jr.split(key)

        # pick and disable an enabled connection
        c = enabled[int(jr.randint(k_sel, (), 0, len(enabled)))]
        c.enabled = False  # disable old connection

        in_node = self.nodes[c.in_node]
        out_node = self.nodes[c.out_node]

        # ensure a level gap so new node can sit strictly between
        if out_node.level <= in_node.level + 1:
            self._bump_levels_from(out_node.level)

        # split the connection - internally, the tracker will check if this split has been done before in this generation
        new_node_id, in_innov, out_innov = tracker.split_connection(c.innovation)

        # create the new hidden node
        new_node = NodeGene(id=new_node_id, type=HIDDEN, level=in_node.level + 1)
        self.nodes[new_node_id] = new_node

        # add the two replacement connections
        # NEAT paper convention: in→new weight = 1.0, new→out weight = old c.weight
        self.connections[in_innov] = ConnectionGene(
            innovation=in_innov,
            in_node=c.in_node,
            out_node=new_node_id,
            weight=1.0,
            enabled=True,
        )
        self.connections[out_innov] = ConnectionGene(
            innovation=out_innov,
            in_node=new_node_id,
            out_node=c.out_node,
            weight=c.weight,
            enabled=True,
        )

    def mutate_toggle_connection(self, key: jax.Array):
        """
        Toggle the enabled state of a random connection in the genome.

        Args:
            key: JAX random key for reproducible randomness
        """
        if not self.connections:
            return
        idx = int(jr.randint(key, (), 0, len(self.connections)))
        c = list(self.connections.values())[idx]
        c.enabled = not c.enabled
    
    def compatibility_distance(self, other: "Genome", c1: float = 1.0, c2: float = 1.0, c3: float = 0.4) -> float:
        a = self.connections
        b = other.connections
        innovation_a = sorted(a.keys())
        innovation_b = sorted(b.keys())
        set_a = set(innovation_a)
        set_b = set(innovation_b)
        max_a = innovation_a[-1] if innovation_a else -1
        max_b = innovation_b[-1] if innovation_b else -1

        matching = set_a & set_b
        disjoint = {k for k in (set_a ^ set_b) if (k <= max_a and k <= max_b)}
        excess = {k for k in (set_a ^ set_b) if (k > max_a or k > max_b)}

        w_diff = 0.0
        if matching:
            w_diff = sum(abs(a[k].weight - b[k].weight) for k in matching) / len(matching)

        # N is the normalizer (size of larger genome); NEAT uses N=1 for small genomes
        N = max(len(a), len(b))
        N = 1 if N < 20 else N
        return c1 * (len(excess) / N) + c2 * (len(disjoint) / N) + c3 * w_diff
    
    def crossover(self, other: "Genome", fitness_self: float, fitness_other: float, key: jax.Array) -> "Genome":
        k_tie, k2_pick, k3_dis, key = jr.split(key, 4)
        
        a, b = self, other
        if fitness_other > fitness_self:
            a, b = other, self  # a is more fit
        elif fitness_other == fitness_self:
            if jr.bernoulli(k_tie, 0.5):  # tie-breaker: choose a at random
                a, b = other, self
        
        child = Genome()
        
        # nodes: inherit all from both parents, preferring fitter parent (a)
        for node_id in set(a.nodes) | set(b.nodes):
            child.nodes[node_id] = (a.nodes.get(node_id) or b.nodes[node_id]).copy()
        
        # connections
        ids_a, ids_b = set(a.connections.keys()), set(b.connections.keys())
        matching = sorted(ids_a & ids_b)
        non_matching = ids_a ^ ids_b
        
        # matching connections: pick randomly from either parent
        k2, key = jr.split(key)
        if matching:
            pick_keys = jr.split(k2_pick, len(matching))
            dis_keys = jr.split(k3_dis, len(matching))
            for i, innovation in enumerate(matching):
                pick_a = jr.bernoulli(pick_keys[i], 0.5)
                src = a.connections[innovation] if pick_a else b.connections[innovation]
                child.connections[innovation] = src.copy()
                
                # if the innovation was disabled in either, keep it disabled in child with 75% probability
                either_disabled = (not a.connections[innovation].enabled) or (not b.connections[innovation].enabled)
                if either_disabled:
                    child.connections[innovation].enabled = bool(jr.bernoulli(dis_keys[i], 0.25))
        
        # non-matching connections: pick from fitter parent
        for innovation in non_matching:
            if innovation in a.connections:
                child.connections[innovation] = a.connections[innovation].copy()
        
        # remove connections referencing nodes not in the child
        for c in list(child.connections.values()):
            if c.in_node not in child.nodes or c.out_node not in child.nodes:
                del child.connections[c.innovation]
        
        return child

def _phenotype_forward(genome: Genome, x: jax.Array) -> jax.Array:
    """
    A pure, JIT-compilable function to execute the forward pass of a genome's network.
    """
    bias_ids = [n.id for n in genome.nodes.values() if n.type == BIAS]
    bias_id = bias_ids[0] if bias_ids else None
    input_ids = sorted([n.id for n in genome.nodes.values() if n.type == INPUT])
    output_ids = sorted([n.id for n in genome.nodes.values() if n.type == OUTPUT])
    
    # by_level: dict of level -> list of node ids in that level
    by_level: Dict[int, List[int]] = {}
    for node_id, node in genome.nodes.items():
        by_level.setdefault(node.level, []).append(node_id)
    
    levels = sorted(by_level.keys())
    
    # for each target node, a list of its enabled incoming edges (in_node_id, weight)
    incoming: Dict[int, List[Tuple[int, float]]] = {}
    for c in genome.connections.values():
        if c.enabled:
            incoming.setdefault(c.out_node, []).append((c.in_node, c.weight))
    
    exec_levels = [level for level in levels if level > 0]
    
    # activation functions
    act_hidden = jnp.tanh
    act_output = jnp.tanh
    
    # set inputs (will also store outputs from hidden and output layers)
    vals: Dict[int, jax.Array] = {id: x[i] for i, id in enumerate(input_ids)}
    if bias_id is not None:
        vals[bias_id] = jnp.array(1.0, dtype=x.dtype)
    
    for level in exec_levels:
        for node_id in by_level[level]:
            inc = incoming.get(node_id, [])
            s = jnp.array(0.0, dtype=x.dtype)
            for in_id, w in inc:
                s += jnp.array(w, dtype=x.dtype) * vals[in_id]
            
            activation = act_output if genome.nodes[node_id].type == OUTPUT else act_hidden
            vals[node_id] = activation(s)
    
    return jnp.stack([vals[nid] for nid in output_ids], axis=0)

# Create the JIT-compiled version of the function, marking the genome as a static argument.
phenotype_forward = jax.jit(_phenotype_forward, static_argnames="genome")