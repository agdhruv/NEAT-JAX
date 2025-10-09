# src/evaluator.py
from abc import ABC, abstractmethod
from typing import List, Callable
import jax
import jax.numpy as jnp
import jax.random as jr
from .genome import Genome
from .topology import Topology

class Evaluator(ABC):
    """Abstract base class for genome evaluation strategies."""
    
    @abstractmethod
    def evaluate(self, genomes: List[Genome], key: jax.Array) -> List[float]:
        """Evaluate all genomes and return fitness scores."""
        pass


class SimpleEvaluator(Evaluator):
    """Sequential evaluation for simple/non-JAX environments."""
    
    def __init__(self, eval_fn: Callable[[Genome, jax.Array], float]):
        self.eval_fn = eval_fn
    
    def evaluate(self, genomes: List[Genome], key: jax.Array) -> List[float]:
        """Evaluate genomes one at a time using the provided eval_fn."""
        scores: List[float] = [0.0] * len(genomes)
        for i, g in enumerate(genomes):
            key, k = jr.split(key)
            scores[i] = float(self.eval_fn(g, k))
        return scores


class VectorizedEvaluator(Evaluator):
    """Vectorized evaluation for JAX environments (e.g., SlimeVolley)."""
    
    def __init__(self, rollout_factory: Callable, n_episodes: int = 3):
        """
        Args:
            rollout_factory: Function (topology, n_episodes) -> compiled eval fn (key, weights) -> mean_reward
            n_episodes: Number of episodes to evaluate per genome
        """
        self.rollout_factory = rollout_factory
        self.n_episodes = n_episodes
    
    def evaluate(self, genomes: List[Genome], key: jax.Array) -> List[float]:
        """Evaluate genomes using topology-based batching and vectorization."""
        from src.topology import build_topology_and_weights
        
        scores = [0.0] * len(genomes)
        
        # 1) Bucket genomes by topology
        buckets = {}  # sig -> {topology, idxs, weights}
        for i, g in enumerate(genomes):
            topology, w = build_topology_and_weights(g)
            sig = topology.signature()
            if sig not in buckets:
                buckets[sig] = {"topology": topology, "idxs": [i], "weights": [w]}
            else:
                buckets[sig]["idxs"].append(i)
                buckets[sig]["weights"].append(w)
        
        # 2) Evaluate each bucket in parallel
        import tqdm
        for sig, data in tqdm.tqdm(buckets.items(), desc="Evaluating buckets"):
            topology = data["topology"]
            idxs = data["idxs"]
            W = jnp.stack(data["weights"])  # [G, M] - G genomes, M weights
            G = W.shape[0]
            
            # Keys for each genome in this bucket
            key, sub = jr.split(key)
            keys = jr.split(sub, G)
            
            # Get compiled evaluation function for this topology
            eval_one = self.rollout_factory(topology, self.n_episodes)
            
            # Single-device vectorization (vmap over genomes in this bucket)
            batched = jax.vmap(eval_one, in_axes=(0, 0))
            bucket_scores = batched(keys, W)
            
            # Store scores for this bucket
            bucket_scores = bucket_scores.tolist()
            for j, s in zip(idxs, bucket_scores):
                scores[j] = float(s)
        
        return scores