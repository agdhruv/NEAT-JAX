"""Backpropagation utilities for NEAT genomes."""

from typing import Callable, Tuple, Optional
import jax
import jax.numpy as jnp
from .genome import Genome
from .topology import build_topology_and_weights, topology2policy
import optax


def optimize_weights(
    genome: Genome,
    loss_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    data: Tuple[jnp.ndarray, jnp.ndarray],
    n_steps: int,
    lr: float,
    batch_size: Optional[int] = None,
) -> None:
    """Optimize genome weights via gradient descent.
    
    Args:
        genome: NEAT genome to optimize (modified in-place)
        loss_fn: Loss function (predictions, targets) -> scalar loss
        data: Tuple of (inputs, targets) for training
        n_steps: Number of gradient descent steps
        lr: Learning rate
        batch_size: Batch size for minibatch training
    """
    # Extract static topology and dynamic weights
    topology, weights = build_topology_and_weights(genome)
    
    # Get differentiable forward pass for this topology
    forward_fn = topology2policy(topology)
    
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(weights)
    
    # Define loss function over weights
    def compute_loss(w: jnp.ndarray, batch_inputs: jnp.ndarray, batch_targets: jnp.ndarray) -> jnp.ndarray:
        predictions = forward_fn(w, batch_inputs)
        return loss_fn(predictions, batch_targets)
    
    @jax.jit
    def step(w: jnp.ndarray, state, batch_inputs: jnp.ndarray, batch_targets: jnp.ndarray):
        """Single optimization step (JIT-compiled)."""
        loss, grads = jax.value_and_grad(compute_loss)(w, batch_inputs, batch_targets)
        updates, new_state = optimizer.update(grads, state)
        new_w = optax.apply_updates(w, updates)
        return jnp.array(new_w), new_state, loss
    
    # Training loop
    inputs, targets = data
    n_samples = inputs.shape[0]
    
    if batch_size is None or batch_size >= n_samples:
        # Full batch training
        for _ in range(n_steps):
            weights, opt_state, loss = step(weights, opt_state, inputs, targets)
    else:
        # Minibatch training
        import jax.random as jr
        key = jr.PRNGKey(0)  # Fixed seed for reproducibility
        
        for _ in range(n_steps):
            # Random minibatch
            key, subkey = jr.split(key)
            indices = jr.choice(subkey, n_samples, shape=(batch_size,), replace=False)
            batch_inputs = inputs[indices]
            batch_targets = targets[indices]
            
            weights, opt_state, loss = step(weights, opt_state, batch_inputs, batch_targets)
    
    # Update genome's connection weights in-place
    # Note: must be done in exact order as the weights were initialized in build_topology_and_weights() in topology.py
    enabled_connections = [c for c in genome.connections.values() if c.enabled]
    for i, conn in enumerate(enabled_connections):
        conn.weight = float(weights[i])

