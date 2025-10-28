"""Backprop NEAT example on 2D classification tasks."""

import jax
import jax.numpy as jnp
import jax.random as jr
from functools import partial
from typing import Tuple
from src.population import NEATConfig
from src.topology import build_topology_and_weights, topology2policy
from src.trainer import evolve
from src.evaluator import SimpleEvaluator
from src.genome import Genome


def generate_circles_dataset(key: jax.Array, n_samples: int = 500, noise: float = 0.1):
    """Generate concentric circles dataset (inner circle = class 0, outer = class 1)."""
    key, k1, k2, k3, k4, k5 = jr.split(key, 6)
    
    # Inner circle
    n_inner = n_samples // 2
    r_inner = jr.uniform(k1, (n_inner,), minval=0.0, maxval=0.3)
    theta_inner = jr.uniform(k2, (n_inner,), minval=0.0, maxval=2*jnp.pi)
    x_inner = r_inner * jnp.cos(theta_inner)
    y_inner = r_inner * jnp.sin(theta_inner)
    labels_inner = jnp.zeros(n_inner)
    
    # Outer circle
    n_outer = n_samples - n_inner
    r_outer = jr.uniform(k3, (n_outer,), minval=0.6, maxval=1.0)
    theta_outer = jr.uniform(k4, (n_outer,), minval=0.0, maxval=2*jnp.pi)
    x_outer = r_outer * jnp.cos(theta_outer)
    y_outer = r_outer * jnp.sin(theta_outer)
    labels_outer = jnp.ones(n_outer)
    
    # Combine and add noise
    X = jnp.stack([
        jnp.concatenate([x_inner, x_outer]),
        jnp.concatenate([y_inner, y_outer])
    ], axis=1)
    X = X + jr.normal(k5, X.shape) * noise
    y = jnp.concatenate([labels_inner, labels_outer])
    
    return X, y


def binary_cross_entropy_loss(predictions: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """Binary cross-entropy loss.
    
    Args:
        predictions: [N, 1] raw network outputs
        targets: [N] binary labels (0 or 1)
    """
    # Apply sigmoid to get probabilities
    probs = jax.nn.sigmoid(predictions.squeeze())
    # Clip to avoid log(0)
    probs = jnp.clip(probs, 1e-7, 1.0 - 1e-7)
    # Binary cross-entropy
    loss = -jnp.mean(targets * jnp.log(probs) + (1 - targets) * jnp.log(1 - probs))
    return loss


def evaluate_genome(genome: Genome, key: jax.Array, test_data: Tuple[jnp.ndarray, jnp.ndarray]) -> float:
    """Evaluate genome accuracy on test data."""
    X_test, y_test = test_data
    
    # Get predictions
    topology, weights = build_topology_and_weights(genome)
    policy = topology2policy(topology)
    raw_outputs = policy(weights, X_test)
    predictions = jax.nn.sigmoid(raw_outputs.squeeze())
    predicted_labels = (predictions > 0.5).astype(jnp.float32)
    
    # Calculate accuracy
    accuracy = jnp.mean(predicted_labels == y_test).item()
    fitness = accuracy - 0.1 * jnp.log(1 + genome.num_parameters)
    return float(fitness)

if __name__ == "__main__":
    # Generate dataset
    key = jr.PRNGKey(42)
    key, train_key, test_key, val_key = jr.split(key, 4)
    X_train, y_train = generate_circles_dataset(train_key, n_samples=400, noise=0.05)
    X_test, y_test = generate_circles_dataset(test_key, n_samples=200, noise=0.05)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Configure NEAT with backprop
    config = NEATConfig(
        pop_size=50,
        delta_threshold=8.0,
        enable_backprop=True,
        backprop_steps=100,
        backprop_lr=0.01,
        backprop_batch_size=128,
    )
    GENERATIONS = 100
    
    # Create evaluator (evaluates on test set)
    test_data = (X_test, y_test)
    eval_fn = partial(evaluate_genome, test_data=test_data)
    evaluator = SimpleEvaluator(eval_fn)
    
    # Training data for backprop
    train_data = (X_train, y_train)
    
    # Run evolution
    print("Starting evolution with backprop...")
    result = evolve(
        n_inputs=2,
        n_outputs=1,
        evaluator=evaluator,
        key=key,
        config=config,
        generations=GENERATIONS,
        add_bias=True,
        verbose=True,
        loss_fn=binary_cross_entropy_loss,
        train_data=train_data,
    )
    
    print("Evolution Complete!")
    
    # Analyze best genome
    fitness_history = [h.best_fitness for h in result.history]
    best_idx = int(jnp.argmax(jnp.array(fitness_history)))
    best_genome = result.history[best_idx].best_genome
    best_fitness = fitness_history[best_idx]
    
    # validate the best genome
    X_val, y_val = generate_circles_dataset(val_key, n_samples=200, noise=0.05)
    val_data = (X_val, y_val)
    val_accuracy = evaluate_genome(best_genome, val_key, val_data)
    print(f"Validation accuracy: {val_accuracy:.4f}") # 99.5%