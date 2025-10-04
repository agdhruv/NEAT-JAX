import jax
import jax.numpy as jnp
import jax.random as jr
from src.population import NEATConfig
from src.genome import Genome, phenotype_forward
from src.trainer import evolve
from functools import partial

# --- 1. Define the Target Function ---
# A moderately complex function for the network to learn.
# It takes a 3-element vector and returns a 5-element vector.
@jax.jit
def target_function(inputs: jax.Array) -> jax.Array:
    """
    Inputs: jnp.array([x, y, z])
    Outputs: 5-element jnp.array
    """
    x, y, z = inputs[0], inputs[1], inputs[2]
    
    o1 = jnp.sin(jnp.pi * x) * jnp.cos(jnp.pi * y)
    o2 = (x**2 + y**2 + z**2) / 3.0 # Scaled to keep output range reasonable
    o3 = jnp.tanh(x + y - z)
    o4 = jnp.clip(x * y * z, -1.0, 1.0)
    o5 = jnp.where(z > 0, x, y)
    
    return jnp.array([o1, o2, o3, o4, o5])

# --- 2. Create the Evaluation Function ---
# This function determines the "fitness" of a single genome.
def evaluate_genome(genome: Genome, key: jax.Array, batch_size: int = 128) -> float:
    """
    Calculates the fitness of a genome by testing its network against the target function.
    """
    # Generate a batch of random input data
    key, k = jr.split(key)
    inputs = jr.uniform(k, (batch_size, 3), minval=-1.0, maxval=1.0)
    
    # Calculate the true target outputs
    targets = jax.vmap(target_function)(inputs)
    
    # Vectorize the pure forward function.
    # in_axes=(None, 0) means: don't map over the first arg (genome), 
    # but do map over the second arg (inputs).
    batched_forward = jax.vmap(phenotype_forward, in_axes=(None, 0))
    predictions = batched_forward(genome, inputs)
    
    mse = jnp.mean((predictions - targets)**2)
    fitness = -mse
    return float(fitness)

# --- 3. Set Up and Run the NEAT Algorithm ---
def main():
    N_INPUTS = 3
    N_OUTPUTS = 5
    N_GENERATIONS = 100
    config = NEATConfig(delta_threshold=1.0, pop_size=100)
    
    key = jr.PRNGKey(42)
    eval_fn = partial(evaluate_genome, batch_size=128)

    print("Starting evolution...")
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
    
    print("\nEvolution finished. Testing the best genome...")
    
    # Identify best genome after evolution
    best_idx = int(jnp.argmax(jnp.array(result.population.fitness)))
    best_genome = result.population.genomes[best_idx]
    best_fit = float(result.population.fitness[best_idx])
    
    print(f"Best genome had {len(best_genome.nodes)} nodes and {len(best_genome.connections)} connections.")
    
    # Test the best genome
    key, test_key = jr.split(key)
    test_inputs = jr.uniform(test_key, shape=(5, 3), minval=-1.0, maxval=1.0)
    
    test_targets = jax.vmap(target_function)(test_inputs)
    test_predictions = jax.vmap(phenotype_forward, in_axes=(None, 0))(best_genome, test_inputs)
    
    for i in range(len(test_inputs)):
        print(f"\nInput: {test_inputs[i]}")
        print(f"  Target:     {test_targets[i]}")
        print(f"  Prediction: {test_predictions[i]}")
        
    final_mse = jnp.mean((test_predictions - test_targets)**2)
    print(f"\nFinal MSE on test data: {final_mse:.6f}")

if __name__ == "__main__":
    main()