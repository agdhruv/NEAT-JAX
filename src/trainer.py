from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Callable, Tuple

import jax
import jax.random as jr
import jax.numpy as jnp

from .population import Population, NEATConfig
from .evaluator import Evaluator
from .genome import Genome


@dataclass
class EvolutionMetrics:
    generation: int
    best_fitness: float
    avg_fitness: float
    num_species: int
    mean_parameters: float
    best_genome: Genome

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "generation": self.generation,
            "best_fitness": self.best_fitness,
            "avg_fitness": self.avg_fitness,
            "num_species": self.num_species,
            "mean_parameters": self.mean_parameters,
            "best_genome": self.best_genome.to_dict()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvolutionMetrics":
        """Create from dictionary."""
        return cls(
            generation=data["generation"],
            best_fitness=data["best_fitness"],
            avg_fitness=data["avg_fitness"],
            num_species=data["num_species"],
            mean_parameters=data["mean_parameters"],
            best_genome=Genome.from_dict(data["best_genome"])
        )


@dataclass
class EvolutionResult:
    population: Population
    history: List[EvolutionMetrics]


def evolve(
    n_inputs: int,
    n_outputs: int,
    evaluator: Evaluator,
    *,
    key: jax.Array,
    config: Optional[NEATConfig] = None,
    generations: int = 100,
    add_bias: bool = True,
    verbose: bool = True,
    loss_fn: Optional[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]] = None,
    train_data: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
) -> EvolutionResult:
    """Run the NEAT evolutionary loop for a fixed number of generations.

    Args:
        n_inputs: Number of input nodes for initial topology.
        n_outputs: Number of output nodes for initial topology.
        evaluator: Evaluator that maps (genome, key) -> fitness (higher is better).
        key: JAX PRNGKey.
        config: NEAT configuration. If None, defaults are used.
        generations: Number of generations to evolve.
        add_bias: Whether to include a bias node in initial topology.
        verbose: If True, print per-generation metrics.
        loss_fn: Optional loss function for backprop (predictions, targets) -> scalar.
        train_data: Optional training data (inputs, targets) for backprop.

    Returns:
        EvolutionResult containing final population, history, and best genome.
    """
    cfg = config or NEATConfig()

    key, pop_key = jr.split(key)
    population = Population.from_initial_feedforward(
        n_inputs=n_inputs,
        n_outputs=n_outputs,
        key=pop_key,
        config=cfg,
        add_bias=add_bias,
    )

    history: List[EvolutionMetrics] = []

    for gen in range(generations):
        # Optional backpropagation step before evaluation
        if cfg.enable_backprop:
            if loss_fn is None or train_data is None:
                raise ValueError("loss_fn and train_data must be provided when enable_backprop=True")
            
            from .backprop import optimize_weights
            for genome in population.genomes:
                optimize_weights(
                    genome,
                    loss_fn,
                    train_data,
                    cfg.backprop_steps,
                    cfg.backprop_lr,
                    cfg.backprop_batch_size,
                )
        
        population.evaluate(evaluator)

        assert population.fitness, Exception("Fitness values should be defined for all genomes (even if they are zero)")
        # Collect metrics
        best_fitness = max(population.fitness)
        avg_fitness = sum(population.fitness) / len(population.fitness)
        num_species = len(population.species)
        mean_parameters = sum(g.num_parameters for g in population.genomes) / len(population.genomes)

        metrics = EvolutionMetrics(
            generation=gen,
            best_fitness=float(best_fitness),
            avg_fitness=float(avg_fitness),
            num_species=int(num_species),
            mean_parameters=float(mean_parameters),
            best_genome=population.genomes[int(jnp.argmax(jnp.array(population.fitness)))],
        )
        history.append(metrics)

        if verbose:
            print(f"Gen {gen:03d} | Best Fitness: {best_fitness:6.4f} | Avg Fitness: {avg_fitness:6.4f} | Species: {num_species} | Mean Parameters: {mean_parameters:6.4f}")

        if gen < generations - 1:
            population.reproduce()

    return EvolutionResult(population=population, history=history)
