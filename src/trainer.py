from dataclasses import dataclass
from typing import List, Optional

import jax
import jax.random as jr

from .population import Population, NEATConfig
from .evaluator import Evaluator


@dataclass
class EvolutionMetrics:
    generation: int
    best_fitness: float
    avg_fitness: float
    num_species: int
    mean_parameters: float


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
        )
        history.append(metrics)

        if verbose:
            print(f"Gen {gen:03d} | Best Fitness: {best_fitness:6.4f} | Avg Fitness: {avg_fitness:6.4f} | Species: {num_species} | Mean Parameters: {mean_parameters:6.4f}")

        if gen < generations - 1:
            population.reproduce()

    return EvolutionResult(population=population, history=history)
