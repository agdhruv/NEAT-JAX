from dataclasses import dataclass
from typing import List, Optional, Callable
from .innovation import InnovationTracker
from .genome import Genome
import jax
import jax.random as jr
import jax.numpy as jnp

@dataclass
class NEATConfig:
    pop_size: int = 100
    # speciation
    c1: float = 1.0
    c2: float = 1.0
    c3: float = 0.4
    delta_threshold: float = 3.0
    # reproduction
    elite_per_species: int = 1
    crossover_prob: float = 0.75
    # mutation
    p_mutate_weights: float = 0.9
    p_mutate_add_connection: float = 0.05
    p_mutate_add_node: float = 0.05
    p_mutate_toggle_connection: float = 0.01
    weight_sigma: float = 0.5
    weight_reset_prob: float = 0.1
    w_init_std: float = 1.0

@dataclass
class Species:
    representative: int
    members: List[int]

class Population:
    def __init__(self, genomes: List[Genome], tracker: InnovationTracker, key: jax.Array, config: NEATConfig):
        self.genomes = genomes
        self.tracker = tracker
        self.key = key
        self.config = config
        self.fitness: List[float] = [0.0] * len(genomes)
        self.species: List[Species] = []
        self.generation: int = 0

    @staticmethod
    def from_initial_feedforward(
        n_inputs: int,
        n_outputs: int,
        key: jax.Array,
        config: Optional[NEATConfig] = None,
        add_bias: bool = True
    ) -> "Population":
        config = config or NEATConfig()
        tracker = InnovationTracker()
        genomes: List[Genome] = []
        
        # 1) Build a single template with global node/innov IDs
        k0, key = jr.split(key)
        template_genome = Genome.from_initial_feedforward(
            n_inputs, n_outputs, tracker=tracker, key=k0, add_bias=add_bias, w_init_std=1.0
        )
        
        # 2) Copy it pop_size times and reinit weights per copy
        for i in range(config.pop_size):
            key, k = jr.split(key)
            g = template_genome.copy()
            # reinit all connection weights
            g.mutate_weights(k, p_reset=1.0, w_init_std=config.w_init_std)
            genomes.append(g)
        return Population(genomes, tracker, key, config)
    
    # ------------- main loop -------------
    def evaluate(self, eval_fn: Callable[[Genome, jax.Array], float]) -> None:
        scores: List[float] = [0.0] * len(self.genomes)
        key = self.key
        for i, g in enumerate(self.genomes):
            key, k = jr.split(key)
            scores[i] = float(eval_fn(g, k))
        self.key = key
        self.fitness = scores
    
    def speciate(self) -> None:
        if not self.genomes:
            self.species = []
            return

        # Distance cache (i, j) -> distance
        dist_cache = {}
        
        def dist(i: int, j: int) -> float:
            a, b = (i, j) if i <= j else (j, i)
            if (a, b) not in dist_cache:
                d = self.genomes[a].compatibility_distance(self.genomes[b], self.config.c1, self.config.c2, self.config.c3)
                dist_cache[(a, b)] = d
            return dist_cache[(a, b)]

        unspeciated = set(range(len(self.genomes)))
        new_rep_indices: List[int] = []
        new_members: List[List[int]] = []
        
        # 1) For each existing species, choose the closest genome in this population as new representative
        for s in self.species:
            # If all genomes are already taken as reps (can happen with tiny populations), stop.
            if not unspeciated:
                break
            
            # Find closest genome to old representative.
            rep_idx = min(unspeciated, key=lambda gid: dist(gid, s.representative))
            new_rep_indices.append(rep_idx)
            new_members.append([rep_idx])
            unspeciated.remove(rep_idx)
        
        # 2) Assign remaining genomes to the closest species under threshold; otherwise create a new species.
        while unspeciated:
            gid = unspeciated.pop()
            candidates = [] # (distance, species_idx)
            for species_idx, rep_idx in enumerate(new_rep_indices):
                d = dist(gid, rep_idx)
                if d <= self.config.delta_threshold:
                    candidates.append((d, species_idx))
            if candidates:
                d, species_idx = min(candidates, key=lambda x: x[0])
                new_members[species_idx].append(gid)
            else:
                new_rep_indices.append(gid)
                new_members.append([gid])
        
        # 3) Rebuild species list with chosen representatives and members.
        self.species = [
            Species(representative=rep_idx, members=members)
            for rep_idx, members in zip(new_rep_indices, new_members)
        ]
    
    def _adjust_fitness(self) -> List[float]:
        adjusted_fitness = [0.0] * len(self.genomes)  # will store adjusted fitness for each genome
        # for each genome, adjusted fitness is its fitness divided by the size of its species
        
        # we will eventually use the adjusted fitness to define a probability distribution for selection
        # (i.e., to decide which species should get more offspring)
        # So, adjusted fitness should be positive for all species (so that we can normalize and get a valid probability distribution)
        # so, we will just shift all fitness values to be positive
        assert self.fitness, Exception("Fitness values should be defined for all genomes (even if they are zero)")
        min_fitness = min(self.fitness)
        shift = (-min_fitness + 1e-8) if min_fitness < 0 else 0.0
        
        for s in self.species:
            assert len(s.members) > 0
            size = len(s.members)
            for gid in s.members:
                fitness = self.fitness[gid] + shift
                adjusted_fitness[gid] = fitness / size
        return adjusted_fitness
    
    def reproduce(self) -> None:
        """Evolve the population for one generation using NEAT reproduction.
        
        This method implements the core NEAT reproduction algorithm:
        1. Speciation: Group genomes into species based on compatibility
        2. Fitness sharing: Adjust fitness within species to promote diversity
        3. Selection: Allocate offspring to species based on adjusted fitness
        4. Reproduction: Create new genomes via crossover and mutation
        
        The reproduction process maintains diversity through speciation while
        promoting improvement through fitness-based selection and genetic operators.
        """
        # Prepare per-generation innovation sharing
        self.tracker.new_gen()
        
        # divide into species
        self.speciate()
        
        # adjust fitness
        adjusted_fitness = self._adjust_fitness()
        
        # species adjusted total (sum of adjusted fitness members)
        species_adj = [sum(adjusted_fitness[gid] for gid in s.members) for s in self.species]
        total_adj = sum(species_adj)
        
        key = self.key
        new_genomes: List[Genome] = []
        
        # Degenerate case: no fitness signal; create genomes in new generation by mutating random parents
        if total_adj <= 0 or not self.species:
            """Handle edge case where there's no fitness signal or no species.
            
            This can occur early in evolution when all genomes have zero fitness,
            or in pathological cases where speciation fails. We fall back to
            simple mutation of random parents to maintain population diversity.
            """
            for _ in range(self.config.pop_size):
                key, k_sel, k_mw, k_ac_b, k_ac = jr.split(key, 5)
                # Select random parent from current population
                parent_idx = int(jr.randint(k_sel, (), 0, len(self.genomes)))
                parent = self.genomes[parent_idx]
                child = parent.copy()
                
                # Apply basic mutations to create variation
                child.mutate_weights(k_mw)
                if jr.bernoulli(k_ac_b, self.config.p_mutate_add_connection):
                    child.mutate_add_connection(k_ac, self.tracker)
                new_genomes.append(child)
            
            # Update population state and exit early
            self.genomes = new_genomes
            self.fitness = [0.0] * len(new_genomes)
            self.key = key
            self.generation += 1
            return
        
        # Normal case: select weighted parents based on species adjusted fitness

        # Elitism: carry over the best from each species
        """Preserve the top performers from each species without modification.
        
        Elitism ensures that the best solutions aren't lost during reproduction.
        Each species contributes its top performers directly to the next generation,
        maintaining good solutions while allowing for further evolution.
        """
        for s in self.species:
            if not s.members:
                continue
            
            # Sort members by raw fitness (not adjusted) for elitism
            sorted_members = sorted(s.members, key=lambda gid: self.fitness[gid], reverse=True)
            for i in range(min(self.config.elite_per_species, len(sorted_members))):
                elite_idx = sorted_members[i]
                new_genomes.append(self.genomes[elite_idx].copy())
        
        # Calculate offspring allocation per species
        """Allocate remaining offspring slots to species based on adjusted fitness.
        
        Species with higher adjusted fitness get more offspring, promoting
        successful lineages while maintaining diversity through speciation.
        The allocation is proportional to each species' contribution to total fitness.
        """
        num_offspring = self.config.pop_size - len(new_genomes)
        offspring_per_species = []
        if total_adj > 0:
            # Proportional allocation based on species adjusted fitness
            for s_adj in species_adj:
                offspring_per_species.append(round(num_offspring * (s_adj / total_adj)))
        else:
            # Fallback: equal distribution if all adjusted fitness is zero
            offspring_per_species = [num_offspring // len(self.species)] * len(self.species)
            
        # Distribute rounding errors to maintain exact population size
        """Handle rounding errors from proportional allocation.
        
        Since we use round() for allocation, the total may not equal num_offspring.
        We redistribute excess/deficit offspring to maintain exact population size,
        favoring the best/worst species respectively.
        """
        current_total = sum(offspring_per_species)
        while current_total < num_offspring:
            # Give extra offspring to the best species
            best_species_idx = max(range(len(species_adj)), key=lambda i: species_adj[i])
            offspring_per_species[best_species_idx] += 1
            current_total += 1
        
        while current_total > num_offspring:
            # Remove offspring from the worst species (that still has some)
            worst_species_idx = min([i for i, n in enumerate(offspring_per_species) if n > 0], key=lambda i: species_adj[i])
            offspring_per_species[worst_species_idx] -= 1
            current_total -= 1
        
        # Reproduction loop: create offspring for each species
        """Generate offspring through selection, crossover, and mutation.
        
        For each species, we:
        1. Create fitness-proportional selection probabilities
        2. Choose reproduction method (asexual vs sexual)
        3. Apply genetic operators (crossover and/or mutation)
        4. Add resulting offspring to the new generation
        """
        for s, n_offspring in zip(self.species, offspring_per_species):
            if n_offspring == 0 or not s.members:
                continue
            
            # Prepare selection probabilities based on adjusted fitness
            member_fitness = [adjusted_fitness[gid] for gid in s.members]
            total_member_fitness = sum(member_fitness)

            # Create a selection probability distribution
            if total_member_fitness > 0:
                # Fitness-proportional selection
                probs = [f / total_member_fitness for f in member_fitness]
            else:
                # Uniform selection if all fitness is zero
                probs = [1.0 / len(s.members)] * len(s.members)
            
            # Generate offspring for this species
            for _ in range(n_offspring):
                key, k_sel, k_cross, k_mut = jr.split(key, 4)
                
                # Choose reproduction method
                # Asexual reproduction (mutation only)
                if len(s.members) == 1 or jr.bernoulli(k_sel, 1.0 - self.config.crossover_prob):
                    """Asexual reproduction: clone and mutate a single parent.
                    
                    Used when species has only one member or randomly chosen
                    based on crossover probability. Maintains parental genome
                    structure while introducing variation through mutation.
                    """
                    parent_idx = int(jr.choice(k_sel, len(s.members), p=jnp.array(probs)))
                    parent = self.genomes[s.members[parent_idx]]
                    child = parent.copy()
                # Sexual reproduction (crossover)
                else:
                    """Sexual reproduction: crossover between two parents.
                    
                    Combines genetic material from two parents using NEAT's
                    historical marking system. The crossover respects innovation
                    numbers and fitness differences between parents.
                    """
                    indices = jr.choice(k_sel, len(s.members), shape=(2,), p=jnp.array(probs), replace=False)
                    p1_idx, p2_idx = int(indices[0]), int(indices[1])
                    p1 = self.genomes[s.members[p1_idx]]
                    p2 = self.genomes[s.members[p2_idx]]
                    f1 = self.fitness[s.members[p1_idx]]
                    f2 = self.fitness[s.members[p2_idx]]
                    child = p1.crossover(p2, f1, f2, k_cross)
                
                # Apply mutation operators to the child
                """Mutate the offspring using various NEAT mutation operators.
                
                Each mutation type is applied probabilistically:
                - Weight mutation: Perturb or reset connection weights
                - Add connection: Create new connection between existing nodes
                - Add node: Split existing connection by inserting new node
                - Toggle connection: Enable/disable existing connections
                
                These operators allow both fine-tuning and structural innovation.
                """
                k_mw, k_ac_b, k_ac, k_an_b, k_an, k_tc_b, k_tc = jr.split(k_mut, 7)
                
                # Weight mutation: most common, fine-tunes existing connections
                if jr.bernoulli(k_mw, self.config.p_mutate_weights):
                    child.mutate_weights(k_mw, self.config.weight_sigma, self.config.weight_reset_prob)
                
                # Structural mutations: add complexity to the network
                if jr.bernoulli(k_ac_b, self.config.p_mutate_add_connection):
                    child.mutate_add_connection(k_ac, self.tracker)
                if jr.bernoulli(k_an_b, self.config.p_mutate_add_node):
                    child.mutate_add_node(k_an, self.tracker)
                
                # Connection state mutation: modify network topology
                if jr.bernoulli(k_tc_b, self.config.p_mutate_toggle_connection):
                    child.mutate_toggle_connection(k_tc)
                
                new_genomes.append(child)

        # Update population state for next generation
        """Finalize the new generation and prepare for next iteration.
        
        Replace the current population with the new generation, reset fitness
        values (to be evaluated in the next cycle), and increment generation counter.
        """
        self.genomes = new_genomes
        self.fitness = [0.0] * len(new_genomes)
        self.key = key
        self.generation += 1