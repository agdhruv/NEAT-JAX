"""
This file contains utilities to draw the network structure, mostly written by AI.
It doesn't contain any of the core functions of the NEAT algorithm, it's just a
utility to draw the network structure.
"""

from src.topology import Topology, build_topology_and_weights
from src.genome import Genome, INPUT, OUTPUT, HIDDEN, BIAS
import jax.numpy as jnp
from typing import Optional, List, Dict, Tuple, Set

def draw(plan: Topology, weights: Optional[jnp.ndarray] = None, save_path: Optional[str] = None, draw_weight_labels: bool = False):
    """Draw the network structure vertically with matplotlib.

    Args:
        weights: Optional weight array to display connection weights
        save_path: Optional path to save the figure (e.g., 'network.png')
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import networkx as nx
    except ImportError as e:
        print(f"Required library not available: {e}")
        print("Install with: pip install matplotlib networkx")
        return None

    # Convert arrays to lists for easier processing
    input_indices = set(plan.input_idx.tolist())
    output_indices = set(plan.output_idx.tolist())
    bias_indices = {plan.bias_idx} if plan.bias_idx >= 0 else set()

    # Create node information
    nodes = []
    for i in range(plan.n_nodes):
        if i in input_indices:
            node_type = 'input'
            color = 'lightblue'
            label = f'I{i}'
        elif i in output_indices:
            node_type = 'output'
            color = 'lightgreen'
            label = f'O{i}'
        elif i in bias_indices:
            node_type = 'bias'
            color = 'yellow'
            label = f'B{i}'
        else:
            node_type = 'hidden'
            color = 'lightgray'
            label = f'H{i}'

        nodes.append({
            'id': i,
            'type': node_type,
            'color': color,
            'label': label,
            'level': int(plan.levels[i])
        })

    # Build networkx graph for better layout
    G = nx.DiGraph()
    for node in nodes:
        G.add_node(node['id'], **node)
    
    # Add edges
    edges = list(zip(plan.src_idx.tolist(), plan.dst_idx.tolist()))
    G.add_edges_from(edges)
    
    # Find connected nodes (nodes that have at least one edge)
    connected_nodes = set()
    for src, dst in edges:
        connected_nodes.add(src)
        connected_nodes.add(dst)
    
    # Filter nodes to only include connected ones
    connected_node_data = [node for node in nodes if node['id'] in connected_nodes]
    isolated_count = len(nodes) - len(connected_node_data)
    
    # Use multipartite layout based on levels (horizontal layers)
    # This creates a much cleaner hierarchical visualization
    level_dict = {node['id']: node['level'] for node in connected_node_data}
    
    # Set subset attribute for ALL nodes in the graph, not just connected ones
    for node_id in G.nodes():
        if node_id in level_dict:
            G.nodes[node_id]['subset'] = level_dict[node_id]
        else:
            # For isolated nodes, assign them to level 0 or their actual level
            node_level = next((node['level'] for node in nodes if node['id'] == node_id), 0)
            G.nodes[node_id]['subset'] = node_level
    
    # Use multipartite layout with vertical orientation and increased scale
    node_positions = nx.multipartite_layout(G, subset_key='subset', align='horizontal', scale=6)
    
    # Swap x and y to make it horizontal (inputs on left, outputs on right)
    node_positions = {k: (v[1], -v[0]) for k, v in node_positions.items()}
    
    # Filter positions to only include connected nodes for drawing
    connected_positions = {k: v for k, v in node_positions.items() if k in connected_nodes}
    
    # Use fixed figure size for consistent dimensions across all generated images
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Calculate layout bounds for centering
    if connected_positions:
        x_coords = [pos[0] for pos in connected_positions.values()]
        y_coords = [pos[1] for pos in connected_positions.values()]
        x_center = (max(x_coords) + min(x_coords)) / 2
        y_center = (max(y_coords) + min(y_coords)) / 2
    else:
        x_center, y_center = 0, 0
    
    # Set fixed axis limits centered on the network
    ax.set_xlim(x_center - 6, x_center + 6)
    ax.set_ylim(y_center - 4, y_center + 4)
    
    # Draw nodes with improved positioning (only connected nodes)
    for node in connected_node_data:
        x, y = connected_positions[node['id']]
        
        # Draw node
        circle = patches.Circle((x, y), 0.15,
                                facecolor=node['color'],
                                edgecolor='black',
                                linewidth=1.0,
                                label=node['type'])
        ax.add_patch(circle)

        # Add label
        ax.text(x, y, node['label'], ha='center', va='center',
                fontweight='bold', fontsize=10)

    # Draw connections
    if weights is not None:
        edges = list(zip(plan.src_idx.tolist(), plan.dst_idx.tolist()))
        edge_weights = weights.tolist() if hasattr(weights, 'tolist') else weights
    else:
        edges = list(zip(plan.src_idx.tolist(), plan.dst_idx.tolist()))
        edge_weights = [1.0] * len(edges)  # Default weight for visualization

    for i, (src_idx, dst_idx) in enumerate(edges):
        if src_idx in connected_positions and dst_idx in connected_positions:
            src_pos = connected_positions[src_idx]
            dst_pos = connected_positions[dst_idx]

            # Draw connection line
            weight = float(edge_weights[i])
            linewidth = max(1, min(5, abs(weight) * 2))  # Scale line width by absolute weight
            alpha = min(1.0, 0.3 + abs(weight) * 0.7)  # Scale opacity by absolute weight

            # Color based on weight sign
            if weight > 0:
                color = 'green'
            elif weight < 0:
                color = 'red'
            else:
                color = 'gray'

            ax.plot([src_pos[0], dst_pos[0]], [src_pos[1], dst_pos[1]],
                    color=color, linewidth=linewidth, alpha=alpha, zorder=1)

            if draw_weight_labels:
                # Add weight label in the middle of the connection
                mid_x = (src_pos[0] + dst_pos[0]) / 2
                mid_y = (src_pos[1] + dst_pos[1]) / 2
                ax.text(mid_x, mid_y, f'{weight:.2f}', ha='center', va='center',
                        fontsize=8, bbox=dict(boxstyle='round,pad=0.2',
                                            facecolor='white', alpha=0.8))

    # Customize plot
    ax.set_aspect('equal')
    ax.axis('off')

    # Add legend
    legend_elements = [
        patches.Patch(color='lightblue', label='Input'),
        patches.Patch(color='lightgray', label='Hidden'),
        patches.Patch(color='lightgreen', label='Output'),
        patches.Patch(color='yellow', label='Bias')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    # Add text about isolated nodes if any exist
    if isolated_count > 0:
        ax.text(0.98, 0.02, f'{isolated_count} additional isolated nodes not shown',
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=9, style='italic', color='gray')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Network visualization saved to {save_path}")
        return None
    
    return fig


def plot_evolution(genomes: List[Genome], save_paths: Optional[List[str]] = None, 
                   draw_weight_labels: bool = False, draw_node_labels: bool = True,
                   generations: Optional[List[int]] = None) -> Optional[List]:
    """Plot multiple genomes with aligned structure for evolution visualization.
    
    This function ensures all genome visualizations have identical layouts by:
    1. Building a supergraph containing all nodes and edges from all genomes
    2. Creating a clean 3-column mesh: inputs left, hidden middle, outputs right
    3. Drawing each genome on this structure, with absent nodes/edges invisible
    
    Args:
        genomes: List of genomes to visualize
        save_paths: Optional list of save paths (same length as genomes)
        draw_weight_labels: Whether to show weight values on edges
        draw_node_labels: Whether to show labels on nodes
        generations: Optional list of generations to show in the plot
        
    Returns:
        List of figures if save_paths is None, otherwise None
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
    except ImportError as e:
        print(f"Required library not available: {e}")
        print("Install with: pip install matplotlib")
        return None
    
    if not genomes:
        print("No genomes provided")
        return None
    
    # Convert all genomes to topologies + weights
    topologies_weights = [build_topology_and_weights(g) for g in genomes]
    
    # Verify same inputs/outputs
    first_topo = topologies_weights[0][0]
    n_inputs = len(first_topo.input_idx)
    n_outputs = len(first_topo.output_idx)
    has_bias = first_topo.bias_idx >= 0
    
    for i, (topo, _) in enumerate(topologies_weights[1:], 1):
        if len(topo.input_idx) != n_inputs:
            raise ValueError(f"Genome {i} has {len(topo.input_idx)} inputs, expected {n_inputs}")
        if len(topo.output_idx) != n_outputs:
            raise ValueError(f"Genome {i} has {len(topo.output_idx)} outputs, expected {n_outputs}")
    
    # Build supergraph: collect all unique nodes across all genomes
    # Track nodes by TYPE only, ignoring NEAT levels
    
    # Structure: {type: count} to track max nodes for each type
    type_counts: Dict[int, int] = {}
    
    for genome in genomes:
        genome_type_counts: Dict[int, int] = {}
        for node in genome.nodes.values():
            if node.type not in genome_type_counts:
                genome_type_counts[node.type] = 0
            genome_type_counts[node.type] += 1
        
        # Update supergraph max counts
        for node_type, count in genome_type_counts.items():
            type_counts[node_type] = max(
                type_counts.get(node_type, 0), 
                count
            )
    
    # Build supergraph node list
    # Nodes are assigned sequential IDs in the supergraph
    super_nodes = []
    super_node_id = 0
    type_to_super_ids: Dict[int, List[int]] = {}
    
    for node_type in sorted(type_counts.keys()):
        count = type_counts[node_type]
        super_id_list = []
        for idx_in_type in range(count):
            super_nodes.append({
                'id': super_node_id,
                'type': node_type,
                'type_idx': idx_in_type  # Store for labeling
            })
            super_id_list.append(super_node_id)
            super_node_id += 1
        type_to_super_ids[node_type] = super_id_list
    
    # Collect all unique edges across all genomes
    # Edge defined by (src_type, src_idx_in_type, dst_type, dst_idx_in_type)
    super_edges: Set[Tuple[int, int]] = set()
    
    for genome in genomes:
        # Map genome nodes to their position in their type group
        node_to_type_idx: Dict[int, Tuple[int, int]] = {}
        type_counters: Dict[int, int] = {}
        
        for node_id in sorted(genome.nodes.keys()):
            node = genome.nodes[node_id]
            idx = type_counters.get(node.type, 0)
            type_counters[node.type] = idx + 1
            node_to_type_idx[node_id] = (node.type, idx)
        
        # Map to supergraph IDs and collect edges
        node_to_super_id: Dict[int, int] = {}
        for node_id, (node_type, idx) in node_to_type_idx.items():
            super_ids = type_to_super_ids[node_type]
            node_to_super_id[node_id] = super_ids[idx]
        
        # Add edges to supergraph
        for conn in genome.connections.values():
            if conn.enabled:
                src_super = node_to_super_id[conn.in_node]
                dst_super = node_to_super_id[conn.out_node]
                super_edges.add((src_super, dst_super))
    
    # Manually compute positions for a clean 3-column mesh layout
    # Group nodes by TYPE
    nodes_by_type: Dict[int, List[Dict]] = {INPUT: [], HIDDEN: [], OUTPUT: [], BIAS: []}
    for node in super_nodes:
        nodes_by_type[node['type']].append(node)
    
    # Sort nodes within each type for consistent ordering
    for node_type in nodes_by_type:
        nodes_by_type[node_type].sort(key=lambda n: n['id'])
    
    # Define 3 columns: inputs (+ bias) on left, hidden in middle, outputs on right
    # X positions
    x_input = 0.0
    x_hidden = 10.0
    x_output = 20.0
    
    # Y spacing between nodes
    y_spacing = 1.0
    
    # Compute positions
    super_positions = {}
    
    # Input nodes (left column)
    input_nodes = nodes_by_type[INPUT]
    bias_nodes = nodes_by_type[BIAS]
    left_nodes = input_nodes + bias_nodes
    n_left = len(left_nodes)
    if n_left > 0:
        start_y = -(n_left - 1) * y_spacing / 2
        for i, node in enumerate(left_nodes):
            super_positions[node['id']] = (x_input, start_y + i * y_spacing)
    
    # Hidden nodes (middle column)
    hidden_nodes = nodes_by_type[HIDDEN]
    n_hidden = len(hidden_nodes)
    if n_hidden > 0:
        start_y = -(n_hidden - 1) * y_spacing / 2
        for i, node in enumerate(hidden_nodes):
            super_positions[node['id']] = (x_hidden, start_y + i * y_spacing)
    
    # Output nodes (right column)
    output_nodes = nodes_by_type[OUTPUT]
    n_output = len(output_nodes)
    if n_output > 0:
        start_y = -(n_output - 1) * y_spacing / 2
        for i, node in enumerate(output_nodes):
            super_positions[node['id']] = (x_output, start_y + i * y_spacing)
    
    # Plot each genome
    figures = [] if save_paths is None else None
    
    for genome_idx, (genome, (topo, weights)) in enumerate(zip(genomes, topologies_weights)):
        # Map genome nodes to supergraph IDs
        genome_node_to_type_idx: Dict[int, Tuple[int, int]] = {}
        genome_type_counters: Dict[int, int] = {}
        
        for node_id in sorted(genome.nodes.keys()):
            node = genome.nodes[node_id]
            idx = genome_type_counters.get(node.type, 0)
            genome_type_counters[node.type] = idx + 1
            genome_node_to_type_idx[node_id] = (node.type, idx)
        
        genome_node_to_super_id: Dict[int, int] = {}
        super_id_to_node: Dict[int, int] = {}
        for node_id, (node_type, idx) in genome_node_to_type_idx.items():
            super_ids = type_to_super_ids[node_type]
            super_id = super_ids[idx]
            genome_node_to_super_id[node_id] = super_id
            super_id_to_node[super_id] = node_id
        
        # Map genome edges to supergraph edges
        genome_edges: Set[Tuple[int, int]] = set()
        edge_weights_map: Dict[Tuple[int, int], float] = {}
        
        for i, conn in enumerate(genome.connections.values()):
            if conn.enabled:
                src_super = genome_node_to_super_id[conn.in_node]
                dst_super = genome_node_to_super_id[conn.out_node]
                genome_edges.add((src_super, dst_super))
                edge_weights_map[(src_super, dst_super)] = float(conn.weight)
        
        # Create figure with fixed size for consistency
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Calculate axis bounds to center the network
        if super_positions:
            x_coords = [pos[0] for pos in super_positions.values()]
            y_coords = [pos[1] for pos in super_positions.values()]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            # Add padding around the network (20% on each side)
            x_range = x_max - x_min
            y_range = y_max - y_min
            x_padding = x_range * 0.2
            y_padding = y_range * 0.2
            
            ax.set_xlim(x_min - x_padding, x_max + x_padding)
            ax.set_ylim(y_min - y_padding, y_max + y_padding)
        
        # Draw all edges (supergraph)
        for src_super, dst_super in super_edges:
            src_pos = super_positions[src_super]
            dst_pos = super_positions[dst_super]
            
            if (src_super, dst_super) in genome_edges:
                # Edge exists in this genome - draw normally
                weight = edge_weights_map[(src_super, dst_super)]
                linewidth = max(1, min(5, abs(weight) * 2))
                alpha = min(1.0, 0.3 + abs(weight) * 0.7)
                color = 'green' if weight > 0 else 'red' if weight < 0 else 'gray'
                
                ax.plot([src_pos[0], dst_pos[0]], [src_pos[1], dst_pos[1]],
                       color=color, linewidth=linewidth, alpha=alpha, zorder=1)
                
                if draw_weight_labels:
                    mid_x = (src_pos[0] + dst_pos[0]) / 2
                    mid_y = (src_pos[1] + dst_pos[1]) / 2
                    ax.text(mid_x, mid_y, f'{weight:.2f}', ha='center', va='center',
                           fontsize=8, bbox=dict(boxstyle='round,pad=0.2',
                                               facecolor='white', alpha=0.8))
            else:
                # Edge doesn't exist - draw invisible
                ax.plot([src_pos[0], dst_pos[0]], [src_pos[1], dst_pos[1]],
                       color='white', linewidth=0.5, alpha=0.01, zorder=1)
        
        # Draw all nodes (supergraph)
        for node in super_nodes:
            super_id = node['id']
            x, y = super_positions[super_id]
            
            if super_id in super_id_to_node:
                # Node exists in this genome - draw normally
                node_type = node['type']
                type_idx = node['type_idx']
                if node_type == INPUT:
                    color = 'lightblue'
                    label = f"I{type_idx}"
                elif node_type == OUTPUT:
                    color = 'lightgreen'
                    label = f"O{type_idx}"
                elif node_type == BIAS:
                    color = 'yellow'
                    label = f"B{type_idx}"
                else:  # HIDDEN
                    color = 'lightgray'
                    label = f"H{type_idx}"
                
                # Smaller radius for cleaner look with many nodes
                node_radius = 0.4
                circle = patches.Circle((x, y), node_radius,
                                       facecolor=color,
                                       edgecolor='black',
                                       linewidth=0.5)
                ax.add_patch(circle)
                if draw_node_labels:
                    ax.text(x, y, label, ha='center', va='center',
                           fontweight='bold', fontsize=8, zorder=3)
            else:
                # Node doesn't exist - draw invisible
                node_radius = 0.4
                circle = patches.Circle((x, y), node_radius,
                                       facecolor='white',
                                       edgecolor='white',
                                       linewidth=0.5,
                                       alpha=0.01)
                ax.add_patch(circle)
        
        # Customize plot
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Add legend with fixed positioning
        legend_elements = [
            patches.Patch(color='lightblue', label='Input'),
            patches.Patch(color='lightgray', label='Hidden'),
            patches.Patch(color='lightgreen', label='Output'),
            patches.Patch(color='yellow', label='Bias')
        ]
        ax.legend(handles=legend_elements, bbox_to_anchor=(1, 0), loc='upper right')
        
        # Add genome info as title
        n_nodes = len(super_id_to_node)
        n_edges = len(genome_edges)
        generation = generations[genome_idx] if generations else genome_idx
        ax.set_title(f'Genome {generation}: {n_nodes} nodes, {n_edges} edges',
                    fontsize=12, fontweight='bold', pad=15,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_paths and genome_idx < len(save_paths):
            plt.savefig(save_paths[genome_idx], dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Genome {generation} visualization saved to {save_paths[genome_idx]}")
        elif save_paths is None and figures is not None:
            figures.append(fig)
    
    return figures