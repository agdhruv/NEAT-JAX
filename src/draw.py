from src.structure import Plan
import jax.numpy as jnp
from typing import Optional

def draw(plan: Plan, weights: Optional[jnp.ndarray] = None, save_path: Optional[str] = None):
    """Draw the network structure vertically with matplotlib.

    Args:
        weights: Optional weight array to display connection weights
        save_path: Optional path to save the figure (e.g., 'network.png')
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
    except ImportError:
        print("Matplotlib not available. Install with: pip install matplotlib")
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

    # Group nodes by level and position them
    level_groups = {}
    for node in nodes:
        level = node['level']
        if level not in level_groups:
            level_groups[level] = []
        level_groups[level].append(node)

    # Calculate positions
    max_nodes_per_level = max(len(nodes_in_level) for nodes_in_level in level_groups.values())
    level_positions = sorted(level_groups.keys())
    x_spacing = 2.0
    y_spacing = 1.5

    fig, ax = plt.subplots(figsize=(max(8, len(level_positions) * x_spacing),
                                    max(6, max_nodes_per_level * y_spacing)))

    # Position nodes
    node_positions = {}
    for level_idx, level in enumerate(level_positions):
        nodes_in_level = level_groups[level]
        start_y = (max_nodes_per_level - len(nodes_in_level)) / 2

        for node_idx, node in enumerate(nodes_in_level):
            x = level_idx * x_spacing
            y = start_y + node_idx * y_spacing
            node_positions[node['id']] = (x, y)

            # Draw node
            circle = patches.Circle((x, y), 0.3,
                                    facecolor=node['color'],
                                    edgecolor='black',
                                    linewidth=2,
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
        if src_idx in node_positions and dst_idx in node_positions:
            src_pos = node_positions[src_idx]
            dst_pos = node_positions[dst_idx]

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

            # Add weight label in the middle of the connection
            mid_x = (src_pos[0] + dst_pos[0]) / 2
            mid_y = (src_pos[1] + dst_pos[1]) / 2
            ax.text(mid_x, mid_y, f'{weight:.2f}', ha='center', va='center',
                    fontsize=8, bbox=dict(boxstyle='round,pad=0.2',
                                        facecolor='white', alpha=0.8))

    # Customize plot
    ax.set_xlim(-0.5, len(level_positions) * x_spacing - 0.5)
    ax.set_ylim(-1, max_nodes_per_level * y_spacing)
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

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Network visualization saved to {save_path}")

    return fig