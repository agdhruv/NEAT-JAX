from typing import Tuple


class InnovationTracker:
    """
    Tracks innovation numbers and node IDs for NEAT evolution.
    
    In NEAT, innovation numbers are crucial for determining which genes
    represent the same structural innovation across different genomes.
    This enables proper crossover operations by aligning corresponding
    genes between parents.
    
    The tracker maintains:
    - Global counters for innovation numbers and node IDs
    - History of connection innovations within a generation
    - History of node split operations within a generation
    """
    
    def __init__(self):
        """Initialize tracker with zero counters and empty histories."""
        self.next_innovation = 0
        self.next_node_id = 0
        self.conn_history = {}   # (in_id, out_id) -> innov
        self.node_history = {}   # split_conn_innov -> (new_node_id, innov_in, innov_out)

    def new_gen(self):
        """Clear generation-specific histories for the next generation."""
        self.conn_history.clear()
        self.node_history.clear()

    def allocate_connection(self, in_id: int, out_id: int) -> int:
        """
        Get innovation number for a connection between two nodes.
        
        If this exact connection has been created before in this generation,
        returns the existing innovation number. Otherwise allocates a new one.
        This ensures identical structural innovations get the same number.
        
        Args:
            in_id: Source node ID
            out_id: Target node ID
            
        Returns:
            Innovation number for this connection
        """
        key = (in_id, out_id)
        if key not in self.conn_history:
            innovation = self._alloc_innov()
            self.conn_history[key] = innovation
        return self.conn_history[key]
    
    def allocate_node(self) -> int:
        """Allocate next node ID."""
        return self._alloc_node()

    def split_connection(self, split_conn_innov: int) -> Tuple[int, int, int]:
        """
        Get innovation data for splitting a connection by adding a node.
        
        When a connection is split, a new node is inserted and the original
        connection is replaced by two new connections. If this exact split
        has been done before in this generation, returns the same IDs.
        
        Args:
            split_conn_innov: Innovation number of the connection being split
            
        Returns:
            Tuple of (new_node_id, input_connection_innov, output_connection_innov)
        """
        if split_conn_innov not in self.node_history:
            nid = self._alloc_node()
            inn1 = self._alloc_innov()
            inn2 = self._alloc_innov()
            self.node_history[split_conn_innov] = (nid, inn1, inn2)
        return self.node_history[split_conn_innov]

    # ----------------- Private Methods -----------------
    def _alloc_innov(self) -> int:
        """Allocate next innovation number."""
        i = self.next_innovation; self.next_innovation += 1; return i
    
    def _alloc_node(self) -> int:
        """Allocate next node ID."""
        n = self.next_node_id; self.next_node_id += 1; return n