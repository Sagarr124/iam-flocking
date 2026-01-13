import numpy as np
from scipy.spatial import KDTree
from scipy.sparse import csgraph, csr_matrix

def calculate_order_parameter(vel):
    """
    Calculates the global polarization (order parameter).
    phi = | sum(v_i / |v_i|) | / N
    Closer to 1 means aligned, 0 means disordered.
    """
    speeds = np.linalg.norm(vel, axis=1, keepdims=True)
    # Avoid division by zero for stationary agents
    valid = speeds.flatten() > 1e-6
    if not np.any(valid):
        return 0.0
        
    normalized_vel = vel[valid] / speeds[valid]
    sum_vel = np.sum(normalized_vel, axis=0)
    return np.linalg.norm(sum_vel) / vel.shape[0]

def calculate_fragmentation(pos, connection_radius):
    """
    Calculates the number of connected components and the size of the largest cluster.
    Two agents are connected if dist(i, j) < connection_radius.
    This defines the 'physical' groups.
    """
    tree = KDTree(pos)
    # query_pairs finds all pairs with dist < r
    pairs = tree.query_pairs(connection_radius)
    
    if not pairs:
        return pos.shape[0], 1 # All isolated
        
    # Build graph
    # N nodes
    N = pos.shape[0]
    # Create adjacency matrix from pairs
    pairs = list(pairs)
    rows = [p[0] for p in pairs]
    cols = [p[1] for p in pairs]
    # Symmetric
    data = np.ones(len(rows))
    adj = csr_matrix((data, (rows, cols)), shape=(N, N))
    # Make symmetric (undirected)
    adj = adj + adj.T
    
    n_components, labels = csgraph.connected_components(adj)
    
    # Calculate sizes
    _, counts = np.unique(labels, return_counts=True)
    largest_cluster_size = np.max(counts) if len(counts) > 0 else 0
    
    return n_components, largest_cluster_size
