import numpy as np
from scipy.spatial.distance import pdist, squareform

def calculate_velocity_correlation(pos, vel, dr=2.0, max_r=None):
    """
    Calculates the Velocity Correlation Function C(r).
    C(r) = < u_i * u_j >_r
    where u_i is the fluctuation velocity: v_i - <v>
    
    Returns:
        rs: array of distance bins
        Cr: array of correlation values for each bin
    """
    N = pos.shape[0]
    
    # 1. Compute Fluctuation Velocities
    # Global mean velocity
    mean_vel = np.mean(vel, axis=0)
    # Mean speed (scalar) to normalize?
    # Ballerini Eq 1: u_i = v_i - V (where V is flock velocity)
    # Normalized C(r) is usually C(r) / C(0).
    u = vel - mean_vel
    
    # Speed of fluctuations for normalization
    # Not strictly necessary if we just look for zero crossing, 
    # but good for plotting 1.0 -> 0.0
    
    # 2. Pairwise Distances and Dot Products
    # For N=400, N^2 = 160,000, which is fine for numpy.
    dists = pdist(pos) # (N*(N-1)/2,)
    
    # Pairwise dot products of u
    # u_dot_u[i,j] = u[i] . u[j]
    # Efficient way for condensed list?
    # pdist doesn't do dot product.
    # Expand to square matrix for ease, or manual loop?
    # N=400 is small enough for square matrix.
    
    u_dot = np.dot(u, u.T) # (N, N)
    # Extract upper triangle to match pdist
    u_prod = u_dot[np.triu_indices(N, k=1)]
    
    # Normalize by variance? C(0) = sum(u_i^2)/N.
    # We want C(distance). 
    
    # 3. Binning
    if max_r is None:
        max_r = np.max(dists)
        
    bins = np.arange(0, max_r, dr)
    # digitize returns 1-based indices
    inds = np.digitize(dists, bins)
    
    # Average u_prod per bin
    # Use bincount for speed
    
    # specific heat/mass accumulation
    bin_counts = np.bincount(inds, minlength=len(bins)+1)
    bin_sums = np.bincount(inds, weights=u_prod, minlength=len(bins)+1)
    
    # Avoid div by zero
    valid = bin_counts > 0
    
    # C(r)
    Cr = np.zeros_like(bin_sums)
    Cr[valid] = bin_sums[valid] / bin_counts[valid]
    
    # Start usually from r=0 (self correlation) which is 1.0 normalized
    # But pdist is strict i!=j.
    # We can prepend distance 0, C=1.0 (normalized)
    
    # Normalize so curve starts near 1?
    # C(0) theoretical = <u^2>
    c_zero = np.mean(np.sum(u**2, axis=1))
    
    if c_zero > 1e-9:
        Cr /= c_zero
        
    # Return valid bins (inds 1 maps to bins[0]...bins[1])
    # digitize index i corresponds to bins[i-1] <= x < bins[i]
    # so index 1 means bin 0
    # Return bin centers
    
    centers = bins[:-1] + dr/2
    # inds goes from 1 to len(bins)
    # We want indices 1...len(bins)-1 usually
    # bin_sums has length len(bins)+1
    
    return centers, Cr[1:len(centers)+1]

def calculate_correlation_length(rs, Cr):
    """
    Finds the distance r where C(r) first drops below zero.
    """
    # Find indices where C(r) < 0
    below_zero = np.where(Cr < 0)[0]
    if len(below_zero) > 0:
        return rs[below_zero[0]]
    else:
        # If never crosses zero, correlation length is "infinite" (or system size)
        return rs[-1]
