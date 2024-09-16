"""
Computes 2-Wasserstein distance of two 2d Gaussian distributions using mean and covariance.
The 2d Gaussian is a 2nd-order approximation to the distributions due to computational constraints and curse of dimensionality.
"""
import math
from tqdm import tqdm
import torch

def sqrt_mat(M):
    eigenvalues, eigenvectors = torch.linalg.eigh(M)
    D_half = torch.diag(torch.sqrt(eigenvalues))
    Mhalf = eigenvectors @ D_half @ eigenvectors.T
    return Mhalf

def compute_w2(mu1, mu2, sig1, sig2, eps=1e-5):
    sig1_half = sqrt_mat(sig1)
    M = sig1_half @ sig2 @ sig1_half
    sqrtM = sqrt_mat(M)
    w2sq = torch.square(mu1 - mu2).sum() + torch.trace(sig1 + sig2 - 2 * sqrtM)
    if w2sq > -eps and w2sq < 0.:
        w2sq = torch.tensor(0., device=w2sq.device, dtype=w2sq.dtype)
    w2 = torch.sqrt(w2sq)
    return w2

# Batch version
def sqrt_mat_batch(M):
    # M: batch of matrices with shape [batch_size, d, d]
    eigenvalues, eigenvectors = torch.linalg.eigh(M)  # Eigen decomposition for each matrix in the batch
    D_half = torch.sqrt(eigenvalues).diag_embed()  # Create batch of diagonal matrices with square roots
    M_half = torch.matmul(torch.matmul(eigenvectors, D_half), eigenvectors.transpose(-2, -1))
    return M_half

def compute_w2_batch(mu1, mu2, sig1, sig2, eps=1e-5):
    # mu1, mu2: batches of means with shape [batch_size, d]
    # sig1, sig2: batches of covariance matrices with shape [batch_size, d, d]
    sig1_half = sqrt_mat_batch(sig1)
    M = torch.matmul(torch.matmul(sig1_half, sig2), sig1_half)
    sqrtM = sqrt_mat_batch(M)
    diff_mu = mu1 - mu2
    trace_term = torch.diagonal(sig1 + sig2 - 2 * sqrtM, dim1=-2, dim2=-1).sum(-1)  # Compute trace for each matrix in the batch
    w2sq = (diff_mu ** 2).sum(-1) + trace_term
    w2sq[(w2sq > -eps)&(w2sq < 0.)] = 0.
    w2 = torch.sqrt(w2sq)
    return w2

def batch_compute_w2_dist(mu1, mu2, cov1, cov2, batch_size):
    """
    Computes the Wasserstein distance between pairs of distributions in batches.
    
    Parameters:
    - mu: Tensor of means with shape [n_points, d].
    - cov: Tensor of covariance matrices with shape [n_points, d, d].
    - batch_size: Integer, the number of points to process in each batch.
    
    Returns:
    - distances: Tensor of Wasserstein distances.
    """
    n_points = mu1.shape[0]
    distances = torch.zeros((n_points), device=mu1.device)
    
    # Calculate the number of batches needed
    n_batches = math.ceil(n_points / batch_size)
    
    for i in tqdm(range(n_batches)):
        start_i = i * batch_size
        end_i = min((i + 1) * batch_size, n_points)
        # Compute Wasserstein distance for the current batch pair
        batch_distances = compute_w2_batch(
            mu1[start_i:end_i, :], mu2[start_i:end_i, :],
            cov1[start_i:end_i, :, :], cov2[start_i:end_i, :, :])
            
            # Assign the computed distances to the appropriate positions in the distances matrix
            
        distances[start_i:end_i] = batch_distances
    
    return distances

def vector_to_square_matrix(dist_vec, n):
    """
    Convert a distance vector to a square distance matrix.
    
    Parameters:
    - dist_vec: A 1D tensor of distances for the upper triangular part, excluding the diagonal.
    - n: The number of points (the dimension of the square matrix).
    
    Returns:
    - A 2D tensor representing the square distance matrix.
    """
    # Initialize an empty matrix
    distance_matrix = torch.zeros((n, n), dtype=dist_vec.dtype, device=dist_vec.device)
    
    # Get the upper triangular indices, excluding the diagonal
    triu_indices = torch.triu_indices(n, n, 1)
    
    # Fill the upper triangular part
    distance_matrix[triu_indices[0], triu_indices[1]] = dist_vec
    
    # Mirror the upper triangular to lower triangular
    distance_matrix = distance_matrix + distance_matrix.t()
    
    return distance_matrix
