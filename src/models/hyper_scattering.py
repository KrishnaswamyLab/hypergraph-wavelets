import numpy as np

def get_P(G):
  #incidence matrix
  H = G.incidence_matrix().todense()

  # vertex degree matrix (sum rows of H)
  D_v = np.matrix(np.diag(np.array(H).sum(axis=1)))

  # edge degree matrix (sum cols of H)
  D_e = np.matrix(np.diag(np.array(H).sum(axis=0)))

  # random walk operator matrix
  P = D_v.I*H*D_e.I*H.T

  N = P.shape[0]

  # return transpose so it operates Px
  return P.T

def get_wavelet_matrix(G, largest_scale):
  ''' Takes a largest scale, and returns a list of wavelet
  matrices up to those scales. For example, for largest_scale = 4,
  this will return (Phi_1, Phi 2, Phi_3, Phi_4)
  '''
  P = get_P(G)    # lazy random walk matrix
  N = P.shape[0]  # number of nodes
  powered_P = P   # we will exploit the dyadic scales for computational efficiency
  Phi = powered_P @ (np.identity(N) - powered_P)    # First wavelet
  wavelets = list()
  wavelets.append(Phi)
  for scale in range(2, largest_scale + 1):
    powered_P = powered_P @ powered_P               # Returns P^{2^(scale -1)}
    Phi = powered_P @ (np.identity(N) - powered_P)  # Calculate next wavelet
    wavelets.append(Phi)

  return wavelets

def geom_scattering(G, s, largest_scale, highest_moment):
  # J = largest_scale, Q = highest_moment, s = signal
  wavelets = get_wavelet_matrix(G, largest_scale)
  scattering_coefficients = list()

  # Calculate zero order scattering coefficients
  for q in range(1, highest_moment + 1):
    coeff = np.sum(np.power(s, q))
    scattering_coefficients.append(coeff)

  # Calculate first order scattering coefficients
  for scale in range(largest_scale):
    wavelet_transformed = wavelets[scale] @ s
    abs_wavelet = np.abs(wavelet_transformed)
    for q in range(1, highest_moment + 1):
      coeff = np.sum( np.power(abs_wavelet, q) )
      scattering_coefficients.append(coeff)

  # Calculate second order scattering coefficients
  for scale1 in range(1, largest_scale):
    # the second scale only goes up to the size of the first scale
    for scale2 in range(scale1):
      first_wavelet_transform = np.abs(wavelets[scale2] @ s )
      second_wavelet_transform = np.abs(wavelets[scale1] @ first_wavelet_transform)
      for q in range(1, highest_moment + 1):
        coeff = np.sum(np.power(second_wavelet_transform, q) )
        scattering_coefficients.append(coeff)

  return np.array(scattering_coefficients)