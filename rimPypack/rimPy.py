import numpy as np
import itertools

def rimPy(
        micPos: np.ndarray,
        sourcePos: np.ndarray,
        roomDim: np.ndarray,
        beta: np.ndarray,
        rirDuration: float,
        fs,
        randDist=0,
        tw=None,
        fc=None,
        c=343
    ):
    """
    Generates the impulse response of the (randomised) image
    method as proposed in De Sena et al. "On the modeling of 
    rectangular geometries in  room acoustic simulations." IEEE/ACM 
    Transactions on Audio, Speech and Language Processing (TASLP) 23.4 
    (2015): 774-786.
    It can also generate the response of the standard image method,
    if needed. The script uses fractional delays as proposed by 
    Peterson, ``Simulating the response of multiple microphones to a 
    single acoustic source in a reverberant room,'' JASA, 1986.

    Parameters
    ----------
    -micPo : [M x 3] np.ndarray[float] 
        3D coordinates of the `M` omnidirectional microphones [meters]. 
    -sourcePos : [3 x 1] np.ndarray[float]
        3D coordinates of the omni sound source [meters].
    -roomDim : [3 x 1] np.ndarray[float]
        Dimensions of the room (x,y,z) [meters].
    -beta : [2 x 3] np.ndarray[float]
        Reflection coefficient of the walls: [x1, y1, z1; x2, y2, z2],
        where, e.g., `x1` and `x2` are the reflection coefficient of the
        surface orthogonal to the x-axis, with the subscript 1 referring to
        walls adjacent to the origin, and the subscript 2 referring to the
        opposite wall. For an anechoic scenario, set: `beta = np.zeros((2, 3))`.
    -rirDuration : float
        Duration of the RIR [seconds].
    -fs : int or float
        Sampling frequency [Hz].
    -randDist : float
        Random distance added to the position of the image sources [meters]
        (rand_dist=0 for the standard image method).
    -tw : float
        Length of the low-pass filter [seconds].
        Default is 40 samples, i.e., `tw = 40 / fs`.
    -fc : int or float
        Cut-off frequency of the fractional delay filter [Hz]
        Default is `Fc = fs / 2` -- CHANGE MADE ON 28.10.2021 by Paul Didier
        (from originally `Fc = 0.9 * (fs / 2)` in De Sena's original
        MATLAB implementation).
    -c : int or float
        Speed of sound [meters/seconds].

    If you use this code, please cite De Sena et al. "On the modeling of 
    rectangular geometries in  room acoustic simulations." IEEE/ACM 
    Transactions on Audio, Speech and Language Processing (TASLP) 23.4 
    (2015): 774-786.

    Author: Paul Didier (paul.didier AT kuleuven DOT be)
        Adapted from MATLAB script by E. De Sena.
    """
    
    # Default input arguments
    if tw is None:
        tw = 40 / fs
    if fc is None:
        # Fc = 0.9*fs/2
        fc = fs / 2
    
    # Useful variables
    if micPos.ndim == 1:
        micPos = micPos[np.newaxis, :]
    nMics = micPos.shape[0]
    npts = int(np.ceil(rirDuration * fs))

    # Check that room dimensions are not too small
    if len(micPos.shape) == 2:
        condition = np.linalg.norm(micPos, axis=1) > np.linalg.norm(roomDim)
    else:
        condition = np.linalg.norm(micPos) > np.linalg.norm(roomDim)
    if condition.any():
        raise ValueError("Some microphones are located outside the room.")
    if np.linalg.norm(sourcePos) > np.linalg.norm(roomDim):
        raise ValueError("Some sources are located outside the room.")
    
    h = np.zeros((npts, nMics))
    ps = perm([0, 1], [0, 1], [0, 1])   # all binary numbers between 000 and 111
    orr = np.ceil(np.divide(rirDuration * c, roomDim * 2))
    rs = perm(
        range(-int(orr[0]), int(orr[0]) + 1),
        range(-int(orr[1]), int(orr[1]) + 1),
        range(-int(orr[2]), int(orr[2]) + 1)
    )
    nPermutations = rs.shape[1]
    
    for ii in range(nPermutations):
        print(f'Iter {ii+1}/{nPermutations}...')
        
        for jj in range(8):

            imagePos = np.multiply(1 - 2 * ps[:, jj], sourcePos +\
                        2 * np.multiply(rs[:, ii], roomDim))
            randomDelta = 2 * np.random.rand(1, 3) - np.ones((1, 3))
            randomizedImagePos = imagePos + randDist * randomDelta
            
            if ii == 999 and jj == 0:
                stop = 1
                
            for m in range(nMics):
                d = np.linalg.norm(randomizedImagePos - micPos[m, :])
                # Init outputs
                vals = np.array([0])
                n = np.array([0], dtype=int)
                # Compute
                sampleDist = np.round(d / c * fs)
                if sampleDist >= 1 and sampleDist <= npts:
                    am = np.multiply(
                        np.power(
                            beta[0, :],
                            np.abs(rs[:, ii] + ps[:, jj])
                        ),
                        np.power(
                            beta[1 :],
                            np.abs(rs[:, ii])
                        )
                    )
                    if tw == 0:
                        n = np.array([sampleDist], dtype=int)
                        vals = np.array([np.prod(am) / (4 * np.pi * d)])
                    else:
                        n = np.arange(
                            np.maximum(np.ceil(fs * (d / c - tw / 2)), 1) - 1,
                            np.minimum(
                                np.floor(fs * (d / c + tw / 2)),
                                npts - 1
                            ),
                            dtype=int
                        )
                        t = (n + 1) / fs - d / c
                        s = np.multiply(
                            1 + np.cos(2 * np.pi * t / tw),
                            np.sinc(2 * fc * t) / 2
                        )
                        vals = s * np.prod(am) / (4 * np.pi * d)    # Build RIR
                h[n+1, m] += vals

    return h


def perm(a, b, c):
    """Helper function: perform adequate permutations."""
    s = [a, b, c]
    arr = np.array(list(itertools.product(*s))).T
    return np.flipud(arr)

# @njit   
# def get_h_outerloop(
#     ps,
#     source_pos,
#     r,
#     room_dim,
#     rand_dist,
#     M,
#     mic_pos,
#     c,
#     fs,npts,beta,tw,Fc,h):
    
#     # JIT-ed computations
#     for jj in range(8):

#         p = ps[jj,:]

#         part1 = np.multiply(1 - 2*p, source_pos + 2*np.multiply(r, room_dim))
#         part2 = rand_dist*(2*np.random.rand(1,3) - np.ones((1,3)))
#         image_pos = part1 + part2
        
#         for m in range(M):
#             vals, n = get_h_innerloop(image_pos, mic_pos[m,:], c, fs, npts, beta, r, p, tw, Fc)
#             h[n.astype(np.int_),m] += vals

#     return h


# @njit
# def get_h_innerloop(image_pos, mic_pos, c, fs, npts, beta, r, p, tw, Fc):
#     # JIT-ed rimPy deepest inner-loop computations
#     d = np.linalg.norm(image_pos - mic_pos)
#     # init outputs
#     vals = np.array([0.0])
#     n = np.array([0.0])
#     # compute
#     if np.round(d/c*fs) >= 1 and np.round(d/c*fs) <= npts:

#         am = np.multiply(np.power(beta[0,:], np.abs(r + p)), np.power(beta[1,:], np.abs(r)))
#         if tw == 0:
#             n = np.array([np.round(d/c*fs)])
#             vals = np.array([np.prod(am)/(4*math.pi*d)])
#         else:
#             n = np.arange(np.maximum(np.ceil(fs*(d/c - tw/2)), 1.0),\
#                         np.minimum(np.floor(fs*(d/c + tw/2)), npts - 1.0))
#             t = n/fs - d/c
#             s = np.multiply(1.0 + np.cos(2*math.pi*t/tw), np.sinc(2*Fc*t)/2)
#             vals = s*np.prod(am)/(4*math.pi*d)    # Build RIR
#     # if d == 0:
#     #     vals[np.abs(vals) == math.inf] = 1  # Account for the special case where source point and receiver points are the same
#     #     print('The source and receiver points are the same.')


#     return vals, n


# def perm(a,b,c):
#     s = [a,b,c]
#     return np.array(list(itertools.product(*s)))


# def main():
    
#     alpha = 1
#     mic_pos = np.array([[0.1,0.1,0.1],])
#     source_pos = np.array([1,2,3])
#     # Special case
#     # mic_pos = source_pos[:,np.newaxis] + np.array([[0.01],[0.01],[0.01]])
#     room_dim = [5,6,7]
#     rir_length = 2**11
#     fs = 16e3
    
#     beta = -np.sqrt(1 - alpha)
    
#     h = rimPy(mic_pos, source_pos, room_dim, beta, rir_length/fs, fs, randDist=0, tw=None, fc=None, c=343)
        
#     fig, ax = plt.subplots()
#     ax.plot(h)
#     plt.show()

#     stop = 1


# # main()
