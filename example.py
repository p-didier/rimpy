import sys
import numpy as np
import rimPypack.rimPy as rimpy
import matplotlib.pyplot as plt

SEED = 12345
ROOM_DIM_BOUNDS = [5, 10]  # [min, max], meters
RIR_DURATION = 0.5  # seconds
FS = 16000  # Hz

def main():
    # Read parameters from file
    rng = np.random.default_rng(SEED)

    # Set parameters
    roomDims = (np.amax(ROOM_DIM_BOUNDS) - np.amin(ROOM_DIM_BOUNDS)) *\
        rng.random(3) + np.amin(ROOM_DIM_BOUNDS)
    micPos = roomDims * rng.random(3)
    sourcePos = roomDims * rng.random(3)
    beta = 2 * rng.random(2, 3) - 1

    # Compute
    h = rimpy.rimPy(
        micPos=micPos,
        sourcePos=sourcePos,
        roomDim=roomDims,
        beta=beta,
        rirDuration=RIR_DURATION,
        fs=FS
    )

    # Plot
    fig = plot_rir(h)
    fig.savefig(f'RIRplot.png')
    plt.show()

def plot_rir(h):
    fig, axes = plt.subplots(1,1)
    fig.set_size_inches(8.5, 3.5)
    axes.plot(h)
    axes.grid()
    plt.tight_layout()	
    return fig

if __name__ == '__main__':
    sys.exit(main())