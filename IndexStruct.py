from typing import NamedTuple

class IndexStruct(NamedTuple):
    """
    Named tuple used to map the state vector, the simulation vector and the observation vector:
    * xs[simulation] element is the ground truth element being estimated in xk[state] and being observed by zk[observation].

    This index structure is required for the logging and plotting functions.

    :attr:`state`: index of the estimated state vector
    :attr:`simulation`: index of the simulated state vector
    :attr:`observation`: index of the observation vector
    """

    state: str  # degree of freedom
    simulation: int  # index of the simulated state
    observation: int  # index of the observation
