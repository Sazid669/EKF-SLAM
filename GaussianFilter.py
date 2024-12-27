class GaussianFilter:
    """
    Gaussian Filter Interface

    """
    def __init__(self, x0, P0, *args):
        """
        Constructor of the GaussianFilter class.

        **Attributes**:

        * :attr:`xk`: mean of the state vector at time step k
        * :attr:`Pk`: covariance of the state vector at time step k
        * :attr:`xk_1`: mean of the state vector at time step k-1
        * :attr:`Pk_1`: covariance of the state vector at time step k-1

        :param x0: initial mean state vector
        :param P0: initial covariance matrix
        """
        self.xk_1 = x0  # initialize state vector
        self.Pk_1 = P0  # initialize covariance matrix
        self.xk = x0  # initialize state vector
        self.Pk = P0  # initialize covariance matrix

    def Prediction(self, uk, Qk, xk_1=None, Pk_1=None):
        """
        Prediction step of the Gaussian Filter to be overwritten by the child class.

        :param uk: input vector
        :param Qk: covariance matrix of the motion model noise
        :param xk_1: previous mean state vector
        :param Pk_1: previous covariance matrix
        :return xk_bar, Pk_bar: current mean state vector and covariance matrix
        """
        pass

    def Update(self, zk, Rk, xk_bar=None, Pk_bar=None):
        """
        Update step of the Gaussian Filter to be overwritten by the child class.

        :param zk: observation vector
        :param Rk: covariance of the observation model noise
        :param xk_bar: mean of the predicted state
        :param Pk_bar: covariance  of the predicted state
        :return: xk, Pk: current mean state vector and covariance matrix
        """
        pass
