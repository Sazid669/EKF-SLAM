from GaussianFilter import *

class KF(GaussianFilter):
    """
    Kalman Filter class. Implements the :class:`GaussianFilter` interface for the particular case of the Kalman Filter.
    """
    def __init__(self, Ak, Bk, Hk, Vk, x0, P0, *args):
        """
        Constructor of the KF class.

        :param Ak: Transition matrix of the motion model
        :param Bk: Input matrix of the motion model
        :param Hk: Observation matrix of the observation model
        :param Vk: Noise projection matrix of the motion model
        :param x0: initial mean of the state vector
        :param P0: initial covariance matrix
        :param args: arguments to be passed to the parent class
        """
        self.Ak = Ak
        self.Bk = Bk
        self.Hk = Hk
        self.Vk = Vk
        self.xk_1 = x0
        self.Pk_1 = P0
        super().__init__(x0, P0, *args)

    def Prediction(self, uk, Qk, xk_1=None, Pk_1=None):
        """
        Prediction step of the Kalman Filter.

        :param uk: input vector
        :param Qk: covariance matrix of the motion model noise
        :param xk_1: previous mean state vector
        :param Pk_1: previous covariance matrix
        :return xk_bar, Pk_bar: current mean state vector and covariance matrix
        """
        # logging for plotting
        self.xk_1 = xk_1 if xk_1 is not None else self.xk_1
        self.Pk_1 = Pk_1 if Pk_1 is not None else self.Pk_1

        self.uk = uk
        self.Qk = Qk  # store the input and noise covariance for logging

        # KF equations begin here

        # TODO: To be implemented by the student

        return self.xk_bar, self.Pk_bar  # returns the predicted state vector

    def Update(self, zk, Rk, xk_bar=None, Pk_bar=None):
        """
        Update step of the Kalman Filter.

        :param zk: observation vector
        :param Rk: covariance of the observation model noise
        :param xk_bar: predicted mean state vector
        :param Pk_bar: predicted covariance matrix
        :return xk.Pk:  current mean state vector and covariance matrix
        """
        # logging for plotting
        self.xk_bar = xk_bar if xk_bar is not None else self.xk_bar
        self.Pk_bar = Pk_bar if Pk_bar is not None else self.Pk_bar
        self.zk = zk;
        self.Rk = Rk  # store the observation and noise covariance for logging

        # KF equations begin here

        # TODO: To be implemented by the student

        return self.xk, self.Pk
