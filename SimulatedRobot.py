from roboticstoolbox.mobile.Animations import *
import numpy as np

class SimulatedRobot:
    """
    This is the base class to simulate a robot. There are two operative frames: the world  N-Frame (North East Down oriented) and the robot body frame body B-Frame.
    Each robot has a motion model and a measurement model. The motion model is used to simulate the robot motion and the measurement model is used to simulate the robot measurements.

    **All Robot simulation classes must derive from this class** .

    """

    dt = 0.1  #:class attribute containing sample time of the simulation

    def __init__(self, xs0, map=[], * args):
        """
        :param xs0: initial simulated robot state :math:`x_{s_0}` used to initialize the the motion model
        :param map: feature map of the environment :math:`M=[^Nx_{F_1}^T,...,^Nx_{F_{nf}}^T]^T`

        Constructor. First, it initializes the robot simulation defining the following attributes:

        * **k** : time step
        * **Qsk** : **To be defined in the derived classes**. Object attribute containing Covariance of the simulation motion model noise
        * **usk** : **To be defined in the derived classes**. Object attribute contining the simulated input to the motion model
        * **xsk** : **To be defined in the derived classes**. Object attribute contining the current simulated robot state
        * **zsk** : **To be defined in the derived classes**. Object attribute contining the current simulated robot measurement
        * **Rsk** : **To be defined in the derived classes**. Object attribute contining the observation noise covariance matrix
        * **xsk** : current pose is the initial state
        * **xsk_1** : previouse state is the initial robot state
        * **M** : position of the features in the N-Frame
        * **nf** : number of features

        Then, the robot animation is initialized defining the following attributes:

        * **vehicleIcon** : Path file of the image of the robot to be used in the animation
        * **vehicleFig** : Figure of the robot to be used in the animation
        * **vehicleAxes** : Axes of the robot to be used in the animation
        * **xTraj** : list containing the x coordinates of the robot trajectory
        * **yTraj** : list containing the y coordinates of the robot trajectory
        * **visualizationInterval** : time-steps interval between two consecutive frames of the animation
        """
        super().__init__(*args)

        # Inititalize the robot simulation
        self.k = 0  # time step
        self.Qsk = None # **To be defined in the derived classes**. Object attribute containing Covariance of the simulation motion model noise
        self.usk = None # **To be defined in the derived classes**. Object attribute contining the simulated input to the motion model
        self.xsk = None # **To be defined in the derived classes**. Object attribute contining the current simulated robot state
        self.Rsk = None # **To be defined in the derived classes**. Object attribute contining the observation noise covariance matrix
        self.xsk = xs0 # Current pose is the initial state
        self.xsk_1 = xs0  # Previouse state is the initial robot state
        self.M = map  # position of the features in the N-Frame
        self.nf, self.DoF_f, _ = np.shape(self.M)  # number of features and the degree of freedom of the features

        # Inititalize the robot animation
        self.vehicleIcon = VehicleIcon('DifferentialDrive.png', scale=1, rotation=90) # Image of the robot to be used in the animation. By default it uses the image of a Differential Mobile Robot.
        self.vehicleFig = plt.figure()
        self.vehicleAxes = plt.axes()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Robot Simulation")

        feature_circle = []
        for i in range(len(map)):
            feature_circle.append(patches.Circle((map[i][0], map[i][1]), 0.5, fc='r'))
            self.vehicleAxes.add_patch(feature_circle[i])

        self.xTraj = [xs0[0, 0]]
        self.yTraj = [xs0[1, 0]]
        self.trajectory = plt.plot(self.xTraj, self.yTraj, 'b')

        self.visualizationInterval = 1 # number of time steps between two visualization updates

        self.vehicleIcon.plot([xs0[0], xs0[1], xs0[3]])
        self.plt_samples = []
        plt.title("Simulated Robot")


    def PlotRobot(self):
        """ Updates the plot of the robot at the current pose """

        self.vehicleIcon.update([self.xsk[0], self.xsk[1], self.xsk[3]])
        plt.pause(0.0000001)
        return

    def fs(self, xsk_1, uk):  # input velocity motion model with velocity noise
        """ Motion model used to simulate the robot motion. Computes the current robot state :math:`x_k` given the previous robot state :math:`x_{k-1}` and the input :math:`u_k`.
        It also updates the object attributes :math:`xsk`, :math:`xsk_1` and  :math:`usk` to be made them available for plotting purposes.
        *To be overriden in child class*.


        :parameter xsk_1: previous robot state :math:`x_{k-1}`
        :parameter usk: model input :math:`u_{s_k}`
        :return: current robot state :math:`x_k`
        """
        pass

    def SetMap(self, map):
        """ Initializes the map of the environment."""

        self.M = map
        return

    def _PlotSample(self, x, P, n):
        """
        Plots n samples of a multivariate gaussian distribution. This function is used only for testing, to plot the
        uncertainty through samples.
        :param x: mean pose of the distribution
        :param P: covariance of the distribution
        :param n: number of samples to plot
        """
        ns = len(self.plt_samples)
        if ns > 0:
            for i in range(ns): self.plt_samples[i].remove()
            self.plt_samples = []

        sample = np.random.multivariate_normal(x[0:2, 0], P[0:2, 0:2], n).T

        for i in range(sample.shape[1]):
            plt_sample, = plt.plot(sample[0, i], sample[1, i], 'r.')
            self.plt_samples.append(plt_sample)
        return