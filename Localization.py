import matplotlib.pyplot as plt
import numpy as np

class Localization:
    """
    Localization base class. Implements the localization algorithm.
    """

    def __init__(self, index, kSteps, robot, x0, P0, *args):
        """
        Constructor of the DRLocalization class.

        :param index: Logging index structure (:class:`prpy.Index`)
        :param kSteps: Number of time steps to simulate
        :param robot: Simulation robot object (:class:`prpy.Robot`)
        :param args: Rest of arguments to be passed to the parent constructor
        :param x0: Initial Robot pose in the N-Frame
        """
        self.robot = robot  # robot object
        self.index = index  # list of index structures
        self.kSteps = kSteps
        self.k = 0  # initialize log time step
        self.xk_1 = x0  # initialize state
        self.Pk_1 = P0  # initialize covariance P

        x_state_exists = False
        y_state_exists = False
        self.plot_xy_estimation = False
        for s in range(len(index)):
            if index[s].state == 'x': x_state_exists = True
            if index[s].state == 'y': y_state_exists = True
        self.plot_xy_estimation = x_state_exists & y_state_exists

        self.robot.visualizationInterval = 1#20

        self.xTraj = []
        self.yTraj = []
        self.trajectory = plt.plot([], [], marker='.', color='blue', markersize=1)

    def GetInput(self):  # get the input from the robot. To be overidden by the child class
        """
        Gets the input from the robot. To be overidden by the child class.

        :return uk: input variable
        """
        pass


    def Localize(self, xk_1, uk):
        """
        Single Localization iteration invoked from :meth:`prpy.DRLocalization.Localization`. Given the previous robot pose, the function reads the inout and computes the current pose.

        :parameter xk_1: previous robot pose
        :return xk: current robot pose

        """
        pass

    def LocalizationLoop(self, x0, usk):
        """
        Given an initial robot pose :math:`x_0` and the input to the :class:`prpy.SimulatedRobot` this method calls iteratively :meth:`prpy.DRLocalization.Localize` for k steps, solving the robot localization problem.

        :param x0: initial robot pose

        """
        xk_1 = x0
        xsk_1 = self.robot.xsk_1

        for self.k in range(self.kSteps):
            xsk = self.robot.fs(xsk_1, usk)  # Simulate the robot motion

            uk = self.GetInput()  # Get the input from the robot
            self.xk = self.Localize(xk_1,uk)  # Localize the robot

            xsk_1 = xsk  # current state becomes previous state for next iteration
            xk_1 = self.xk

            self.PlotTrajectory()  # plot the estimated trajectory

        plt.show()
        return

    def Log(self, xsk, xk):
        """
        Logs the results for later plotting.

        :param xsk: ground truth robot pose from the simulation
        :param xk: estimated robot pose
        """
        # initialize the log arrays if they don't exist
        if not hasattr(self, 'log_x'): self.log_x = np.zeros((xk.shape[0], self.kSteps))
        if not hasattr(self, 'log_xs'): self.log_xs = np.zeros((xsk.shape[0], self.kSteps))
        # if not hasattr(self, 'log_z'): self.log_z = np.zeros((zk.shape[0], self.kSteps))

        self.log_xs[0:xsk.shape[0], self.k] = xsk.T
        self.log_x[0:xk.shape[0], self.k] = xk.T
        self.k += 1

    def PlotXY(self):
        """
        Plots, in a new figure, the ground truth (orange) and estimated (blue) trajectory of the robot at the end of the Localization Loop.
        """
        fig, axs = plt.figure(), plt.axes()
        axs.plot(self.log_xs[0, 0:self.kSteps], self.log_xs[1, 0:self.kSteps], ls='-', c='orange')
        if self.plot_xy_estimation:
            axs.plot(self.log_x[0, 0:self.kSteps], self.log_x[1, 0:self.kSteps], ls='-', c='blue')

    def PlotTrajectory(self):
        """
        Plots the estimated trajectory (blue) of the robot during the localization process.
        """
        xk = self.xk
        if self.k % self.robot.visualizationInterval == 0:
            self.xTraj.append(xk[0, 0])
            self.yTraj.append(xk[1, 0])
            self.trajectory.pop(0).remove()
            self.trajectory = plt.plot(self.xTraj, self.yTraj, marker='.', color='blue', markersize=1)