from Localization import *
import numpy as np
from DifferentialDriveSimulatedRobot import *
from Pose import *

from Feature import *


class DR_3DOFDifferentialDrive(Localization):
    """
    Dead Reckoning Localization for a Differential Drive Mobile Robot.
    """
    def __init__(self, index, kSteps, robot, x0,*args):
        """
        Constructor of the :class:`prlab.DR_3DOFDifferentialDrive` class.

        :param args: Rest of arguments to be passed to the parent constructor
        """
        P0 = np.zeros((3, 3))

        super().__init__(index, kSteps, robot, x0, P0, *args)  # call parent constructor

        self.dt = 0.1  # dt is the sampling time at which we iterate the DR
        self.t_1 = 0.0  # t_1 is the previous time at which we iterated the DR
        self.wheelRadius = 0.1  # wheel radius
        self.wheelBase = 0.5  # wheel base
        self.robot.pulse_x_wheelTurns = 4096  # number of pulses per wheel turn

        # Compute Kn_inv matrix
        self.Kn_inv = self.robot.pulse_x_wheelTurns*self.dt/(2*np.pi*self.wheelRadius) * np.array([[1, -self.wheelBase/2], [1, self.wheelBase/2]])

    def Localize(self, xk_1, uk):  # motion model
        """
        Motion model for the 3DOF (:math:`x_k=[x_{k}~y_{k}~\psi_{k}]^T`) Differential Drive Mobile robot using as input the readings of the wheel encoders (:math:`u_k=[n_L~n_R]^T`).

        :parameter xk_1: previous robot pose estimate (:math:`x_{k-1}=[x_{k-1}~y_{k-1}~\psi_{k-1}]^T`)
        :parameter uk: input vector (:math:`u_k=[u_{k}~v_{k}~w_{k}~r_{k}]^T`)
        :return xk: current robot pose estimate (:math:`x_k=[x_{k}~y_{k}~\psi_{k}]^T`)
        """

        # Store previous state and input for Logging purposes
        self.etak_1 = xk_1  # store previous state
        self.uk = uk  # store input
       
        # TODO: to be completed by the student
        # Compute travel distance of 2 wheels [meter] from output of the encoder
        d_L     = uk[0, 0] * (2*np.pi*self.wheelRadius/self.robot.pulse_x_wheelTurns)
        d_R     = uk[1, 0] * (2*np.pi*self.wheelRadius/self.robot.pulse_x_wheelTurns)

        # Compute travel distance of the center point of robot between k-1 and k
        d       = (d_L + d_R) / 2.
        # Compute rotated angle of robot around the center point between k-1 and k
        delta_theta_k   = np.arctan2(d_R - d_L, self.wheelBase)
        
        # Compute xk from xk_1 and the travel distance and rotated angle. Got the equations from chapter 1.4.1: Odometry 
        xk              = np.array([[xk_1[0, 0] + d * np.cos(xk_1[2, 0] + delta_theta_k)],
                                    [xk_1[1, 0] + d * np.sin(xk_1[2, 0] + delta_theta_k)],
                                    [xk_1[2, 0] + delta_theta_k]])

        return xk

    def GetInput(self):
        """
        Get the input for the motion model. In this case, the input is the readings from both wheel encoders.

        :return: uk:  input vector (:math:`u_k=[n_L~n_R]^T`)
        """

        # TODO: to be completed by the student
        # Get output of encoder via ReadEncoder() function
        uk, Rsk     = self.robot.ReadEncoders()
        
        return uk

