from GFLocalization import *
from EKF import *
from DR_3DOFDifferentialDrive import *
from DifferentialDriveSimulatedRobot import *
from MapFeature import *

class EKF_3DOFDifferentialDriveInputDisplacement(GFLocalization, DR_3DOFDifferentialDrive, EKF):
    """
    This class implements an EKF localization filter for a 3 DOF Diffenteial Drive using an input displacement motion model incorporating
    yaw measurements from the compass sensor.
    It inherits from :class:`GFLocalization.GFLocalization` to implement a localization filter, from the :class:`DR_3DOFDifferentialDrive.DR_3DOFDifferentialDrive` class and, finally, it inherits from
    :class:`EKF.EKF` to use the EKF Gaussian filter implementation for the localization.
    """
    def __init__(self, kSteps, robot, *args):
        """
        Constructor. Creates the list of  :class:`IndexStruct.IndexStruct` instances which is required for the automated plotting of the results.
        Then it defines the inital stawe vecto mean and covariance matrix and initializes the ancestor classes.

        :param kSteps: number of iterations of the localization loop
        :param robot: simulated robot object
        :param args: arguments to be passed to the base class constructor
        """

        self.dt = 0.1  # dt is the sampling time at which we iterate the KF
        x0 = np.zeros((3, 1))  # initial state x0=[x y z psi u v w r]^T
        P0 = np.zeros((3, 3))  # initial covariance

        # this is required for plotting
        index = [IndexStruct("x", 0, None), IndexStruct("y", 1, None), IndexStruct("z", 2, 0), IndexStruct("yaw", 3, 1)]

        self.t_1 = 0
        self.t = 0
        self.Dt = self.t - self.t_1
        super().__init__(index, kSteps, robot, x0, P0, *args)
    def f(self, xk_1, uk):
        xk_bar=Pose3D.oplus(xk_1,uk)
        print(xk_bar.shape)
        return xk_bar

    def Jfx(self, xk_1,uk):
        # TODO: To be completed by the student
        J=Pose3D.J_1oplus(xk_1,uk)
        print(J.shape)
        print('jfx',J)

        return J

    def Jfw(self,xk_1):
        # TODO: To be completed by the student
        J=Pose3D.J_2oplus(xk_1)
       
        print('jfw',J.shape)

        return J
 


    def h_heading(self, xk):  #:hm(self, xk):
        # TODO: To be completed by the student
        # Obserse the heading of the robot
        h   = xk[2,0]
        return h  # return the expected observations

    def GetInput(self):
        """

        :return: uk,Qk:
        """
        # TODO: To be completed by the student
        
        
        
        uk, _     = self.robot.ReadEncoders()
        
        # Compute travel distance of 2 wheels [meter] from output of the encoder
        dl    = uk[0, 0] * (2*np.pi*self.wheelRadius/self.robot.pulse_x_wheelTurns)
        dr     = uk[1, 0] * (2*np.pi*self.wheelRadius/self.robot.pulse_x_wheelTurns)

        # Compute travel distance of the center point of robot between k-1 and k
        d       = (dl + dr) / 2.
        # Compute rotated angle of robot around the center point between k-1 and k
        delta_theta_k   = np.arctan2(dr - dl, self.wheelBase)

        # Compute xk from xk_1 and the travel distance and rotated angle. 
        uk              = np.array([[d],
                                    [0],
                                    [delta_theta_k]])
        
        # Given parameters
        delta_t = 0.1  # time step in seconds
        R_DVL = np.diag([0.1**2, 0.1**2])  # variance for the DVL
        sigma_yaw = np.deg2rad(5)  # convert yaw rate from degrees to radians per second

        # Calculate the components of Qk
        R_DVL = R_DVL 
        sigma_yaw = sigma_yaw**2 

        # Construct the Qk matrix
        Qk = np.zeros((3, 3))  # initialize a 4x4 matrix with zeros
        Qk[:2, :2] = R_DVL  # set the top-left 3x3 to R_DVL * delta_t^2
        Qk[2, 2] = sigma_yaw  # set the bottom-right element to sigma_yaw^2 * delta_t^2
        print(uk.shape)
        print(Qk.shape)
        return uk, Qk

    def GetMeasurements(self):  # override the observation model
        """

        :return: zk, Rk, Hk, Vk
        """
        # TODO: To be completed by the student
        # Read compass sensor
        zk, Rk  = self.robot.ReadCompass()

        # Compute H matrix
        Hk      = np.array([0., 0., 1.]).reshape((1,3))
        # Compute V matrix
        Vk      = np.diag([1.])
        # Raise flag got measurement
        if len(zk) != 0:
            self.headingData = True

        return zk, Rk, Hk, Vk


if __name__ == '__main__':

    M = [CartesianFeature(np.array([[-40, 5]]).T),
           CartesianFeature(np.array([[-5, 40]]).T),
           CartesianFeature(np.array([[-5, 25]]).T),
           CartesianFeature(np.array([[-3, 50]]).T),
           CartesianFeature(np.array([[-20, 3]]).T),
           CartesianFeature(np.array([[40,-40]]).T)]  # feature map. Position of 2 point features in the world frame.

    xs0 = np.zeros((6,1))  # initial simulated robot pose

    robot = DifferentialDriveSimulatedRobot(xs0, M)  # instantiate the simulated robot object
    kSteps = 1000

    xs0 = np.zeros((6, 1))  # initial simulated robot pose
    index = [IndexStruct("x", 0, None), IndexStruct("y", 1, None), IndexStruct("yaw", 2, 1)]

    x0 = np.zeros((3, 1))
    P0 = np.zeros((3, 3))

    dd_robot = EKF_3DOFDifferentialDriveInputDisplacement(kSteps,robot)  # initialize robot and KF
    dd_robot.LocalizationLoop(x0, P0, np.array([[0.5, 0.03]]).T)

    exit(0)