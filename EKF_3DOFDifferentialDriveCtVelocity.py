from GFLocalization import *
from EKF import *
from DR_3DOFDifferentialDrive import *
from DifferentialDriveSimulatedRobot import *
from MapFeature import *

class EKF_3DOFDifferentialDriveCtVelocity(GFLocalization, DR_3DOFDifferentialDrive, EKF):

    def __init__(self, kSteps, robot, *args):

        self.x0 = np.zeros((6, 1))  # initial state x0=[x y z psi u v w r]^T
        self.P0 = np.zeros((6, 6))  # initial covariance

        # this is required for plotting
        self.index = [IndexStruct("x", 0, None), IndexStruct("y", 1, None), IndexStruct("yaw", 2, 1),
                 IndexStruct("u", 3, 2), IndexStruct("v", 4, 3), IndexStruct("yaw_dot", 5, None)]

        # TODO: To be completed by the student
        self.t_1 = 0
        self.t = 0
        self.Dt = self.t - self.t_1
        super().__init__(self.index, kSteps, robot, self.x0, self.P0, *args)

    def f(self, xk_1, uk):
        # TODO: To be completed by the student
        vk = xk_1[3:6].reshape((3,1))
        uk     = vk * self.dt
        xk_bar = Pose3D.oplus(xk_1[0:3].reshape((3,1)), uk.reshape((3,1)))
        
        xk_bar = np.block([[xk_bar], [vk]])
        return xk_bar

    def Jfx(self, xk_1, uk):
        # TODO: To be completed by the student
        uk = xk_1[3:6].reshape((3,1)) * self.dt

        J_eta_x = np.block([Pose3D.J_1oplus(xk_1[0:3].reshape((3,1)), uk.reshape((3,1))), Pose3D.J_2oplus(xk_1[0:3].reshape((3,1)))*self.dt])
        J_v_x  = np.block([np.zeros((3,3)), np.diag(np.ones(3))])

        J = np.block([[J_eta_x], [J_v_x]])
        return J

    def Jfw(self, xk_1):
        # TODO: To be completed by the student
        J  = np.block([[Pose3D.J_2oplus(xk_1[0:3].reshape((3,1)))*self.dt*self.dt], [np.diag(np.ones(3))*self.dt]])
        return J

    def h_heading(self, xk):  #:hm(self, xk):
        # TODO: To be completed by the student
        H       = np.zeros((0,6))
        
        if self.yaw == True:
            H_yaw   = np.array([[0,0,1,0,0,0]])
            H = np.block([[H], [H_yaw]])
        if self.vel == True:
            H_n= np.array([[  0,0,0,self.Kn_inv[0,0],0,self.Kn_inv[0,1]],
                            [ 0,0,0,self.Kn_inv[1,0],0,self.Kn_inv[1,1]]])
            H = np.block([[H], [H_n]])
        # Combine number of pulses read from the left and right wheel encoders
        

        h = H @ xk[0:6]
        
        return h  # return the expected observations

    def GetInput(self):
        """

        :return: uk,Qk:
        """
        # TODO: To be completed by the student
 
        uk , Qk = DR_3DOFDifferentialDrive.GetInput(self)  # gets uk,Qk from DR_3DOFDiffer_Robot Class  
        uk = None      #  
        Qk = np.diag(np.array([0.1 ** 2, 0.01 ** 2, np.deg2rad(1) ** 2]))  #  acceleration noise                           
        return uk, Qk

    def GetMeasurements(self):  # override the observation model
        """

        :return:
        """
        zk = np.zeros((0, 1))  # empty vector
        Rk = np.zeros((0, 0))  # empty matrix

        self.yaw = False
        self.vel = False

        Hk, Vk = np.zeros((0,6)), np.zeros((0,0))
        H_yaw, V_yaw = np.array([[0,0,1,0,0,0]]), np.eye(1)

        z_yaw, sigma2_yaw = self.robot.ReadCompass()

        if z_yaw.size>0:  # if there is comapss  measurement
            zk, Rk = np.block([[zk], [z_yaw]]), scipy.linalg.block_diag(Rk, sigma2_yaw)
            Hk, Vk = np.block([[Hk], [H_yaw]]), scipy.linalg.block_diag(Vk, V_yaw)
            self.yaw = True
            self.measurement_flag = True

        n, Rn = self.robot.ReadEncoders(); L=0; R=1  # read sensors

        if n.size>0:  # if there is encoder measurement
            self.vel=True

            H_n= np.array([[ 0,0,0,self.Kn_inv[0,0],0,self.Kn_inv[0,1]],
                            [ 0,0,0,self.Kn_inv[1,0],0,self.Kn_inv[1,1]]])

            zk, Rk = np.block([[zk], [n]]), scipy.linalg.block_diag(Rk, Rn)
            Hk, Vk = np.block([[Hk], [H_n]]), scipy.linalg.block_diag(Vk, np.eye(2))
            self.measurement_flag = True

        if zk.shape[0] == 0: # if there is no measurement
            return [], [],[], []
        else:
            return zk, Rk, Hk, Vk

# if __name__ == '__main__':

#     M = [CartesianFeature(np.array([[-40, 5]]).T),
#            CartesianFeature(np.array([[-5, 40]]).T),
#            CartesianFeature(np.array([[-5, 25]]).T),
#            CartesianFeature(np.array([[-3, 50]]).T),
#            CartesianFeature(np.array([[-20, 3]]).T),
#            CartesianFeature(np.array([[40,-40]]).T)]  # feature map. Position of 2 point features in the world frame.

#     xs0 = np.zeros((6,1))  # initial simulated robot pose

#     robot = DifferentialDriveSimulatedRobot(xs0, M)  # instantiate the simulated robot object
#     kSteps = 5000

#     xs0 = np.zeros((6, 1))  # initial simulated robot pose
#     index = [IndexStruct("x", 0, None), IndexStruct("y", 1, None), IndexStruct("yaw", 2, 1),
#                  IndexStruct("u", 3, 2), IndexStruct("v", 4, 3), IndexStruct("yaw_dot", 5, None)]

#     x0 = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).T
#     P0 = np.diag(np.array([0.0, 0.0, 0.0, 0.5 ** 2, 0 ** 2, 0.05 ** 2]))

#     dd_robot = EKF_3DOFDifferentialDriveCtVelocity(kSteps, robot)  # initialize robot and KF
#     dd_robot.LocalizationLoop(x0, P0, np.array([[0.5, 0.03]]).T)  # run localization loop

#     exit(0)