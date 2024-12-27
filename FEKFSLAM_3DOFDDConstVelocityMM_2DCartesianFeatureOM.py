from MapFeature import *
from FEKFMBL import *
from FEKFSLAM import *
from EKF_3DOFDifferentialDriveCtVelocity import *
from conversions import *
from FEKFSLAMFeature import *

class FEKFSLAM_3DOFDDConstVelocityMM_2DCartesianFeatureOM(FEKFSLAM2DCartesianFeature, FEKFSLAM, EKF_3DOFDifferentialDriveCtVelocity):
    """
    Feature EKF Map based Localization of a 3 DOF Differential Drive Mobile Robot (:math:`x_k=[^Nx_{B_k} ~^Ny_{B_k} ~^N\\psi_{B_k} ~]^T`) using a 2D Cartesian feature map (:math:`M=[[^Nx_{F_1} ~^Ny_{F_1}] ~[x_{F_2} ~^Ny_{F_2}] ~... ~[^Nx_{F_n} ~^Ny_{F_n}]]^T`),
    and a Constant Velocity Motion model with encoder readings. The class inherits from the following classes:
    * :class:`Cartesian2DMapFeature`: 2D Cartesian MapFeature using the Catesian coordinates for both, storage and landmark observations.
    * :class:`FEKFMBL`: Feature EKF Map based Localization class.
    * :class:`EKF_3DOFDifferentialDriveCtVelocity`: EKF for 3 DOF Differential Drive Mobile Robot with Constant Velocity Motion Model and encoder readings.
    """

    def __init__(self, *args):
      
        self.Feature = globals()["CartesianFeature"]
        self.Pose = globals()["Pose3D"]
        super().__init__( *args)

if __name__ == '__main__':

    M = [  CartesianFeature(np.array([[-40, 5]]).T),
           CartesianFeature(np.array([[-5, 40]]).T),
           CartesianFeature(np.array([[-5, 25]]).T),
           CartesianFeature(np.array([[-3, 50]]).T),
           CartesianFeature(np.array([[-20, 3]]).T),
           CartesianFeature(np.array([[40,-40]]).T) 
        ]  # feature map. Position of 2 point features in the world frame.

    xs0 = np.zeros((6,1))  # initial simulated robot pose

    robot = DifferentialDriveSimulatedRobot(xs0, M)  # instantiate the simulated robot object
    kSteps = 5000

    xs0 = np.zeros((6, 1))  # initial simulated robot pose
    index = [IndexStruct("x", 0, None), IndexStruct("y", 1, None), IndexStruct("yaw", 2, 1),
                 IndexStruct("u", 3, 2), IndexStruct("v", 4, 3), IndexStruct("yaw_dot", 5, None)]
    

    x0 = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).T
    P0 = np.diag(np.array([0.0, 0.0, 0.0, 0.5 ** 2, 0 ** 2, 0.05 ** 2]))

    alpha = 0.95

    

    robot = DifferentialDriveSimulatedRobot(xs0, M)  # instantiate the simulated robot object
    

    auv = FEKFSLAM_3DOFDDConstVelocityMM_2DCartesianFeatureOM(M, alpha, kSteps, robot)

    P0 = np.zeros((6, 6))
    usk=np.array([[0.5, 0.03]]).T


    # Observation
    znp = np.zeros((0,1))
    Rnp = np.zeros((0,0))  # empty matrix
 # Retrieve feature observations and their covariances
    zf, Rf, Hf, Vf= auv.GetFeatures()
    
    # Process each feature observation
    for obs, R in zip(zf, Rf):
        # Append the new observation to the observation array
        znp = np.vstack((znp, obs))
        # Append the new covariance to the covariance matrix
        if Rnp.size == 0:
            Rnp = R
        else:
            Rnp = scipy.linalg.block_diag(Rnp, R)
    

    
    x0, P0 = auv.AddNewFeatures(x0 , P0, znp, Rnp)
    
    robot.SetMap(M)
    auv.LocalizationLoop(x0, P0, usk)

    exit(0)