from FEKFSLAM import *
from FEKFMBL import *
from EKF_3DOFDifferentialDriveInputDisplacement import *
from Pose import *
from blockarray import *
from MapFeature import *
import numpy as np
from FEKFSLAMFeature import *

class FEKFSLAM_3DOFDD_InputVelocityMM_2DCartesianFeatureOM(FEKFSLAM2DCartesianFeature, FEKFSLAM, EKF_3DOFDifferentialDriveInputDisplacement):
    def __init__(self, *args):

        self.Feature = globals()["CartesianFeature"]
        self.Pose = globals()["Pose3D"]
        self.measurement_flag = False
        self.feature_flag = False
       
        super().__init__(*args)
        
        # self.nf = 6 # to be removed in part 3 


if __name__ == '__main__':

    M = [  
           CartesianFeature(np.array([[-40, 5]]).T),
           CartesianFeature(np.array([[-5, 40]]).T),
           CartesianFeature(np.array([[-5, 25]]).T),
           CartesianFeature(np.array([[-3, 50]]).T),
           CartesianFeature(np.array([[-20, 3]]).T),
           CartesianFeature(np.array([[40,-40]]).T)
           
        ]  # feature map. Position of 2 point features in the world frame.

    xs0 = np.zeros((6, 1))
    kSteps = 5000
    alpha = 0.99

    index = [IndexStruct("x", 0, None), IndexStruct("y", 1, None), IndexStruct("yaw", 2, 1)]
    

    robot = DifferentialDriveSimulatedRobot(xs0, M)  # instantiate the simulated robot object

    x0 = Pose3D(np.zeros((3, 1)))
    

    auv = FEKFSLAM_3DOFDD_InputVelocityMM_2DCartesianFeatureOM([], alpha, kSteps, robot)

    P0 = np.zeros((3, 3))
    usk=np.array([[0.5,0.03]]).T
    
    #Manually Check for Prediction and Update
    
    # x0 = np.array([[0],[0],[0],[-40],[5],[-5],[41],[-5],[25],[-3],[50],[-20],[3],[40],[-40]]) #Adding random features
    # # x0 = np.array([[0],[0],[0],[-41],[8],[-4],[40],[-5],[26],[-3],[50],[-30],[8],[45],[-30]]) #Adding random features
    
    # P0 = np.zeros((15, 15)) #random add the features covariance
    # np.fill_diagonal(P0, 0.01**2)
    # P0[0,0] = 0.01
    # P0[1,1] = 0.01
    # P0[2,2] = 0.01
    
    #observation
    znp = np.zeros((0,1))
    Rnp = np.zeros((0,0))  # empty matrix

     # Retrieve feature observations and their covariances
    zf, Rf, Hf, Vf= auv.GetFeatures()
    
    # Process each feature observation
    for obs, R in zip(zf, Rf):
        # Append the new observation to the observation array
        znp = np.vstack((znp, obs))
        print('znp',znp)
        # Append the new covariance to the covariance matrix
        if Rnp.size == 0:
            Rnp = R
        else:
            Rnp = scipy.linalg.block_diag(Rnp, R)
            print('Rnp',Rnp)
    
    x0, P0 = auv.AddNewFeatures(x0 , P0, znp, Rnp)
    
    robot.SetMap(M)
    auv.LocalizationLoop(x0, P0, usk)

    exit(0)
