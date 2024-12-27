from MapFeature import *
from blockarray import *
class FEKFSLAMFeature(MapFeature):
 
    
    """
    This class extends the :class:`MapFeature` class to implement the Feature EKF SLAM algorithm.
    The  :class:``MapFeature`` class is a base class providing support to localize the robot using a map of point features.
    The main difference between FEKMBL and FEAKFSLAM is that the former uses the robot pose as a state variable,
    while the latter uses the robot pose and the feature map as state variables. This means that few methods provided by
    class need to be overridden to gather the information from state vector instead that from the deterministic map.
    # """
    def hfj(self, xk_bar, Fj):
        """
        Computes the expected observation of a single feature indexed by Fj 
        based on the current state estimate.

        :param xk_bar: Mean of the predicted state vector.
        :param Fj: Index of the feature to observe.
        :return: Expected observation of the feature.
        """
        self.xF_dim=2
        # Assuming the first 3 elements of xk_bar are the robot pose 
        NxB = xk_bar[:3].reshape((3, 1))

        # Calculate the start index of the feature in the state vector
        feature_index = 3 + Fj * self.xF_dim

        # Extract the state of the specified feature
        # Ensure to reshape the feature state correctly based on its dimension 
        NxF = xk_bar[feature_index:feature_index + self.xF_dim].reshape((self.xF_dim, 1))

        # Compute the expected observation for the feature
       
        inverse_pose = Pose3D.ominus(NxB)  
        relative_position = CartesianFeature(NxF).boxplus(inverse_pose) 
        
        # Convert the relative position to the observation space
        hfj = self.s2o(relative_position)  
        print('hfj_EKF_SLAM',hfj)
        
        return hfj

 
    def Jhfjx(self, xk, Fj):
        """
        Jacobian of the single feature direct observation model with respect to the state vector.

        :param xk: state vector mean
        :param Fj: map index of the observed feature
        :return: Jacobian matrix
        """
        # xk = [xr xf0 xf1 .... xfn]
        
       # Jhx = [Jhxr 0 Jhxf1 ..0.. 0 ]
        self.xF_dim=2
        xF_dim=self.xF_dim# Dimension of a single feature
        NxB = xk[:3].reshape((3, 1))  # Extract robot pose
        
        # Initialize J with zeros. The size depends on the state vector xk and observation dimension.
        J = np.zeros((2, len(xk)))  # Assuming 2D observation 

        # Calculate the start index of the feature in the state vector
        feature_index = 3 + Fj * xF_dim

        # Extract the state of the specified feature
        NxF = CartesianFeature(xk[feature_index:feature_index + xF_dim].reshape((xF_dim, 1)))

        # Construct J1 for the robot pose part
        J1 = self.J_s2o(NxF.boxplus(Pose3D.ominus(NxB))) @ NxF.J_1boxplus(Pose3D.ominus(NxB)) @ Pose3D.J_ominus(NxB)
        
        # Construct J2 for the feature part
        J2 = self.J_s2o(NxF.boxplus(Pose3D.ominus(NxB))) @ NxF.J_2boxplus(Pose3D.ominus(NxB))

        # Assign J1 and J2 
        J[:, :3] = J1  # Robot pose part
        J[:, feature_index:feature_index + xF_dim] = J2  #Feature part
        print(J)

        return J
        
        
        
        
        
        
        
        
        
        
         
            
            
            

class FEKFSLAM2DCartesianFeature(FEKFSLAMFeature, Cartesian2DMapFeature):
    """
    Class to inherit from both :class:`FEKFSLAMFeature` and :class:`Cartesian2DMapFeature` classes.
    Nothing else to do here (if using s2o & o2s), only needs to be defined.
    """
    pass


