import numpy as np
from Feature import *

class MapFeature:
    """
    This class provides the functionality required to use Map Features for Robot     Localization. It has methods for reading
    the feature pose using the robot the sensors (:meth:`GetFeatures`), as well as for computing its:

        * observation model  (:meth:`hf`),
        * inverse observation model (:meth:`g`)
        * and all the required Jacobians (:meth:`Jhfx`, :meth:`Jhfv`, :meth:`Jgx` and :meth:`Jgv`).

    When mapped, a feature may involve 2 different representations:

    * **The observation representation:** which is the representation used by the sensors to observe the feature.
    * **The storage representation:** which is the representation used to store the feature in the map within the state vector.

    For instance, a feature may be observed in polar coordinates but stored in Cartesian coordinates. In this case, the
    observation representation is the Polar coordinates and the storage representation is the Cartesian coordinates. The
    class provides methods to convert from one representation to the other (:meth:`s2o` and :meth:`o2s`) and their corresponding
    Jacobians (:meth:`J_s2o` and :meth:`J_o2s`). By default, the observation representation is the same as the storage representation,
    but this behaviour may be overriden in child classes.
    """

    def __init__(self,*args):
      

        super().__init__(*args)
        
        
    def GetFeatures(self):
        """
        Reads the Feature observations from the sensors. For all features within the field of view of the sensor, the
        method returns the list of robot-related poses, the covariance of their corresponding observation noise, the
        corresponding observation matrix and the noise Jacobian matrix.
        **This is a pure virtual method that must be overriden in child classes**.

        :returns: vector of features observations in the B-Frame and the covariance of their corresponding noise.

            * :math:`z_k=[^Bx_{F_i}^T \\cdots ^Bx_{F_j}^T \\cdots ^Bx_{F_k}^T]^T`
            * :math:`R_k=block\\_diag([R_{F_i} \\cdots R_{F_j} \\cdots R_{F_k}])`
            * :math:`H_k=block\\_diag([H_{F_i} \\cdots H_{F_j} \\cdots H_{F_k}])`
            * :MATH:`V_K=I_{z_{nf} \\times z_{nf}}`
        """
        pass

    def s2o(self, v):
        """
        Conversion function from the storage representation to the observation representation.
        By default, it returns the same vector as the one provided as input, assuming that the observation representation is the same as the storage representation.
        In case it is not, this method must be overriden in the child class.

        :param v: vector in the storage representation
        :return: vector in the observation representation
        """
        # TODO: To be implemented by the student

        
        print('v',v)
        return v

    def o2s(self, v):
        """
        Conversion function from the observation representation to the storage representation. By default, it returns the
        same vector as the one provided as input, assuming that the observation representation is the same as the storage representation.
        In case it is not, this method must be overriden in child classes.

        :param v: vector in the observation representation
        :return: vector in the storage representation
        """
        # TODO: To be implemented by the student

    

        return v

    def J_s2o(self, v):
        """
        Jacobian of the conversion function from the storage representation to the observation representation.
        By default, it returns the identity matrix, assuming that the observation representation is the same as the storage representation.
        In case it is not, this method must be overriden in the derived class.

        :param v: vector in the storage representation
        :return: Jacobian of the conversion function from the storage representation to the observation representation
        """
       
        # TODO: To be implemented by the student
        
        # Check if v is a scalar (has an empty shape tuple)
        if np.isscalar(v) or np.shape(v) == ():
            # For a scalar, the Jacobian would be 1, since d(v)/d(v) = 1
            J = np.array([[1]])
        else:
            # For vectors and matrices, create an identity matrix of appropriate size
            dim = np.shape(v)[0]  # This will work for a vector or for a square matrix
            J = np.eye(dim)
    
        print('Js20',J)
        return J

    def J_o2s(self, v):
        """
        Jacobian of the conversion function from the observation representation to the storage representation.
        By default, it returns the identity matrix, assuming that the observation representation is the same as the storage representation.
        In case it is not, this method must be overriden in the derived class.

        :param v: vector in the observation representation
        :return: Jacobian of the conversion function from the observation representation to the storage representation
        """
        # TODO: To be implemented by the student
        J = np.eye(np.shape(v)[0])
        return J


    def hf(self, xk, H):  # Observation function for al zf observations
        """
        This is the direct observation model, implementing the feature observation equation for the data
        association hypothesis :math:`\mathcal{H}`, the features observation vector :math:`z_f, the state vector :math:`x_k`,
        and the observation noise :math:`v_k`:

        .. math::
            \mathcal{H}&=[F_a~\\cdots~F_b~\\cdots~F_c]\\\\
            z_f&=[z_{f_1}^T~\\cdots~z_{f_i}^T~\\cdots~z_{f_{n_{zf}}}^T]^T\\\\
            x_k&=[^Nx_B^T~x_{rest}^T]^T\\\\
            v_k&=[v_{f1_k}^T~\\cdots~v_{fi_k}^T~\\cdots~v_{fn_{zf_k}}^T]^T\\\\\
            z_f&=h_f(x_k,v_k) \\\\
            :label: eq-hf

        which may be expanded as follows:

        .. math::
            \\begin{bmatrix} z_{f_1} \\\\ \\vdots  \\\\ z_{f_i} \\\\ \\vdots \\\\ z_{n_{zf}} \\end{bmatrix} = \\begin{bmatrix} h_{F_a}(x_k,v_k) \\\\ \\vdots \\\\ h_{F_b}(x_k,v_k) \\\\ \\vdots \\\\ h_{Fc}(x_k,v_k) \\end{bmatrix}
            = \\begin{bmatrix} s2o(\\ominus ^Nx_B \\boxplus^Nx_{F_{a}})+ v_{f1_k}\\\\ \\vdots \\\\  s2o(\\ominus ^Nx_B \\boxplus^Nx_{F_{b}})+ v_{fi_k}\\\\ \\vdots \\\\ s2o(\\ominus ^Nx_B \\boxplus ^Nx_{F_{c}}) + v_{fn_{zf}}\\end{bmatrix}
            :label: eq-hf-element-wise

        being :math:`h_{F_j}(\cdot)` (:meth:`hfj`) the observation function (eq. :eq:`eq-hfj`) for the data association hypothesis :math:`\\mathcal{H}` and  :meth:`s2o` the conversion
        function from the storage representation to the observation one.

        The method computes the expected observations :math:`h_{f}` for the observed features contained within the :math:`z_{f}` features observation vector.
        To do it, it iterates over each feature observation :math:`z_{f_i}` calling the method :meth:`hfj` for its corresponding associated feature :math:`\mathcal{H}_i=F_j`
        to compute the expected observation :math:`h_{F_j}`, collecting all them in the returned vector.

        :param xk: state vector mean :math:`\\hat x_k`.
        :return: vector of expected features observations corresponding to the vector of observed features :math:`z_f`.
        """
        # TODO: To be implemented by the student
     
        hf = np.zeros((0,1))
        for i in range(len(H)):           
            if(H[i] != 0):
               hf = np.block([[hf],[self.hfj(xk, H[i]-1)]])
        return hf

    def Jhfx(self, xk):  # Jacobian wrt x of the feature observation function for all zf observations
        """
        Computes the Jacobian of the feature observation function :meth:`hf` (eq. :eq:`eq-hf`), with respect to the state vector :math:`\\bar{x}_k`:

        .. math::
            J_{hfx}=\\frac{\\partial h_f(x_k,v_k)}{\\partial x_k}=\\begin{bmatrix} \\frac{\\partial h_{F_{a}}(x_k,v_k)}{\\partial x_k} \\\\ \\vdots \\\\ \\frac{\\partial h_{F_{b}}(x_k,v_k)}{\\partial x_k} \\\\ \\vdots \\\\ \\frac{\\partial h_{F_{c}}(x_k,v_k)}{\\partial x_k} \\end{bmatrix}
            =\\begin{bmatrix} J_{h_{F_a}} \\\\ \\vdots \\\\ J_{h_{F_b}} \\\\ \\vdots \\\\ J_{h_{F_c}} \\end{bmatrix}
            :label: eq-Jhfx

        where :math:`J_{h_{F_j}}` is the Jacobian of the observation function :meth:`hfj` (eq. :eq:`eq-Jhfjx`) for the feature :math:`F_j`and observation :math:`z_{f_i}`.
        To do it, given a vector of observations :math:`z_f=[z_{f_1}~\\cdots~z_{f_i}~\\cdots~z_{f_{n_{zf}}}]` this method iterates over each feature observation :math:`z_{f_i}` calling the method :meth:`Jhfj` to compute
        the Jacobian of the observation function for each feature observation (:math:`J_{hfj}`), collecting all them in the returned Jacobian matrix :math:`J_{hfx}`.

        :param xk: state vector mean :math:`\\hat x_k`.
        :return: Jacobian of the observation function :meth:`hf` with respect ro the robot pose :math:`J_{hfx}=\\frac{\\partial h_f(\\bar{x}_k,v_{f_k})}{\\bar{x}_k}`
        """

        # TODO: To be implemented by the student
            # Initialize list to store Jacobians of each feature observation
        J_list = [self.Jhfjx(xk, 0)]
        

        # Iterate over each feature observation, calling Jhfjx method
        for i in range(1, self.xF_dim):
            J_list.append(self.Jhfjx(xk, i))

        # Concatenate all Jacobians into a single matrix
        J = np.vstack(J_list)
        print('Jkhhhh',J)
    
        return J
    

    def Jhfv(self, xk):  # Jacobian wrt v of the observation function for a feature
        """
        Computes the Jacobian of the observation function :meth:`hf` (eq. :eq:`eq-hf`) with respect to the observation noise :math:`v_k`.
        Normally, the observation noise in the observation B-Frame is linear (see eq. :eq:`eq-hf-element-wise`) so the Jacobian is the identity matrix.

        .. math::
            J_{hfv}&=\\frac{\\partial h_f(x_k,v_k)}{\\partial v_k}\\\\
            &=\\begin{bmatrix} \\frac{\\partial h_{F_a}(x_k,v_k)}{\\partial v_k} \\\\ \\vdots \\\\ \\frac{\\partial h_{F_b}(x_k,v_k)}{\\partial v_k} \\\\ \\vdots \\\\ \\frac{\\partial h_{F_c}(x_k,v_k)}{\\partial v_k} \\end{bmatrix}
            =\\begin{bmatrix}
                \\frac{\\partial h_{F_a}(x_k,v_k)}{\\partial v_{f1_k}} &  \\cdots & 0 & 0 & 0  & \\cdots & 0 \\\\
                \\vdots & \\ddots & \\vdots & \\vdots & \\vdots & \\ddots & \\vdots \\\\
                0 & \\cdots &  0 & \\frac{\\partial h_{F_b}(x_k,v_k)}{\\partial v_{fi_k}} &  0 & \\cdots & 0 \\\\
                \\vdots & \\ddots & \\vdots & \\vdots & \\vdots & \\ddots & \\vdots \\\\
                0 & \\cdots & 0 & 0 & 0 & \\frac{\\partial h_{F_c}(x_k,v_k)}{\\partial v_{fn_{zf_k}}}  \\\\
            \\end{bmatrix}= I_{n_{zf}\\times n_{zf}}
            :label: eq-Jhfv

        If it is not the case, this method must be overriden.

        :param xk: state vector mean :math:`\\hat x_k`.
        :return: Jacobian of the observation function :meth:`hf` with respect ro the observation noise :math:`v_k` :math:`J_{hfv}=I_{n_{zf}\\times n_{zf}}`
        """
        # TODO: To be implemented by the student
        # Compute the Jacobian matrix as an identity matrix
        J = np.eye(self.nf * self.xF_dim)
    
        return J

    def hfj(self, xk_bar, Fj):  # Observation function for zf_i and x_Fj
        """
        This is the direct observation model for a single feature observation  :math:`z_{f_i}` , so it implements its related
        observation function (see eq. :eq:`eq-hfj`). For a single feature observation :math:`z_{f_i}` of the feature :math:`^Nx_{F_{H_i}}` the method computes its
        expected observation from the current robot pose :math:`^Nx_B`.
        This function uses a generic implementation through the following equation:

        .. math::
            z_{f_i}=h_{F_j}(x_k,v_k)=s2o(\\ominus ^Nx_B \\boxplus ^Nx_{F_{j}}) + v_{fi_k}
            :label: eq-hfj

        Where :math:`^Nx_B` is the robot pose included within the state vector (:math:`x_k=[^Nx_B^T~x_{rest}^T]^T`)  and :meth:`s2o` is a conversion
        function from the store representation to the observation representation.

        The method is called by :meth:`hf` to compute the expected observation for each feature
        observation contained in the observation vector :math:`z_f=[z_{f_1}^T~\\cdots~z_{f_i}^T~\\cdots~z_{f_{n_zf}}^T]^T`.

        :param xk_bar: mean of the predicted state vector
        :param Fj: map index of the observed feature: :math:`^Nx_{F_j}=self.M[Fj]`
        :return: expected observation of the feature :math:`^Nx_{F_j}`
        """

        # h(xk_bar,vk)=(-) xk_bar) [+] x_Fj + vk

        # TODO: To be implemented by the student
         
        # Extract robot pose from the state vector
        NxB = xk_bar[:3, 0].reshape((3, 1))
        print('NxB_shape', NxB.shape)
        print('Fj_shape', Fj)

        # Compute the expected observation for the feature
        BxF=self.M[Fj]
        print('Bxf',BxF)
        print('BxF_shape',BxF.shape)
        hfj = self.s2o(BxF.boxplus(Pose3D.ominus(NxB)))
        print('__hfj', hfj.shape)
        return hfj
     

    def Jhfjx(self, xk, Fj):  # Jacobian wrt x of the observation function for feature observation i
        
        """
        Jacobian of the single feature direct observation model :meth:`hfj` (eq. :eq:`eq-hfj`)  with respect to the state vector :math:`\\bar{x}_k`:

        .. math::
            x_k&=[^Nx_B^T~x_{rest}^T]^T\\\\
            ^Nx_B&= F \\cdot x_k ; F=\\begin{bmatrix} I_{p\\times p} & 0_{p\\times np} \\end{bmatrix}\\\\
            J_{hfjx}&=\\frac{\\partial h_{F_j}({x}_k,v_k)}{\\partial {x}_k}=
            \\frac{\\partial s2o(\\ominus F \\cdot x_k \\boxplus ^Nx_{F_j})+v_{fi_k}}{\\partial {x}_k}\\\\&=
             J_{s2o}(\\ominus ^Nx_B \\boxplus^Nx_{F_{j}}) J_{1\\boxplus}(\\ominus ^Nx_B,^Nx_{F_{j}} ) J_{\\ominus}(^Nx_B ) F
            :label: eq-Jhfjx

        where :math:`p` is the dimension of the robot pose :math:`^Nx_B` and :math:`np` is the dimension of the rest of the state vector :math:`x_{rest}`.

        :param xk: state vector mean
        :param Fj: map index of the observed feature
        :return: Jacobian matrix defined in eq. :eq:`eq-Jhfjx`
        """
        # J_hfjx = J_s2o @ J_1[+] * J_(-)
        # TODO: To be implemented by the student
        # Dimensions of the pose and the state
        dim = 3

        # Extract robot pose from the state vector
        NxB = xk[0:dim, 0].reshape((dim, 1))

        # F matrix converts vector from filter state to pose
        F = np.hstack([np.eye(dim), np.zeros((dim, len(xk)- dim))])

        # Compute the Jacobian matrix
        BxF=self.M[Fj]
        J = (self.J_s2o(BxF.boxplus(Pose3D.ominus(NxB))) 
         @ CartesianFeature.J_1boxplus(BxF, Pose3D.ominus(NxB)) 
         @ Pose3D.J_ominus(NxB) 
         @ F)
        print('J_shape',J.shape)
        return J
     

    def g(self, xk, BxFj):  # xBp [+] (BxFj + vk)
        """
        This method provides a generic implementation of the inverse observation model. It computes the feature pose in the N-Frame :math:`^Nx_{F_j}` by compounding the robot pose :math:`^Nx_B`, from where the observation was taken, with the  B-Frame referenced feature
        observation :math:`^Bx_{F_j}`:

        .. math::
            ^Nx_{F_j} = ^Nx_B \\boxplus o2s(^Bx_{F_j} +v_k)
            :label: eq-g

        In this case, :meth:`o2s` is the conversion function converting from the observation space to the representation one.
        It is worth noting that the robot pose :math:`^Nx_B` is included within the state vector :math:`x_k` but might not be the whole state vector.
        For instance, in some cases the state vector may include as well the robot velocity :math:`x_k=[^Nx_B^T~^B\\nu_k^T]^T`.
        Note that the :meth:`g` works with a single feature observation, instead than with a vector of feature observations.

        :param xk: mean state vector containing the robot pose :math:`^Nx_B` from where the observation was taken
        :param BxFj: feature observation in the B-Frame :math:`^Bx_{F_j}`
        :return: mean feature pose in the N-Frame :math:`^Nx_{F_j}`
        """

        # TODO: To be implemented by the student
         # Dimensions of the pose and the state
        NxB = self.GetRobotPose(xk)
        BxFj = self.Feature(BxFj)
        NxFj =(self.o2s(BxFj)).boxplus(NxB)
        

        return NxFj

      

    def Jgx(self, xk, BxFj):  # Jacobian wrt xk of the inverse sensor model for a single feature observation
        """
        Jacobian of the inverse observation model :meth:`g`, with respect to the state vector :math:`x_k`.
        According to the generic implementation of the inverse observation model :meth:`g` eq. :eq:`eq-g`, the Jacobian is computed as follows:

        .. math::
            J_{gx}=\\frac{\\partial g(^Nx_B,^Bx_{F_j},v_k)}{\\partial x_k}=\\begin{bmatrix} J_{1\\boxplus}(^Nx_B,o2s(^Bx_{F_j})) & 0  \\end{bmatrix}
            :label: eq-Jgx

        The zero submatrix, if present, corresponds to the derivate with respect to the non-positional elements of the
        state vector, for instance the robot velocity :math:`^B\\nu_k`, in case this was included within the state vector.
        If the state vector only containes the pose, then the 0 submatix vanishes.

        :param xk_bar: predicted state vector
        :return: Jacobian of the inverse observation model :meth:`g` with respect to the state vector (eq. :eq:`eq-Jgx`)
        """

        # TODO: To be implemented by the student
        # # Dimensions of the of the filter state and the pose
           # Get dimensionality of the filter state and the pose
        xB_dim = self.xB_dim
        xBpose_dim = self.xBpose_dim
        NxB = self.GetRobotPose(xk)
        xF_dim = np.shape(BxFj)[0]       
        Jp = (self.o2s(self.Feature(BxFj))).J_1boxplus(NxB)
        Jnp =  np.zeros((xF_dim, xB_dim-xBpose_dim))       
        J = np.block([Jp,Jnp])
        return J

          
     

    def Jgv(self, xk, BxFj):
        """
        Jacobian of the inverse observation model :meth:`g`, with respect to the observation noise :math:`v_k`.
        According to the generic implementation of the inverse observation model :meth:`g` eq. :eq:`eq-g`, the Jacobian is computed as follows:

        .. math::
            J_{gv}=\\frac{\\partial g(^Nx_k,^Bx_{F_j},v_k)}{\\partial v_k}=J_{2\\boxplus}(^Nx_B,o2s(^Bx_{F_j})) J_{o2s}(^Bx_{F_j})
            :label: eq-Jgv

        :param xk: state vector containing the robot pose :math:`^Nx_B` from where the observation was taken
        :param BxFj: feature observation in the B-Frame :math:`^Bx_{F_j}`
        :return: Jacobian of the inverse observation model :meth:`g` with respect to the observation noise :math:`J_{gv}` (see eq. :eq:`eq-Jgv`).
        """
        # TODO: To be implemented by the student
        xBpose_dim = self.xBpose_dim
        # Get Pose vector from the filter state
        NxB = xk[0:xBpose_dim,0].reshape((xBpose_dim,1))            
        J = (self.o2s(self.Feature(BxFj))).J_2boxplus( NxB) @ self.J_o2s(BxFj)
        return J



class Cartesian2DMapFeature(MapFeature):
    """
    This class inherits from the :class:`MapFeature` and implements a 2D Cartesian feature model for the MBL problem. The Cartesian coordinates are used for both,
    observing the feature and for its storage within the map. This class overrides the :meth:`GetFeatures` method to read
    the 2D Cartesian Features from the robot.
    """

    def GetFeatures(self):
        """
        Reads the Features observations from the sensors. For all features within the field of view of the sensor, the
        method returns the list of robot-related poses and the covariance of their corresponding observation noise in **2D Cartesian coordinates**.

        :return zk, Rk: list of Cartesian features observations in the B-Frame and the covariance of their corresponding observation noise:
            * :math:`z_k=[^Bx_{F_i}^T \\cdots ^Bx_{F_j}^T \\cdots ^Bx_{F_k}^T]^T`
            * :math:`R_k=block\\_diag([R_{F_i} \\cdots R_{F_j} \\cdots R_{F_k}])`

        """
        # TODO: To be implemented by the student
        zf, Rf = self.robot.ReadCartesian2DFeature()
        
        
        # Initialize Hf and Vf as empty lists
        Hf, Vf = [], []

        # Flag to indicate if features are observed
        self.featureData = len(zf) != 0

    
        return zf, Rf, Hf, Vf
    
class Cartesian2DStoredPolarObservedMapFeature(MapFeature):
    """
    This class inherits from the :class:`MapFeature` and implements a 2D Cartesian feature model for the MBL problem. The Cartesian coordinates are used for both,
    observing the feature and for its storage within the map. This class overrides the :meth:`GetFeatures` method to read
    the 2D Cartesian Features from the robot.
    """

    def GetFeatures(self):
        """
        Reads the Features observations from the sensors. For all features within the field of view of the sensor, the
        method returns the list of robot-related poses and the covariance of their corresponding observation noise in **2D Cartesian coordinates**.

        :return zk, Rk: list of Cartesian features observations in the B-Frame and the covariance of their corresponding observation noise:
            * :math:`z_k=[^Bx_{F_i}^T \\cdots ^Bx_{F_j}^T \\cdots ^Bx_{F_k}^T]^T`
            * :math:`R_k=block\\_diag([R_{F_i} \\cdots R_{F_j} \\cdots R_{F_k}])`

        """
        # TODO: To be implemented by the student
        zf, Rf = self.robot.ReadPolar2DFeature()
               
        # Initialize Hf and Vf as empty lists
        Hf, Vf = [], []

        # Flag to indicate if features are observed
        self.featureData = len(zf) != 0
       
       
        return zf, Rf, Hf, Vf
    
    def s2o(self, v):
     
        # TODO: To be implemented by the student
        #s20= [sqrt(x²+y²)
        #      atan2(y,x)]
        Observation_representation = CartesianFeature(np.block([np.sqrt(v[0]**2+v[1]**2), np.arctan2(v[1], v[0])]).reshape(2,1))
        return Observation_representation
    
    def o2s(self, v):
   
        # TODO: To be implemented by the student
        #o2s=[row*cos(theta)
        #     row*sin(theta)]
        
        Storage_representation = CartesianFeature(np.block([v[0]*np.cos(v[1]),v[0]*np.sin(v[1])]).reshape(2,1))

        return Storage_representation
    
    def J_s2o(self, v):
 
        # TODO: To be implemented by the student
        #Js20=[x/sqrt(x²+y²)     y/sqrt(x²+y²)
        #            -y/(x²+y²)  -x/(x²+y²)]
        J = np.block([[v[0]/np.sqrt(v[0]**2+v[1]**2), v[1]/np.sqrt(v[0]**2+v[1]**2)],
                      [-v[1]/(v[0]**2+v[1]**2), v[0]/(v[0]**2+v[1]**2)]])
        return J

    def J_o2s(self, v):
        #Jo2s=[cos(theta)   -row*sin(theta)
     #         sin(theta)    row*cos(theta)]
     
        # TODO: To be implemented by the student
        #
        J = np.block([[np.cos(v[1]), -v[0]*np.sin(v[1])],
                      [np.sin(v[1]),  v[0]*np.cos(v[1])]])
        return J




class PolarMapFeature(MapFeature):
    """
    This class inherits from the :class:`MapFeature` and implements a 2D Cartesian feature model for the MBL problem. The Cartesian coordinates are used for both,
    observing the feature and for its storage within the map. This class overrides the :meth:`GetFeatures` method to read
    the 2D Cartesian Features from the robot.
      
        """
    def GetFeatures(self):
    

        
        # TODO: To be implemented by the student
        zf, Rf = self.robot.ReadPolar2DFeature()
               
        # Initialize Hf and Vf as empty lists
        Hf, Vf = [], []

        # Flag to indicate if features are observed
        self.featureData = len(zf) != 0
       
       
        return zf, Rf, Hf, Vf
    
class PolarFeature(Feature,np.ndarray):
    """
    Cartesian feature class. The class inherits from the :class:`Feature` class providing an implementation of its
    interface for a Cartesian Feature, by implementing the :math:`\\boxplus` operator as well as its Jacobians. The
    class also inherits from the ndarray numpy class allowing to be operated as a numpy ndarray.
    """

    def __new__(BxF, input_array):
        """
        Constructor of the class. It is called when the class is instantiated. It is required to extend the ndarry numpy class.

        :param input_array: array used to initialize the class
        :returns: the instance of a :class:`CartesianFeature class object
        """
        assert input_array.shape == (3,1) or input_array.shape == (2,1), "CartesianFeature must be of 2 or 3 DOF"

        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(BxF)

        # The F matrix is used to convert from a pose to a feature in order to take profit of the oplus operator already implemented in the Pose class
        # The F matrix is (nf x np) where np is de dimension of the pose and nf the dimension of the feature
        # F is build as a list of F matrices, where the index of the list matches the dimension of the feature
        BxF.feature = obj

        BxF.F = np.block([np.diag(np.ones(len(BxF.feature))), np.zeros((len(BxF.feature),1))])

        super().__init__(BxF,obj)

        # Finally, we must return the newly created object:
        return obj

    def boxplus(BxF, NxB):
        """
        Pose-Cartesian Feature compounding operation:

        .. math::
            F&=\\begin{bmatrix} 1 & 0 & 0 & 0 \\\\ 0 & 1 & 0 & 0 \\end{bmatrix}\\\\
            ^Nx_F&=^Nx_B \\boxplus ^Bx_F = F ( ^Nx_B \\oplus ^Bx_F )
            :label: eq-boxplus2DCartesian

        which computes the Cartesian position of a feature in the N-Frame given the pose of the robot in the N-Frame and
        the Cartesian position of the feature in the B-Frame.

        :param NxB: Robot pose in the N-Frame (:math:`^Nx_B`)
        :param BxF: Cartesian feature pose in the B-Frame (:math:`^Bx_F`)
        :return: Feature pose in the N-Frame (:math:`^Nx_F`)
        """

        # TODO: To be completed by the student
        # Convert to cartesian coordinate
        BxF = BxF.ToCartesian()
        NxB = NxB.ToCartesian()

        NxF = BxF.F @ Pose3D.oplus(NxB, (BxF.F).T @ BxF)

        # Convert to polar coordinate
        NxF = PolarFeature.ToPolar(NxF)
        return NxF
    
    def J_1boxplus(BxF, NxB):
        """
        Jacobian of the Pose-Cartesian Feature compounding operation with respect to the robot pose:

        .. math::
            J_{1\\boxplus} = F J_{1\\oplus}
            :label: eq-J1boxplus2DCartesian

        :param NxB: robot pose represented in the N-Frame (:math:`^Nx_B`)
        :param BxF: Cartesian feature pose represented in the B-Frame (:math:`^Bx_F`)
        :return: Jacobian matrix :math:`J_{1\\boxplus}` (eq. :eq:`eq-J1boxplus2DCartesian`) (eq. :eq:`eq-J1boxplus2DCartesian`)
        """

        # TODO: To be completed by the student
        J = BxF.F @ Pose3D.J_1oplus(NxB, (BxF.F).T @ BxF)
        return J

    def J_2boxplus(BxF, NxB):
        """
        Jacobian of the Pose-Cartesian Feature compounding operation with respect to the feature position:

        .. math::
            J_{2\\boxplus} = F J_{2oplus}
            :label: eq-J2boxplus2DCartesian

        :param NxB: robot pose represented in the N-Frame (:math:`^Nx_B`)
        :param BxF: Cartesian feature pose represented in the B-Frame (:math:`^Bx_F`)
        :return: Jacobian matrix :math:`J_{1\\boxplus}` (eq. :eq:`eq-J2boxplus2DCartesian`)
        """

        # TODO: To be completed by the student
        J = BxF.F @ Pose3D.J_2oplus(NxB) @ (BxF.F).T
        return J
    
    def ToCartesian(self):
        """
        Translates from its internal representation to the representation used for plotting.

        :return: Feature in Cartesian Coordinates
        """
        cartesianFeature = CartesianFeature(np.block([self[0]*cos(self[1]), self[0]*sin(self[1]), self[2]]).reshape(3,1))
        return cartesianFeature
    
    def ToPolar(self):
        """
        Translates from its internal representation to the representation used for plotting.

        :return: Feature in Cartesian Coordinates
        """
        polarFeature = PolarFeature(np.block([np.sqrt(self[0]**2+self[1]**2), np.arctan2(self[1], self[0]), self[2]]).reshape(3,1))
        return polarFeature

    
      


     
  
    

 




if __name__ == '__main__':
    M = [CartesianFeature(np.array([[-40, 5]]).T),
           CartesianFeature(np.array([[-5, 40]]).T),
           CartesianFeature(np.array([[-5, 25]]).T),
           CartesianFeature(np.array([[-3, 50]]).T),
           CartesianFeature(np.array([[-20, 3]]).T),
           CartesianFeature(np.array([[40,-40]]).T)]  # feature map. Position of 2 point features in the world frame.

    xs0 = np.zeros((6, 1))
    
    a = Cartesian2DMapFeature().Jgx(xs0, M[0])

    print("a=", a)


    exit(0)