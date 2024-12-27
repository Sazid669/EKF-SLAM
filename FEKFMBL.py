import scipy
from GFLocalization import *
from MapFeature import *
from EKF import *
from blockarray import *

class FEKFMBL(GFLocalization, MapFeature):
    """
    Feature Extended Kalman Filter Map based Localization class. Inherits from :class:`GFLocalization.GFLocalization` and :class:`MapFeature.MapFeature`.
    The first one provides the basic functionality of a localization algorithm, while the second one provides the basic functionality required to use features.
    :class:`FEKFMBL.FEKFMBL` extends those classes by adding functionality to use a map based on features.
    """
    xB = -2  # constant used to index the state vector and the covariance matrix, to select the robot state
    x_eta = -1  # constant used to index the state vector and the covariance matrix, to select the robot pose
    def __init__(self, M, alpha,  *args):
        """
        Constructor of the FEKFMBL class.

        :param xBpose_dim: dimensionality of the robot pose within the state vector
        :param xB_dim: dimensionality of the state vector
        :param xF_dim: dimentsionality of a feature
        :param zfi_dim: dimensionality of a single feature observation
        :param M: Feature Based Map :math:`M =[^Nx_{F_1}^T~...~^Nx_{F_{n_f}}^T]^T`
        :param alpha: Chi2 tail probability. Confidence interaval of the individual compatibility test
        :param args: arguments to be passed to the EKFLocalization constructor
        """

        super().__init__(*args)  # initialize EKFLocalization
        self.xBpose_dim = self.Pose().shape[0] # Robot Pose dimensionality
        self.xB_dim = self.xk_1.shape[0]  # Robot State dimensionality (might include the velocity or other terms)
        self.xF_dim = self.Feature.feature.shape[0]  # Feature dimensionality
        print('xF_dim',self.xF_dim)
        self.zfi_dim = self.s2o(self.Feature.feature).shape[0]  # dimensionality of a single feature observation

        self.M = M  # Feature Based Map
        self.nf = len(M)  # number of features
        self.alpha = alpha  # Chi2 tail probability - Confidence interval 95%
        self.plt_zf_ellipse = []  # used for plotting the robot ellipse
        self.plt_zf_line = []  # used for plotting the line towards the robot ellipse

        self.plt_robotEllipse, = plt.plot([], [], 'b')
        self.plt_hf_ellipse = []
        self.plt_zf_ellipse = []
        self.plt_zf_line = []
        self.plt_samples = []
        

    def h(self, xk):  # overloaded to stack measurements and feature observations
        """
        We do differenciate two types of observations:

        * Measurements: :math:`z_m`correspond to observations of the state variable (position, velocity, etc...)
        * Feature Observations: :math:`z_f` correspond to observations of the features (CartesianFeature, PolarFeature, EsphericalFeature, etc...).

        This method implements the full observation model including the measurements and feature observations:

        .. math::
            z_k = h(x_k,v_k) \\Rightarrow \\begin{bmatrix} z_m \\\\ z_f \\end{bmatrix} = \\begin{bmatrix} h_m(x_k,v_m) \\\\ h_f(x_k,v_f) \\end{bmatrix} ~;~ v_k=[v_m^T ~v_f^T]^T
            :label: eq-mblh

        This method calls :meth:`h_m` to compute the expected measurements and  the :meth:`MapFeature.MapFeature.hf` method to compute the expected feature observations.
        The method returns an stacked vector of expected measurements and feature observations.

        :param xk: mean state vector used as linearization point
        :return: Joint stacked vector of the expected mesurement and feature observations
        """

        # TODO: To be completed by the student
                    # Initialize the vector for expected measurements.
        if self.measurement_flag == True:
            hm = self.hm(xk)
        else:
            hm = np.zeros((0,1))
        # Get features observation 
        if self.feature_flag == True:
            hf = self.hf(xk, self.H)
        else:
            hf = np.zeros((0,1))

        # Stack measurement and feature observation together zf + zm
        h_mf = np.block([[hm], [hf]])

        return h_mf

    def hm(self,xk):
        """
        Measurement observation model. This method computes the expected measurements :math:`h_m(x_k,v_m)` given the
        mean state vector :math:`x_k` and the measurement noise :math:`v_m`. It is implemented by calling to the ancestor
        class :meth:`EKF.EKF.h` method.

        :param xk: mean state vector.
        :return: expected measruments.
        """

        # TODO: To be completed by the student
        _hm = self.h_heading(xk)
        return _hm
    
    def SquaredMahalanobisDistance(self, hfj, Pfj, zfi, Rfi):
        """
        Computes the squared Mahalanobis distance between the expected feature observation :math:`hf_j` and the feature observation :math:`z_{f_i}`.

        :param hfj: expected feature observation
        :param Pfj: expected feature observation covariance
        :param zfi: feature observation
        :param Rfi: feature observation covariance
        :return: Squared Mahalanobis distance between the expected feature observation :math:`hf_j` and the feature observation :math:`z_{f_i}`
        """

        # TODO: To be completed by the student
        # Innovation: Difference between the actual and expected observations.
        # Represents the error in prediction.
        v_ij = zfi - hfj

        # S matrix: Sum of the covariances of the expected and actual observations.
        # Represents the combined uncertainty in the innovation.
        S_ij = Rfi + Pfj

        # Squared Mahalanobis Distance: A scaled distance between the actual and expected observations.
        # Takes into account the covariance (or uncertainty) of the observations.
        D2_ij = v_ij.T @ np.linalg.inv(S_ij) @ v_ij

        return D2_ij
    def IndividualCompatibility(self, D2_ij, dof, alpha):
        """
        Computes the individual compatibility test for the squared Mahalanobis distance :math:`D^2_{ij}`. The test is performed using the Chi-Square distribution with :math:`dof` degrees of freedom and a significance level :math:`\\alpha`.

        :param D2_ij: squared Mahalanobis distance
        :param dof: number of degrees of freedom
        :param alpha: confidence level
        :return: bolean value indicating if the Mahalanobis distance is smaller than the threshold defined by the confidence level
        """
        # TODO: To be completed by the student
        # Calculate the threshold from the Chi-Square distribution for the given confidence level and degrees of freedom.
        chi_square_threshold = scipy.stats.chi2.ppf(alpha, dof)

        # Check if the squared Mahalanobis distance is less than the calculated threshold.
        # If it is, the observation is considered compatible (within the confidence interval).
        isCompatible = D2_ij < chi_square_threshold

        return isCompatible

    def ICNN(self, hf, Phf, zf, Rf):
        """
        Individual Compatibility Nearest Neighbor (ICNN) data association algorithm. Given a set of expected feature
        observations :math:`h_f` and a set of feature observations :math:`z_f`, the algorithm returns a pairing hypothesis
        :math:`H` that associates each feature observation :math:`z_{f_i}` with the expected feature observation
        :math:`h_{f_j}` that minimizes the Mahalanobis distance :math:`D^2_{ij}`.

        :param hf: vector of expected feature observations
        :param Phf: Covariance matrix of the expected feature observations
        :param zf: vector of feature observations
        :param Rf: Covariance matrix of the feature observations
        :param dim: feature dimensionality
        :return: The vector of asociation hypothesis H
        """

        # TODO: To be completed by the student
        H = []
        # Iterate through each feature observation in zf
        for j in range(len(zf)):
            nearest = 0  # Initialize the nearest expected feature index to 0
            D2_min = np.inf  # Initialize the minimum Mahalanobis distance to infinity

            # Iterate through each expected feature observation in hf
            for i in range(len(hf)):
            # Calculate the squared Mahalanobis distance between the current expected and observed features
                D2_ij = self.SquaredMahalanobisDistance(hf[i], Phf[i], zf[j], Rf[j])

                # Check if the current distance is compatible and less than the current minimum distance
                if self.IndividualCompatibility(D2_ij, self.xF_dim, self.alpha) and D2_ij < D2_min:
                    nearest = i + 1  # Update the nearest expected feature index
                    D2_min = D2_ij  # Update the minimum Mahalanobis distance

            # Append the index of the nearest expected feature to the hypothesis list
            H.append(nearest)
            print('kkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk',H)

        # Return the vector of association hypotheses
        return H


    def DataAssociation(self, xk, Pk, zf, Rf):
        """
        Data association algorithm. Given state vector (:math:`x_k` and :math:`P_k`) including the robot pose and a set of feature observations
        :math:`z_f` and its covariance matrices :math:`R_f`,  the algorithm  computes the expected feature
        observations :math:`h_f` and its covariance matrices :math:`P_f`. Then it calls an association algorithms like
        :meth:`ICNN` (JCBB, etc.) to build a pairing hypothesis associating the observed features :math:`z_f`
        with the expected features observations :math:`h_f`.

        The vector of association hypothesis :math:`H` is stored in the :attr:`H` attribute and its dimension is the
        number of observed features within :math:`z_f`. Given the :math:`j^{th}` feature observation :math:`z_{f_j}`, *self.H[j]=i*
        means that :math:`z_{f_j}` has been associated with the :math:`i^{th}` feature . If *self.H[j]=None* means that :math:`z_{f_j}`
        has not been associated either because it is a new observed feature or because it is an outlier.

        :param xk: mean state vector including the robot pose
        :param Pk: covariance matrix of the state vector
        :param zf: vector of feature observations
        :param Rf: Covariance matrix of the feature observations
        :return: The vector of asociation hypothesis H
        """

        # TODO: To be completed by the student
        print('xk',xk.shape)
        print('PKKKKKKKKKKKKKKKKKKKKKKKKKKK',Pk.shape)
        
        hf = [] 
        Pf = []
        for i in range(self.nf):
           
           


            h_f = self.hfj(xk, i)
         
            P_f = self.Jhfjx(xk, i) @ Pk @ self.Jhfjx(xk, i).T
            

            hf.append(h_f)
            Pf.append(P_f)
        # Call the association algorithm
        H = self.ICNN(hf, Pf, zf, Rf)
        self.H = H
        return H

    def Localize(self, xk_1, Pk_1):
        """
        Localization iteration. Reads the input of the motion model, performs the prediction step (:meth:`EKF.EKF.Prediction`), reads the measurements
        and the features, solves the data association calling :meth:`DataAssociation` and the performs the update step (:meth:`EKF.EKF.Update`) and logs the results.
        The method also plots the uncertainty ellipse (:meth:`PlotUncertainty`) of the robot pose, the feature observations and the expected feature observations.

        :param xk_1: previous state vector
        :param Pk_1: previous covariance matrix
        :return xk, Pk: updated state vector and covariance matrix
        """

        # TODO: To be completed by the student
        # Get input to prediction step
        uk, Qk          = self.GetInput()
        # Prediction step
        xk_bar, Pk_bar  = self.Prediction(uk, Qk, xk_1, Pk_1)

        # Get measurement from the environment
        zm, Rm, Hm, Vm  = self.GetMeasurements()
        zf, Rf, Hf, Vf  = self.GetFeatures()

        # Data Association Step
        Hp = self.DataAssociation(xk_bar, Pk_bar, zf, Rf)
        
        #  Combine sensor and feature measurements for the update step.
        [zk, Rk, Hk, Vk, znp, Rnp] = self.StackMeasurementsAndFeatures(xk_bar, zm, Rm, Hm, Vm, zf, Rf, Hp)
        # Update the state estimate and covariance matrix with the combined measurements.
        # 
        xk, Pk          = self.Update(zk, Rk, xk_bar, Pk_bar, Hk, Vk)
      
         # Return the updated state vector, covariance matrix, and other relevant variables.
        return xk, Pk, xk_bar, zk, Rk
    

    def StackMeasurementsAndFeatures(self, xk, zm, Rm, Hm, Vm, zf, Rf, H):
        """
        Given the vector of  measurements observations :math:`z_m` together with their covariance matrix :math:`R_m`,
        the vector of feature observations :math:`z_f` together with their covariance matrix :math:`R_f`, The measurement observation matrix :math:`H_m`, the
        measurement observation noise matrix :math:`V_m` and the vector of feature associations :math:`H`, this method
        returns the joint observation vector :math:`z_k`, its related covariance matrix :math:`R_k`, the stacked
        Observation matrix :math:`H_k`, the stacked noise observation matrix :math:`V_k`, the vector of non-paired features
        :math:`z_{np}` and its xB_dim = self.xB_dim
        xBpose_dim = self.xBpose_dim     
        xF_dim = self.xF_dim noise covariance matrix :math:`R_{np}`.
        It is assumed that the measurements and the features observations are independent, therefore the covariance matrix
        of the joint observation vector is a block diagonal matrix.

        :param zm: measurement observations vector
        :param Rm: covariance matrix of the measurement observations
        :param Hm: measurement observation matrix
        :param Vm: measurement observation noise matrix
        :param zf: feature observations vector
        :param Rf: covariance matrix of the feature observations
        :param Hp: features associations vector
        :return: vector of joint measurement and feature observations :math:`z_k` and its covariance matrix :math:`R_k`
        """

        # TODO: To be completed by the student
        # Split and reorganize features based on the association information.
        zp, Rp, Hp, Vp, znp, Rnp , zfa ,Rfa = self.SplitFeatures(xk, zf, Rf, H)

        row_xk, col_xk  = xk.shape
        row_hm, col_hm = Hm.shape

        Hm  = np.block([[Hm , np.zeros( (row_hm , row_xk - col_hm ))]])
 
        if len(zm) == 0:
            zk, Rk, Hk, Vk = zp, Rp, Hp, Vp
        elif len(zp) == 0:
            zk, Rk, Hk, Vk = zm, Rm, Hm, Vm
        else:
            
            
            zk = np.block([[zm], [zp]])
         
        
            Rk = scipy.linalg.block_diag(Rm, Rp)

            Hk = np.block([[Hm], [Hp]])

            Vk = scipy.linalg.block_diag(Vm, Vp)

        return zk, Rk, Hk, Vk, znp, Rnp , zfa , Rfa

    def SplitFeatures(self, xk, zf, Rf, H):
        """
        Given the vector of feature observations :math:`z_f` and their covariance matrix :math:`R_f`, and the vector of
        feature associations :math:`H`, this function returns the vector of paired feature observations :math:`z_p` together with
        its covariance matrix :math:`R_p`, and the vector of non-paired feature observations :math:`z_{np}` together with its covariance matrix :math:`R_{np}`.
        The paired observations will be used to update the filter, while the non-paired ones will be considered as outliers.
        In the case of SLAM, they become new feature candidates.

        :param zf: vector of feature observations
        :param Rf: covariance matrix of feature observations
        :param H: hypothesis of feature associations
        :return: vector of paired feature observations :math:`z_p`, covariance matrix of paired feature observations :math:`R_p`, vector of non-paired feature observations :math:`z_{np}`, covariance matrix of non-paired feature observations :math:`R_{np}`.
        """
        # TODO: To be completed by the student
                # TODO: To be completed by the student
   
        zp = np.zeros((0, 1))  # empty vector
        Rp = np.zeros((0, 0))  # empty matrix
        Rfa = np.zeros((0, 0)) # empty matrix
        Hp, Vp = np.zeros((0,np.shape(xk)[0])), np.zeros((0,0))
        znp = np.zeros((0,1))  # empty vector)
        zfa = np.zeros((0,1))
        Rnp = np.zeros((0,0))  # empty matrix
       
        for i in range(0,len(H)):
            zfa = np.block([[zfa], [zf[i]]])
            Rfa= scipy.linalg.block_diag(Rfa, Rf[i])
           
            j = H[i]
            if j != 0:
                # Add feature observation
               
                zp = np.block([[zp], [zf[i]]])
                
                # Add feture uncertanity
                Rp = scipy.linalg.block_diag(Rp, Rf[i])
                
                Hp = np.block([[Hp], [self.Jhfjx(xk, j-1)]])

                Vp = scipy.linalg.block_diag(Vp, np.diag(np.ones(self.xF_dim)))

            else:
                znp = np.block([[znp], [zf[i]]])
                Rnp = scipy.linalg.block_diag(Rnp, Rf[i])
        
       
        return zp, Rp, Hp, Vp, znp, Rnp,zfa ,Rfa
        
    def PlotFeatureObservationUncertainty(self, zf, Rf, color):  # plots the feature observation uncertainty ellipse
        """
        Plots the uncertainty ellipse of the feature observations. This method is called by :meth:`FEKFMBL.PlotUncertainty`.

        :param zf: vector of feature observations
        :param Rf: covariance matrix of the feature observations
        """

        if len(zf) % 2 == 1:
            zf = zf[1:]
        else:
            zf = zf

        zf=BlockArray(zf,self.zfi_dim)
        Rf=BlockArray(Rf,self.zfi_dim)

        if zf is not None:
            # Remove previous feature observation ellipses
            for i in range(len(self.plt_zf_ellipse)):
                self.plt_zf_ellipse[i].remove()
                self.plt_zf_line[i].remove()
            self.plt_zf_ellipse = []
            self.plt_zf_line = []

        NxB = self.GetRobotPose(self.robot.xsk)

        # For all feature observations
        nzf = 0 if zf is None else zf.size // self.zfi_dim
        for i in range(0, nzf):
            BxF = self.Feature(zf[[i]])  # feature observation in the B-Frame
            BRF = Rf[[i,i]]  # feature observation covariance in the B-Frame
            NxF = self.Feature(self.g(NxB, BxF))
            J = self.Jgv(NxB, BxF)
            NRf = J @ BRF @ J.T
            NxF_Plot = NxF.ToCartesian()
            NRF_Plot = NxF.J_2c() @ NRf @ NxF.J_2c().T
            feature_ellipse = GetEllipse(NxF_Plot, NRF_Plot)
            plt_ellipse, = plt.plot(feature_ellipse[0], feature_ellipse[1], color)
            plt_line, = plt.plot([self.robot.xsk[0], NxF_Plot[0]],
                                 [self.robot.xsk[1], NxF_Plot[1]], color+'-.')
            self.plt_zf_ellipse.append(plt_ellipse)
            self.plt_zf_line.append(plt_line)
            #for j in range(len(self.M)): print("M[",j,"]=", self.M[j].ToCartesian().T)

    def PlotExpectedFeaturesObservationsUncertainty(self):
        """
        For all features in the map, this method plots the uncertainty ellipse of the expected feature observations. This method is called by :meth:`FEKFMBL.PlotUncertainty`.
        """
        for i in range(len(self.plt_hf_ellipse)):
            self.plt_hf_ellipse[i].remove()
        self.plt_hf_ellipse = []

        # Plot Expected Feature Observation Ellipses
        for Fj in range(self.nf):   # for all map features
            h_Fj = self.Feature(self.hfj(self.xk, Fj)) # expected feature observation in the B-Frame in the observation space
            J = self.Jhfjx(self.xk, Fj)
            P_h_Fj = J @ self.Pk @ J.T # expected feature observation covariance in the B-Frame in the observation space

            Nhx_Fj = self.Feature(self.g(self.xk, h_Fj)) # expected feature observation in the N-Frame in storage space
            #Jx = self.Jgx(self.xk, h_Fj)
            Jv = self.Jgv(self.xk, h_Fj)

            #NP_Fj = Jx @ self.Pk @ Jx.T + Jv @ P_h_Fj @ Jv.T # expected feature observation covariance in the N-Frame in storage representation
            NP_Fj = Jv @ P_h_Fj @ Jv.T  # expected feature observation covariance in the N-Frame in storage representation

            ellipse = GetEllipse(Nhx_Fj.ToCartesian(), Nhx_Fj.J_2c() @ NP_Fj @ Nhx_Fj.J_2c().T)
            plt_ellipse, = plt.plot(ellipse[0], ellipse[1], 'black')  # plot it
            self.plt_hf_ellipse.append(plt_ellipse)  # and add it to the list

            #self.PlotSampleObservationSpace(self.xk, h_Fj, P_h_Fj, 100)

    def PlotRobotUncertainty(self):  # plots the robot trajectory and its uncertainty ellipse
        """
        Plots the robot trajectory and its uncertainty ellipse. This method is called by :meth:`FEKFMBL.PlotUncertainty`.

        """
        # Plot Robot Ellipse
        robot_ellipse = GetEllipse(self.robot.xsk, self.GetRobotPoseCovariance(self.Pk))
        self.plt_robotEllipse.set_data(robot_ellipse[0], robot_ellipse[1])  # update it

        # Plot Robot Trajectory
        self.xTraj.append(self.xk[0, 0])
        self.yTraj.append(self.xk[1, 0])
        self.trajectory.pop(0).remove()
        self.trajectory = plt.plot(self.xTraj, self.yTraj, marker='.', color='blue', markersize=1)

    def PlotUncertainty(self,zf,Rf):
        """
        Plots the uncertainty ellipses of the robot pose (:meth:`PlotRobotUncertainty`), the feature observations
        (:meth:`PlotFeatureObservationUncertainty`) and the expected feature observations (:meth:`PlotExpectedFeaturesObservationsUncertainty`).
        This method is called by :meth:`FEKFMBL.Localize` at the end of a localization iteration in order to update
        the online  visualization.

        :param zf: vector of feature observations
        :param Rf: covariance matrix of the feature observations
        """
        if self.k % self.robot.visualizationInterval == 0:
            self.PlotRobotUncertainty()
            self.PlotFeatureObservationUncertainty(zf, Rf,'b')
            self.PlotExpectedFeaturesObservationsUncertainty()

    def GetRobotPose(self, xk):
        """
        Gets the robot pose from the state vector.

        :param xk: mean of the state vector:math:`x_k`
        :return: The robot pose :math:`x_{B_k}`
        """
        return self.Pose(xk[0:self.xBpose_dim])

    def GetRobotPoseCovariance(self, Pk):
        """
        Returns the robot pose covariance from the state covariance matrix.

        :param Pk: state vector covariance matrix :math:`P_k`
        :return: robot pose covariance :math:`P_{B_k}`
        """
        return Pk[0:self.xBpose_dim, 0:self.xBpose_dim]

