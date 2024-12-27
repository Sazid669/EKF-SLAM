from MapFeature import *
from FEKFMBL import *
from scipy.stats import pearsonr
from FEKFMBL import *
from EKF_3DOFDifferentialDriveInputDisplacement import *



class FEKFSLAM(FEKFMBL):
    """
    Class implementing the Feature-based Extended Kalman Filter for Simultaneous Localization and Mapping (FEKFSLAM).
    It inherits from the FEKFMBL class, which implements the Feature Map based Localization (MBL) algorithm using an EKF.
    :class:`FEKFSLAM` extends :class:`FEKFMBL` by adding the capability to map features previously unknown to the robot.
    """
 
    def __init__(self,  *args):

        super().__init__(*args)

        # self.xk_1 # state vector mean at time step k-1 inherited from FEKFMBL
        
        self.nzm = 0  # number of measurements observed
        self.nzf = 0  # number of features observed

        self.H = None  # Data Association Hypothesis
        self.nf = 0  # number of features in the state vector

        self.plt_MappedFeaturesEllipses = []

        return

 
    def AddNewFeatures(self, xk, Pk, znp, Rnp):
        
        """
        This method adds new features to the map. Given:

        * The SLAM state vector mean and covariance:

        .. math::
            {{}^Nx_{k}} & \\approx {\\mathcal{N}({}^N\\hat x_{k},{}^NP_{k})}\\\\
            {{}^N\\hat x_{k}} &=   \\left[ {{}^N\\hat x_{B_k}^T} ~ {{}^N\\hat x_{F_1}^T} ~  \\cdots ~ {{}^N\\hat x_{F_{nf}}^T} \\right]^T \\\\
            {{}^NP_{k}}&=
            \\begin{bmatrix}
            {{}^NP_{B}} & {{}^NP_{BF_1}} & \\cdots & {{}^NP_{BF_{nf}}}  \\\\
            {{}^NP_{F_1B}} & {{}^NP_{F_1}} & \\cdots & {{}^NP_{F_1F_{nf}}}  \\\\
            \\vdots & \\vdots & \\ddots & \\vdots \\\\
            {{}^NP_{F_{nf}B}} & {{}^NP_{F_{nf}F_1}} & \\cdots & {{}^NP_{nf}}  \\\\
            \\end{bmatrix}
            :label: FEKFSLAM-state-vector-mean-and-covariance

        * And the vector of non-paired feature observations (feature which have not been associated with any feature in the map), and their covariance matrix:

            .. math::
                {z_{np}} &=   \\left[ {}^Bz_{F_1} ~  \\cdots ~ {}^Bz_{F_{n_{zf}}}  \\right]^T \\\\
                {R_{np}}&= \\begin{bmatrix}
                {}^BR_{F_1} &  \\cdots & 0  \\\\
                \\vdots &  \\ddots & \\vdots \\\\
                0 & \\cdots & {}^BR_{F_{n_{zf}}}
                \\end{bmatrix}
                :label: FEKFSLAM-non-paire-feature-observations
                

        this method creates a grown state vector ([xk_plus, Pk_plus]) by adding the new features to the state vector.
        Therefore, for each new feature :math:`{}^Bz_{F_i}`, included in the vector :math:`z_{np}`, and its corresponding feature observation noise :math:`{}^B R_{F_i}`, the state vector mean and covariance are updated as follows:

            .. math::
                {{}^Nx_{k}^+} & \\approx {\\mathcal{N}({}^N\\hat x_{k}^+,{}^NP_{k}^+)}\\\\
                {{}^N x_{k}^+} &=
                \\left[ {{}^N x_{B_k}^T} ~ {{}^N x_{F_1}^T} ~ \\cdots ~{{}^N x_{F_n}^T}~ |~\\left({{}^N x_{B_k} \\boxplus ({}^Bz_{F_i} }+v_k)\\right)^T \\right]^T \\\\
                {{}^N\\hat x_{k}^+} &=
                \\left[ {{}^N\\hat x_{B_k}^T} ~ {{}^N\\hat x_{F_1}^T} ~ \\cdots ~{{}^N\\hat x_{F_n}^T}~ |~{{}^N\\hat x_{B_k} \\boxplus {}^Bz_{F_i}^T } \\right]^T \\\\
                {P_{k}^+}&= \\begin{bmatrix}
                {{}^NP_{B_k}}  &  {{}^NP_{B_kF_1}}   &  \\cdots   &  {{}^NP_{B_kF_n}} & | & {{}^NP_{B_k} J_{1 \\boxplus}^T}\\\\
                {{}^NP_{F_1B_k}}  &  {{}^NP_{F_1}}   &  \\cdots   &  {{}^NP_{F_1F_n}} & | & {{}^NP_{F_1B_k} J_{1 \\boxplus}^T}\\\\
                \\vdots  & \\vdots & \\ddots  & \\vdots & | &  \\vdots \\\\
                {{}^NP_{F_nB_k}}  &  {{}^NP_{F_nF_1}}   &  \\cdots   &  {{}^NP_{F_n}}  & | & {{}^NP_{F_nB_k} J_{1 \\boxplus}^T}\\\\
                \\hline
                {J_{1 \\boxplus} {}^NP_{B_k}}  &  {J_{1 \\boxplus} {}^NP_{B_kF_1}}   &  \\cdots   &  {J_{1 \\boxplus} {}^NP_{B_kF_n}}  & | &  {J_{1 \\boxplus} {}^NP_R J_{1 \\boxplus} ^T} + {J_{2\\boxplus}} {{}^BR_{F_i}} {J_{2\\boxplus}^T}\\\\
                \\end{bmatrix}
                :label: FEKFSLAM-add-a-new-feature

        :param xk: state vector mean
        :param Pk: state vector covariance
        :param znp: vector of non-paired feature observations (they have not been associated with any feature in the map)
        :param Rnp: Matrix of non-paired feature observation covariances
        :return: [xk_plus, Pk_plus] state vector mean and covariance after adding the new features
        """

        assert znp.size > 0, "AddNewFeatures: znp is empty"
        
        """
This method adds new features to the map. Given:

* The SLAM state vector mean and covariance:

    State Vector Mean:
    xk ≈ N(x̂k, Pk)

    State Vector Components:
    x̂k = [x̂B_kᵀ x̂F₁ᵀ ... x̂F_nfᵀ]ᵀ

    Covariance Matrix:
    Pk =
    [  P_B    P_BF₁  ...  P_BF_nf
      P_F₁B  P_F₁   ...  P_F₁F_nf
      ...    ...    ...  ...
      P_F_nfB P_F_nf ...  P_nf   ]

* And the vector of non-paired feature observations (features which have not been associated with any feature in the map), and their covariance matrix:

    Non-paired Feature Observations:
    z_np = [z_BF₁ ... z_BF_nzf]ᵀ

    Covariance Matrix:
    R_np =
    [ R_F₁   ...  0
      ...    ...  ...
      0      ...  R_F_nzf ]

This method creates a grown state vector ([xk_plus, Pk_plus]) by adding the new features to the state vector.
Therefore, for each new feature z_BF_i included in the vector z_np, and its corresponding feature observation noise R_F_i, the state vector mean and covariance are updated as follows:

    Updated State Vector Mean:
    x_k^+ = [x_B_kᵀ x_F₁ᵀ ... x_F_nᵀ | (x_B_k ⊕ (z_BF_i + v_k))ᵀ ]ᵀ

    Updated State Vector Mean (for prediction):
    x̂_k^+ = [x̂_B_kᵀ x̂_F₁ᵀ ... x̂_F_nᵀ | (x̂_B_k ⊕ z_BF_iᵀ) ]ᵀ

    Updated Covariance Matrix:
    P_k^+ =
    [ P_B_k    P_B_kF₁  ...  P_B_kF_n | P_B_k J_1⊕ᵀ
      P_F₁B_k  P_F₁     ...  P_F₁F_n | P_F₁B_k J_1⊕ᵀ
      ...      ...       ...  ...     | ...
      P_F_nB_k P_F_nF₁   ...  P_F_n   | P_F_nB_k J_1⊕ᵀ
      ---------------------------------------------
      J_1⊕ P_B_k    J_1⊕ P_B_kF₁  ...  J_1⊕ P_B_kF_n | J_1⊕ P_R J_1⊕ᵀ + J_2⊕ R_F_i J_2⊕ᵀ ]

:param xk: State vector mean
:param Pk: State vector covariance
:param znp: Vector of non-paired feature observations (not associated with any feature in the map)
:param Rnp: Covariance matrix of non-paired feature observations
:return: [xk_plus, Pk_plus] State vector mean and covariance after adding the new features
"""

        
        ## To be completed by the student
        
     
     
        
        xF_dim    = self.xF_dim      # dimension of one feature  
        xB_dim  = self.xB_dim  # dimension of the robot pose 
        # sate_dim  = self.xB_dim
        n = len(xk) # length of the state vector including feature 
        nf= int( len(znp) / self.xF_dim) # number of un paired  features 
        for i in range(nf):
            start_idx = i * self.xF_dim
            end_idx=start_idx+xF_dim
            
            Rfpi = Rnp[ start_idx: end_idx , start_idx :  end_idx] # extract uncertainity of one observation 
            NxF = self.g(xk[0:xB_dim],znp[ start_idx : end_idx]) # coupute feature observation in the N Frame 
            xk = np.block([[xk],[NxF]]) # Add Unpaired Feature to the state vector 
            Jgx = self.Jgx(xk , znp[ start_idx : end_idx]) # jacobian with respect state 
            Jgv = self.Jgv(xk , znp[ start_idx : end_idx]) # jacobian with respect observation noise 

            End_block = Jgx@(Pk[0:xB_dim,0:xB_dim]) @ Jgx.T + Jgv@Rfpi@Jgv.T # end digonal matrix 

            
            First_block=Pk
            Side_block = Pk[0:xB_dim,0:xB_dim] @ Jgx.T
            # Update side block with covariance updates for existing features
            for j in range(xB_dim , n , xF_dim):
                
               
                Side = Pk[j:j+xF_dim, :xB_dim] @ Jgx.T
                Side_block = np.vstack([Side_block, Side])
            # Merge all blocks together
            Pk = np.block([[First_block,Side_block],[Side_block.T,End_block]])
            n += xF_dim #  # Update state vector length for each new feature

         # Update the number of features in the state
        self.nf = (len(xk) - xB_dim) // xF_dim
        print('nf',self.nf)

        return xk , Pk

        # return xk_plus, Pk_plus

    # def Prediction(self, uk, Qk, xk_1, Pk_1):
        """
        This method implements the prediction step of the FEKFSLAM algorithm. It predicts the state vector mean and
        covariance at the next time step. Given state vector mean and covariance at time step k-1:

        .. math::
            {}^Nx_{k-1} & \\approx {\\mathcal{N}({}^N\\hat x_{k-1},{}^NP_{k-1})}\\\\
            {{}^N\\hat x_{k-1}} &=   \\left[ {{}^N\\hat x_{B_{k-1}}^T} ~ {{}^N\\hat x_{F_1}^T} ~  \\cdots ~ {{}^N\\hat x_{F_{nf}}^T} \\right]^T \\\\
            {{}^NP_{k-1}}&=
            \\begin{bmatrix}
            {{}^NP_{B_{k-1}}} & {{}^NP_{BF_1}} & \\cdots & {{}^NP_{BF_{nf}}}  \\\\
            {{}^NP_{F_1B}} & {{}^NP_{F_1}} & \\cdots & {{}^NP_{F_1F_{nf}}}  \\\\
            \\vdots & \\vdots & \\ddots & \\vdots \\\\
            {{}^NP_{F_{nf}B}} & {{}^NP_{F_{nf}F_1}} & \\cdots & {{}^NP_{nf}}  \\\\
            \\end{bmatrix}
            :label: FEKFSLAM-state-vector-mean-and-covariance-k-1

        the control input and its covariance :math:`u_k` and :math:`Q_k`, the method computes the state vector mean and covariance at time step k:

        .. math::
            {{}^N\\hat{\\bar x}_{k}} &=   \\left[ {f} \\left( {{}^N\\hat{x}_{B_{k-1}}}, {u_{k}}  \\right)  ~  { {}^N\\hat x_{F_1}^T} \\cdots { {}^N\\hat x_{F_n}^T}\\right]^T\\\\
            {{}^N\\bar P_{k}}&= {F_{1_k}} {{}^NP_{k-1}} {F_{1_k}^T} + {F_{2_k}} {Q_{k}} {F_{2_k}^T}
            :label: FEKFSLAM-prediction-step

        where

        .. math::
            {F_{1_k}} &= \\left.\\frac{\\partial {f_S({}^Nx_{k-1},u_k,w_k)}}{\\partial {{}^Nx_{k-1}}}\\right|_{\\begin{subarray}{l} {{}^Nx_{k-1}}={{}^N\\hat x_{k-1}} \\\\ {w_k}={0}\\end{subarray}} \\\\
             &=
            \\begin{bmatrix}
            \\frac{\\partial {f} \\left( {{}^Nx_{B_{k-1}}}, {u_{k}}, {w_{k}}  \\right)}{\\partial {{}^Nx_{B_{k-1}}}} &
            \\frac{\\partial {f} \\left( {{}^Nx_{B_{k-1}}}, {u_{k}}, {w_{k}}  \\right)}{\\partial {{}^Nx_{F1}}} &
            \\cdots &
            \\frac{\\partial {f} \\left( {{}^Nx_{B_{k-1}}}, {u_{k}}, {w_{k}}  \\right)}{\\partial {{}^Nx_{Fn}}} \\\\
            \\frac{\\partial {{}^Nx_{F1}}}{\\partial {{}^Nx_{k-1}}} &
            \\frac{\\partial {{}^Nx_{F1}}}{\\partial {{}^Nx_{F1}}} &
            \\cdots &
            \\frac{\\partial {{}^Nx_{Fn}}}{\\partial {{}^Nx_{Fn}}} \\\\
            \\vdots & \\vdots & \\ddots & \\vdots \\\\
            \\frac{\\partial {{}^Nx_{Fn}}}{\\partial {{}^Nx_{k-1}}} &
            \\frac{\\partial {{}^Nx_{Fn}}}{\\partial {{}^Nx_{F1}}} &
            \\cdots &
            \\frac{\\partial {{}^Nx_{Fn}}}{\\partial {{}^Nx_{Fn}}}
            \\end{bmatrix}
            =
            \\begin{bmatrix}
            {J_{f_x}} & {0} & \\cdots & {0} \\\\
            {0}   & {I} & \\cdots & {0} \\\\
            \\vdots& \\vdots  & \\ddots & \\vdots  \\\\
            {0}   & {0} & \\cdots & {I} \\\\
            \\end{bmatrix}
            \\\\{F_{2_k}} &= \\left. \\frac{\\partial {f({}^Nx_{k-1},u_k,w_k)}}{\\partial {w_{k}}} \\right|_{\\begin{subarray}{l} {{}^Nx_{k-1}}={{}^N\\hat x_{k-1}} \\\\ {w_k}={0}\\end{subarray}}
            =
            \\begin{bmatrix}
            \\frac{\\partial {f} \\left( {{}^Nx_{B_{k-1}}}, {u_{k}}, {w_{k}}  \\right)}{\\partial {w_{k}}} \\\\
            \\frac{\\partial {{}^Nx_{F1}}}{\\partial {w_{k}}}\\\\
            \\vdots \\\\
            \\frac{\\partial {{}^Nx_{Fn}}}{\\partial {w_{k}}}
            \\end{bmatrix}
            =
            \\begin{bmatrix}
            {J_{f_w}}\\\\
            {0}\\\\
            \\vdots\\\\
            {0}\\\\
            \\end{bmatrix}
            :label: FEKFSLAM-prediction-step-Jacobian

        obtaining the following covariance matrix:
        
        .. math::
            {{}^N\\bar P_{k}}&= {F_{1_k}} {{}^NP_{k-1}} {F_{1_k}^T} + {F_{2_k}} {Q_{k}} {F_{2_k}^T}{{}^N\\bar P_{k}}
             &=
            \\begin{bmatrix}
            {J_{f_x}P_{B_{k-1}} J_{f_x}^T} + {J_{f_w}Q J_{f_w}^T}  & |  &  {J_{f_x}P_{B_kF_1}} & \\cdots & {J_{f_x}P_{B_kF_n}}\\\\
            \\hline
            {{}^NP_{F_1B_k} J_{f_x}^T} & |  &  {{}^NP_{F_1}} & \\cdots & {{}^NP_{F_1F_n}}\\\\
            \\vdots & | & \\vdots & \\ddots & \\vdots \\\\
            {{}^NP_{F_nB_k} J_{f_x}^T} & | &  {{}^NP_{F_nF_1}} & \\cdots & {{}^NP_{F_n}}
            \\end{bmatrix}
            :label: FEKFSLAM-prediction-step-covariance

        The method returns the predicted state vector mean (:math:`{}^N\\hat{\\bar x}_k`) and covariance (:math:`{{}^N\\bar P_{k}}`).

        :param uk: Control input
        :param Qk: Covariance of the Motion Model noise
        :param xk_1: State vector mean at time step k-1
        :param Pk_1: Covariance of the state vector at time step k-1
        :return: [xk_bar, Pk_bar] predicted state vector mean and covariance at time step k
        """    ## To be completed by the student
    
    def Prediction(self, uk, Qk, xk_1, Pk_1):
        print('prediction_xk_1',xk_1)
        self.uk = uk
        self.Qk = Qk
        self.xk_1 = xk_1
        self.Pk_1 = Pk_1
        

        #  n is the dimension of the robot's state
        n = self.xB_dim# for x, y, theta
    
       
      
        
        
        # # Predict the robot's state
        
       
        print('Pk_1_shape',Pk_1.shape)
        print('xk1shape',xk_1.shape)
      
        shape=len(Pk_1)
    
        F1k=np.eye(shape)
        
        F1k[:n,:n]=self.Jfx(xk_1, uk)
        print('F1K_SHAPE',F1k.shape)
        
      
       
        
        F2k=np.zeros((shape,3))
        F2k[0:n,0:n]=self.Jfw(xk_1)
      
        
        
        
        

        xk_robot= self.f(xk_1[:self.xB_dim], uk)
        xk_bar= np.vstack((xk_robot , xk_1[self.xB_dim:]))
        
        
        Pk_bar=F1k@Pk_1@F1k.T+F2k@Qk@F2k.T
   
        
        print('xk_bar:', xk_bar)
        print('xk_bar:', xk_bar.shape)
        print('Pk_bar:', Pk_bar)
        print('PK_shape',Pk_bar.shape)
        return xk_bar, Pk_bar


    def Localize(self, xk_1, Pk_1):
        """
        This method implements the FEKFSLAM algorithm. It localizes the robot and maps the features in the environment.
        It implements a single interation of the SLAM algorithm, given the current state vector mean and covariance.
        The unique difference it has with respect to its ancestor :meth:`FEKFMBL.Localize` is that it calls the method
        :meth:`AddNewFeatures` to add new non-paired features to the map.

        :param xk_1: state vector mean at time step k-1
        :param Pk_1: covariance of the state vector at time step k-1
        :return: [xk, Pk] state vector mean and covariance at time step k
        """
        ## To be completed by the student
        # TODO: To be completed by the student
        # Get input to prediction step
        uk, Qk          = self.GetInput()
        
        # Prediction step
        xk_bar, Pk_bar    = self.Prediction(uk, Qk, xk_1, Pk_1)

        # Get measurement
        zm, Rm, Hm, Vm  = self.GetMeasurements()
        # get Feature
        zf, Rf, Hf, Vf= self.GetFeatures()

        # Data Association
        
        Hp = self.DataAssociation(xk_bar, Pk_bar, zf, Rf)
        print("Hp:", Hp)
        # Stack Meaurement and Feature Together  
        [zk, Rk, Hk, Vk, znp, Rnp , zf , Rf] = self.StackMeasurementsAndFeatures(xk_bar, zm, Rm, Hm, Vm, zf, Rf, Hp)
        # Update step
        
        print(zk)
        
        print(Hk)
        
        if len(zk) > 1:
            print("yes")
     
        
        self.xk, self.Pk  = self.Update(zk, Rk, xk_bar, Pk_bar, Hk, Vk)
        # xk = xk_bar
        # Pk = Pk_bar
        # self.xk_bar = xk
        # self.xk = self.xk_bar
        # self.Pk = Pk
        if(len(znp) >= self.xF_dim):
            self.xk , self.Pk = self.AddNewFeatures(self.xk ,self.Pk,znp,Rnp)
        
      
        # Use the variable names zm, zf, Rf, znp, Rnp so that the plotting functions work
        self.Log(self.robot.xsk, self.GetRobotPose(self.xk), self.GetRobotPoseCovariance(self.Pk),
                 self.GetRobotPose(self.xk), zm)  # log the results for plotting

        self.PlotUncertainty(zf, Rf, znp, Rnp)
        return self.xk, self.Pk


    def PlotMappedFeaturesUncertainty(self):
        """
        This method plots the uncertainty of the mapped features. It plots the uncertainty ellipses of the mapped
        features in the environment. It is called at each Localization iteration.
        """
        # remove previous ellipses
        for i in range(len(self.plt_MappedFeaturesEllipses)):
            self.plt_MappedFeaturesEllipses[i].remove()
        self.plt_MappedFeaturesEllipses = []

        self.xk=BlockArray(self.xk,self.xF_dim, self.xB_dim)
        self.Pk=BlockArray(self.Pk,self.xF_dim, self.xB_dim)
        
        print("drawing")
        print(self.xk)

        # draw new ellipses
        for Fj in range(self.nf):
            print("Fj ", Fj)
            print(self.xk[[Fj]])
            feature_ellipse = GetEllipse(self.xk[[Fj]],
                                         self.Pk[[Fj,Fj]])  # get the ellipse of the feature (x_Fj,P_Fj)
            plt_ellipse, = plt.plot(feature_ellipse[0], feature_ellipse[1], 'r')  # plot it
            self.plt_MappedFeaturesEllipses.append(plt_ellipse)  # and add it to the list
        print("done")

    def PlotUncertainty(self, zf, Rf, znp, Rnp):
        """
        This method plots the uncertainty of the robot (blue), the mapped features (red), the expected feature observations (black) and the feature observations.

        :param zf: vector of feature observations
        :param Rf: covariance matrix of feature observations
        :param znp: vector of non-paired feature observations
        :param Rnp: covariance matrix of non-paired feature observations
        :return:
        """
        if self.k % self.robot.visualizationInterval == 0:
            self.PlotRobotUncertainty()
            self.PlotFeatureObservationUncertainty(znp, Rnp,'b')
            self.PlotFeatureObservationUncertainty(zf, Rf,'g')
            self.PlotExpectedFeaturesObservationsUncertainty()
            self.PlotMappedFeaturesUncertainty()