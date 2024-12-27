from SimulatedRobot import *
from IndexStruct import *
from Pose import *
from Feature import *
from MapFeature import *
import scipy
from roboticstoolbox.mobile.Animations import *
import numpy as np


class DifferentialDriveSimulatedRobot(SimulatedRobot):
    """
    This class implements a simulated differential drive robot. It inherits from the :class:`SimulatedRobot` class and
    overrides some of its methods to define the differential drive robot motion model.
    """
    def __init__(self, xs0, map=[],*args):
        """
        :param xs0: initial simulated robot state :math:`\\mathbf{x_{s_0}}=[^Nx{_{s_0}}~^Ny{_{s_0}}~^N\psi{_{s_0}}~]^T` used to initialize the  motion model
        :param map: feature map of the environment :math:`M=[^Nx_{F_1},...,^Nx_{F_{nf}}]`

        Initializes the simulated differential drive robot. Overrides some of the object attributes of the parent class :class:`SimulatedRobot` to define the differential drive robot motion model:

        * **Qsk** : Object attribute containing Covariance of the simulation motion model noise.

        .. math::
            Q_k=\\begin{bmatrix}\\sigma_{\\dot u}^2 & 0 & 0\\\\
            0 & \\sigma_{\\dot v}^2 & 0 \\\\
            0 & 0 & \\sigma_{\\dot r}^2 \\\\
            \\end{bmatrix}
            :label: eq:Qsk

        * **usk** : Object attribute containing the simulated input to the motion model containing the forward velocity :math:`u_k` and the angular velocity :math:`r_k`

        .. math::
            \\bf{u_k}=\\begin{bmatrix}u_k & r_k\\end{bmatrix}^T
            :label: eq:usk

        * **xsk** : Object attribute containing the current simulated robot state

        .. math::
            x_k=\\begin{bmatrix}{^N}x_k & {^N}y_k & {^N}\\theta_k & {^B}u_k & {^B}v_k & {^B}r_k\\end{bmatrix}^T
            :label: eq:xsk

        where :math:`{^N}x_k`, :math:`{^N}y_k` and :math:`{^N}\\theta_k` are the robot position and orientation in the world N-Frame, and :math:`{^B}u_k`, :math:`{^B}v_k` and :math:`{^B}r_k` are the robot linear and angular velocities in the robot B-Frame.

        * **zsk** : Object attribute containing :math:`z_{s_k}=[n_L~n_R]^T` observation vector containing number of pulses read from the left and right wheel encoders.
        * **Rsk** : Object attribute containing :math:`R_{s_k}=diag(\\sigma_L^2,\\sigma_R^2)` covariance matrix of the noise of the read pulses`.
        * **wheelBase** : Object attribute containing the distance between the wheels of the robot (:math:`w=0.5` m)
        * **wheelRadius** : Object attribute containing the radius of the wheels of the robot (:math:`R=0.1` m)
        * **pulses_x_wheelTurn** : Object attribute containing the number of pulses per wheel turn (:math:`pulseXwheelTurn=1024` pulses)
        * **Polar2D_max_range** : Object attribute containing the maximum Polar2D range (:math:`Polar2D_max_range=50` m) at which the robot can detect features.
        * **Polar2D\_feature\_reading\_frequency** : Object attribute containing the frequency of Polar2D feature readings (50 tics -sample times-)
        * **Rfp** : Object attribute containing the covariance of the simulated Polar2D feature noise (:math:`R_{fp}=diag(\\sigma_{\\rho}^2,\\sigma_{\\phi}^2)`)

        Check the parent class :class:`prpy.SimulatedRobot` to know the rest of the object attributes.
        """
        super().__init__(xs0, map,*args) # call the parent class constructor

        # Initialize the motion model noise
        self.Qsk = np.diag(np.array([0.1 ** 2, 0.01 ** 2, np.deg2rad(1) ** 2]))  # simulated acceleration noise
        self.usk = np.zeros((3, 1))  # simulated input to the motion model

        # Inititalize the robot parameters
        self.wheelBase = 0.5  # distance between the wheels
        self.wheelRadius = 0.1  # radius of the wheels
        self.pulse_x_wheelTurns = 10000  # number of pulses per wheel turn

        # Initialize the sensor simulation
        self.encoder_reading_frequency = 1  # frequency of encoder readings
        self.Re= np.diag(np.array([22 ** 2, 22 ** 2]))  # covariance of simulated wheel encoder noise

        self.Cartesian2D_feature_reading_frequency = 50 # frequency of Polar2D feature readings
        self.Cartesian2D_max_range = 50  # maximum Polar2D range, used to simulate the field of view
        self.Rfc = np.diag(np.array([0.01 ** 2, 0.1 ** 2]))  # covariance of simulated cartesian2D feature noise
        self.Rfp = np.diag(np.array([0.01 ** 2, np.deg2rad(0.01) ** 2]))  # covariance of simulated Polar2D feature noise

        self.Polar2D_feature_reading_frequency = 50  # frequency of Polar2D feature readings
        self.Polar2D_max_range = 50  # maximum Polar2D range, used to simulate the field of view
        self.Rfp = np.diag(np.array([1 ** 2, np.deg2rad(5) ** 2]))  # covariance of simulated Polar2D feature noise
        
        self.xy_feature_reading_frequency = 50  # frequency of XY feature readings
        self.xy_max_range = 50  # maximum XY range, used to simulate the field of view

        self.yaw_reading_frequency = 5# frequency of Yasw readings
        self.v_yaw_std = np.deg2rad(5)  # std deviation of simulated heading noise
        self.K = np.diag(np.array([1., 1., 1.]))
    def fs(self, xsk_1, usk):


        

        eta_sk_1=xsk_1[0:3]
        nu_sk_1=xsk_1[3:6]

        
        u_d=usk[0,0]
        r_d=usk[1,0]
        nu_d=np.array([[u_d],[0],[r_d]])
         # model noise

        wsk= np.array([[np.random.normal(0., 0.1)],
                                [0],
                                [np.random.normal(0., 0.005)]])

        
    
        eta_sk_update = Pose3D.oplus(eta_sk_1, (nu_sk_1 * self.dt + 0.5*wsk *(self.dt ** 2)))
        nu_sk_update = nu_sk_1 + np.matmul(self.K,(nu_d - nu_sk_1)) + wsk * self.dt
        
        
        self.xsk = np.concatenate((eta_sk_update, nu_sk_update))
      


        if self.k % self.visualizationInterval == 0:

            self.PlotRobot()
            
            self.xTraj.append(self.xsk[0, 0])
            self.yTraj.append(self.xsk[1, 0])
            self.trajectory.pop(0).remove()
            self.trajectory = plt.plot(self.xTraj, self.yTraj, marker='.', color='orange', markersize=1)
            


        self.k += 1
        return self.xsk



    def ReadEncoders(self):
        
        # linear velocity
        u=self.xsk[3,0] 
        # angular velocity
        r=self.xsk[5,0] 
        # spin velocities of the left and right wheels
        wl=u-(r*(self.wheelBase)/2)
        wr=u+(r*(self.wheelBase)/2)
        # number of pulses generated by left and right wheels
     
        del_n_L    = int((wl * self.dt) / (2*np.pi*self.wheelRadius/self.pulse_x_wheelTurns))
        del_n_R  = int((wr* self.dt) / (2*np.pi*self.wheelRadius/self.pulse_x_wheelTurns))
        # observation vector
        Zsk=np.array([[del_n_L], [del_n_R]])
      
        # Add noise
        noise = np.random.multivariate_normal([0.0,0.0],np.sqrt(self.Re)).reshape(2,1)
        Zsk_noisy=Zsk
        # covariance matrix of the read p[ulses
        Rsk=self.Re
        
        return Zsk_noisy, Rsk



#read2Dcartesianfeature
# xsk-robotposition
# mapfeature
# bxf= ominus etask box m[i] +bvk 


    def ReadCompass(self):
        #orientation of robot
        yaw=np.array([self.xsk[2,0]]).reshape(1,1)
        # the covariance of its noise R_yaw
        R_yaw=np.diag([self.v_yaw_std**2])
        return yaw,R_yaw


    def ReadCartesian2DFeature(self):
       
        zsk = []
        Rsk = []

         
     # Compute the 2D Cartesian distance from the robot to each feature
        zsk = [MapFeature().o2s(CartesianFeature.boxplus(self.M[i],(Pose3D.ominus(self.xsk[:3, 0].reshape((3, 1))))))
           for i in range(self.nf)]

     # Replicate the measurement noise for each feature
        Rsk = [self.Rfc for _ in range(self.nf)]

        

        if self.k % self.Cartesian2D_feature_reading_frequency == 0:
            return zsk, Rsk
        else:
            return [], []
        
    def ReadPolar2DFeature(self):
        """ Simulates the feature reading of the robot.

        :return: 2D Polar distance from the robot to features in the robot frame and the covariance of its noise *R_feature*
        """

        # TODO: to be completed by the student
        # Initialise the measurement zsk with size is DoF of the feature by 1
        zsk = []
        Rsk = []

        # Number of feature loop
        
        noise = np.random.normal(0, np.sqrt(self.Rfp.diagonal())).reshape((len(self.Rfp),1))
        zsk=[Cartesian2DStoredPolarObservedMapFeature().s2o(CartesianFeature.boxplus(self.M[i],(Pose3D.ominus(self.xsk[0:3,0].reshape((3,1))))))+noise
            for i in range(self.nf)]
         # Replicate the measurement noise for each feature
        Rsk = [self.Rfp for _ in range(self.nf)]

        if self.Polar2D_feature_reading_frequency != 0 and self.k % self.Polar2D_feature_reading_frequency == 0:
            return zsk, Rsk
        else:
            return [], []
        

    def PlotRobot(self):
        """ Updates the plot of the robot at the current pose """

        self.vehicleIcon.update([self.xsk[0], self.xsk[1], self.xsk[2]])
        plt.pause(0.0000001)
        return


