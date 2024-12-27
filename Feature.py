from conversions import *
from Pose import *
import numpy as np
from conversions import *

class Feature:
    """
    This class implements the **interface of the pose-feature compounding operation**. This class provides the interface
    to implement the compounding operation between the robot pose (represented in the N-Frame) and the feature pose (represented in
    the B-Frame) obtaining the feature representation in the N-Frame.
    The class also provides the interface to implement the Jacobians of the pose-feature compounding operation.
    """

    def __init__(BxF, feature):
        BxF.feature = feature

    def boxplus(BxF, NxB):
        """
        Pose-Feature compounding operation:

        .. math::
            ^Nx_F=^Nx_B \\boxplus ^Bx_F
            :label: eq-boxplus

        which computes the pose of a feature in the N-Frame given the pose of the robot in the N-Frame and the pose of
        the feature in the B-Frame.
        **This is a pure virtual method that must be overwritten by the child class**.

        :param NxB: Robot pose in the N-Frame (:math:`^Nx_B`)
        :param BxF: Feature pose in the B-Frame (:math:`^Bx_F`)
        :return: Feature pose in the N-Frame (:math:`^Nx_F`)
        """
        pass

    def J_1boxplus(BxF, NxB):
        """
        Jacobian of the Pose-Feature compounding operation (eq. :eq:`eq-boxplus`) with respect to the first argument :math:`^Nx_B`.

        .. math::
            J_{1\\boxplus}=\\frac{\\partial ^Nx_B \\boxplus ^Bx_F}{\\partial ^Nx_B}.
            :label: eq-J_1boxplus

        **To be overriden by the child class**.

        :param NxB: Robot pose in the N-Frame (:math:`^Nx_B`)
        :param BxF: Feature pose in the B-Frame (:math:`^Bx_F`)
        :return: Jacobian matrix :math:`J_{1\\boxplus}`
        """
        pass

    def J_2boxplus(BxF, NxB):
        """
        Jacobian of the Pose-Feature compounding operation (eq. :eq:`eq-boxplus`) with respect to the second argument :math:`^Bx_F`.

        .. math::
            J_{2\\boxplus}=\\frac{\\partial ^Nx_B \\boxplus ^Bx_F}{\\partial ^Bx_F}.
            :label: eq-J_2boxplus

        **To be overriden by the child class**.

        :param NxB: Robot pose in the N-Frame (:math:`^Nx_B`)
        :return: Jacobian matrix :math:`J_{2\\boxplus}`
        """
        pass

    def ToCartesian(self):
        """
        Translates from its internal representation to the representation used for plotting.
        **To be overriden by the child class**.

        :return: Feature in Cartesian Coordinates
        """
        pass

    def J_2c(selfself):
        """
        Jacobian of the ToCartesian method. Required for plotting non Cartesian features.
        **To be overriden by the child class**.

        :return: Jacobian of the transformation
        """
        pass

class CartesianFeature(Feature,np.ndarray):
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
        # print(obj)
        
        BxF.F = np.block([np.diag(np.ones(len(BxF.feature))), np.zeros((len(BxF.feature),1))])
        # BxF.F = np.array([ [1,0,0],[0,1,0]])
        

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
              
            # 2X3 @(3X1, 3X2 . 2X1) =  2X1
        NxF = BxF.F @ Pose3D.oplus(NxB, (BxF.F).T @ BxF) # gives 2X1 array 
       
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
        # print(BxF.F)
        J = BxF.F @ Pose3D.J_1oplus(NxB, (BxF.F).T @ BxF)  # 2X3 matrix
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
        J = BxF.F @ Pose3D.J_2oplus(NxB) @ (BxF.F).T # 2X2 matrix
        return J

    def ToCartesian(self):
        """
        Translates from its internal representation to the representation used for plotting.

        :return: Feature in Cartesian Coordinates
        """
        return self

    def J_2c(self):
        """
        Jacobian of the ToCartesian method. Required for plotting non Cartesian features.
        **To be overriden by the child class**.

        :return: Jacobian of the transformation
        """
        return np.eye(self.shape[0])

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
        # print(obj)
        
        BxF.F = np.block([np.diag(np.ones(len(BxF.feature))), np.zeros((len(BxF.feature),1))])
        # BxF.F = np.array([ [1,0,0],[0,1,0]])
        

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
        BxF = p2c(BxF) 
        
        NxF = BxF.F @ Pose3D.oplus(NxB, (BxF.F).T @ BxF) # 
        
        return c2p(NxF)

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
        # print(BxF.F)
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
        return p2c(self)

    def J_2c(self):
        """
        Jacobian of the ToCartesian method. Required for plotting non Cartesian features.
        **To be overriden by the child class**.

        :return: Jacobian of the transformation
        """
        return J_p2c(self)


if __name__ == '__main__':

    NxB3dof = Pose3D(np.array([[5,5,np.pi/2]]).T)
    BxF = CartesianFeature(np.array([[3,3]]).T)

    NxF = BxF.boxplus(NxB3dof)

    print("NxF=", NxF.T)
    print("J_1boxplus=", BxF.J_1boxplus(NxB3dof))
    print("J_2boxplus=", BxF.J_2boxplus(NxB3dof))

   

    exit(0)