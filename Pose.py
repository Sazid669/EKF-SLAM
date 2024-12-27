import Feature as f
import numpy as np
from math import cos, sin

class Pose(np.ndarray):
    """
    Definition of a robot pose interface from where all the particular poses of different DOF inherit.
    This class defines a robot pose AxB as the pose of the B-Frame expressed in the A-Frame coordinates.
    """

    def oplus(AxB, BxC):
        """
        Given a Pose object *AxB* (the self object) and a Pose object *BxC*, it returns the compounded Pose object *AxC*.

        The operation is defined as:

        .. math::
            \\mathbf{{^A}x_C} = \\mathbf{{^A}x_B} \\oplus \\mathbf{{^B}x_C}
            :label: eq-oplus

        **This is a pure virtual method that must be implemented by a child class.**

        :param BxC: C-Frame pose expressed in B-Frame coordinates
        :returns: C-Frame pose expressed in A-Frame coordinates
        """
        pass

    def J_1oplus(AxB, BxC):
        """
        Jacobian of the pose compounding operation (eq. :eq:`eq-oplus`) with respect to the first pose:

        .. math::
            J_{1\\oplus}=\\frac{\\partial  ^Ax_B \\oplus ^Bx_C}{\\partial ^Ax_B}
            :label: eq-J1oplus

        The method returns a numerical matrix containing the evaluation of the Jacobian for the pose *AxB* (the self object) and the :math:`2^{nd}` pose *BxC*.

        **This is a pure virtual method that must be implemented by a child class.**

        :param BxC: 2nd pose
        :returns: Evaluation of the :math:`J_{1\\oplus}` Jacobian of the pose compounding operation with respect to the first pose (eq. :eq:`eq-J1oplus`)
        """
        pass

    def J_2oplus(AxB):
        """
        Jacobian of the pose compounding operation (:eq:`eq-oplus`) with respect to the second pose:

        .. math::
            J_{2\\oplus}=\\frac{\\partial  ^Ax_B \\oplus ^Bx_C}{\\partial ^Bx_C}
            :label: eq-J2oplus

        The method returns a numerical matrix containing the evaluation of the Jacobian for the :math:`1^{st}` pose *AxB* (the self object).

        **This is a pure virtual method that must be implemented by a child class.**

        :returns: Evaluation of the :math:`J_{2\\oplus}` Jacobian of the pose compounding operation with respect to the second pose (eq. :eq:`eq-J2oplus`)
        """
        pass

    def ominus(AxB):
        """
        Inverse pose compounding of the *AxB* pose (the self object):

        .. math::
            ^Bx_A = \\ominus ^Ax_B
            :label: eq-ominus

        **This is a pure virtual method that must be implemented by a child class.**

        :returns: A-Frame pose expressed in B-Frame coordinates (eq. :eq:`eq-ominus`)
        """
        pass

    def J_ominus(AxB):
        """
        Jacobian of the inverse pose compounding operation (:eq:`eq-oplus`) with respect the pose *AxB* (the self object):

        .. math::
            J_{\\ominus}=\\frac{\\partial  \\ominus ^Ax_B}{\\partial ^Ax_B}
            :label: eq-Jominus

        Returns the numerical matrix containing the evaluation of the Jacobian for the pose *AxB* (the self object).
        **This is a pure virtual method that must be implemented by a child class.**

        :returns: Evaluation of the :math:`J_{\\ominus}` Jacobian of the inverse pose compounding operation with respect to the pose (eq. :eq:`eq-Jominus3dof`)
        """

        pass

    def boxplus(NxB, BxF):
        print('NxB', NxB)
        NxB=Pose3D(NxB)
        print('lllllllllllllllll',BxF.shape)
        """
        Given a Pose object *NxB* (the self object) and a Feature object *BxF*, it returns the Feature object *NxF*
        providing the same feature but now expresend in the N-Frame.

        :param BxF: Feature object expressed in the B-Frame
        :return: NxF Feature object expressed in the N-Frame
        """
        assert isinstance(BxF,f.Feature), "BxF is not a Feature"
        assert isinstance(NxB,Pose), "NxB is not a Pose"

        return BxF.boxplus(NxB)

    def J_1boxplus(NxB, BxF):
        """
        Jacobian of the pose-feature compounding operation (eq. :eq:`eq-boxplus`) with respect to the robot pose:

        .. math::
            J_{1\\boxplus}=\\frac{\\partial  ^Nx_B \\boxplus ^Bx_F}{\\partial ^Nx_B}
            :label: eq-J1boxplus

        :param BxF: Feature object expressed in the B-Frame
        :return: :math:`J_{1\\boxplus}` Jacobian of the feature compounding operation with respect to the robot pose (eq. :eq:`eq-J1boxplus`)
        """
        assert isinstance(BxF,f.Feature), "BxF is not a Feature"
        assert isinstance(NxB,Pose), "NxB is not a Pose"
        return BxF.J_1boxplus(NxB)

    def J_2boxplus(NxB, BxF):
        """
        Jacobian of the pose-feature compounding operation (eq. :eq:`eq-boxplus`) with respect to the feature:

        .. math::
            J_{2\\boxplus}=\\frac{\\partial  ^Nx_B \\boxplus ^Bx_F}{\\partial ^Bx_F}
            :label: eq-J2boxplus

        :param BxF: Feature object expressed in the B-Frame
        :return: :math:`J_{2\\boxplus}` Jacobian of the feature compounding operation with respect to the feature (eq. :eq:`eq-J2boxplus`)
        """
        assert isinstance(BxF,f.Feature), "BxF is not a Feature"
        assert isinstance(NxB,Pose), "NxB is not a Pose"
        return BxF.J_2boxplus(NxB)

class Pose3D(Pose):
    """
    Definition of a robot pose in 3 DOF (x, y, yaw). The class inherits from a ndarray.
    This class extends the ndarray with the :math:`oplus` and :math:`ominus` operators and the corresponding Jacobians.
    """
    # def __new__(cls, input_array):
    #     """
    #     Constructor of the class. It is called when the class is instantiated. It is required to extend the ndarray numpy class.
    #
    #     :param input_array: array used to initialize the class
    #     :returns: the instance of a Pose3D class object
    #     """
    #     assert input_array.shape == (3, 1), "mean must be a 3x1 vector"
    #
    #     # Input array is an already formed ndarray instance
    #     # We first cast to be our class type
    #     obj = np.asarray(input_array).view(cls)
    #     # Finally, we must return the newly created object:
    #     return obj

    def __new__(cls, input_array=np.array([[0.0, 0.0, 0.0]]).T):
        """
        Constructor of the class. It is called when the class is instantiated. It is required to extend the ndarry numpy class.

        :param input_array: array used to initialize the class
        :returns: the instance of a Pose4D class object
        """

        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # Finally, we must return the newly created object:
        return obj
    def __init__(self, input_array=np.array([[0.0, 0.0, 0.0]]).T):

        assert input_array.shape == (3, 1), "mean must be a 3x1 vector"
    def to_numpy(self):
        # Return the pose as a numpy array [x, y, theta]
        return np.array([self.x, self.y, self.theta])

    def oplus(AxB, BxC):
        """
        Given a Pose3D object *AxB* (the self object) and a Pose3D object *BxC*, it returns the Pose3D object *AxC*.

        .. math::
            \\mathbf{{^A}x_B} &= \\begin{bmatrix} ^Ax_B & ^Ay_B & ^A\\psi_B \\end{bmatrix}^T \\\\
            \\mathbf{{^B}x_C} &= \\begin{bmatrix} ^Bx_C & ^By_C & & ^B\\psi_C \\end{bmatrix}^T \\\\

        The operation is defined as:

        .. math::
            \\mathbf{{^A}x_C} &= \\mathbf{{^A}x_B} \\oplus \\mathbf{{^B}x_C} =
            \\begin{bmatrix}
                ^Ax_B + ^Bx_C  \\cos(^A\\psi_B) - ^By_C  \\sin(^A\\psi_B) \\\\
                ^Ay_B + ^Bx_C  \\sin(^A\\psi_B) + ^By_C  \\cos(^A\\psi_B) \\\\
                ^A\\psi_B + ^B\\psi_C
            \\end{bmatrix}
            :label: eq-oplus3dof

        :param BxC: C-Frame pose expressed in B-Frame coordinates
        :returns: C-Frame pose expressed in A-Frame coordinates
        """
        # TODO: To be completed by the student
        AxC = np.array([[AxB[0,0] + BxC[0,0]*cos(AxB[2,0]) - BxC[1,0]*sin(AxB[2,0])],
                        [AxB[1,0] + BxC[0,0]*sin(AxB[2,0]) + BxC[1,0]*cos(AxB[2,0])],
                        [AxB[2,0] + BxC[2,0]]])
        return AxC

    def J_1oplus(AxB, BxC):
        """
        Jacobian of the pose compounding operation (eq. :eq:`eq-oplus3dof`) with respect to the first pose:

        .. math::
            J_{1\\oplus}=\\frac{\\partial  ^Ax_B \\oplus ^Bx_C}{\\partial ^Ax_B} =
            \\begin{bmatrix}
                1 & 0 &  -^Bx_C \\sin(^A\\psi_B) - ^By_C \\cos(^A\\psi_B) \\\\
                0 & 1 &  ^Bx_C \\cos(^A\\psi_B) - ^By_C \\sin(^A\\psi_B) \\\\
                0 & 0 & 1
            \\end{bmatrix}
            :label: eq-J1oplus3dof

        The method returns a numerical matrix containing the evaluation of the Jacobian for the pose *AxB* (the self object) and the :math:`2^{nd}` posepose *BxC*.

        :param BxC: 2nd pose
        :returns: Evaluation of the :math:`J_{1\\oplus}` Jacobian of the pose compounding operation with respect to the first pose (eq. :eq:`eq-J1oplus3dof`)
        """
        # TODO: To be completed by the student
        J1 = np.array([[1.0,    0.0,    -BxC[0,0]*sin(AxB[2,0])-BxC[1,0]*cos(AxB[2,0])],
                       [0.0,    1.0,     BxC[0,0]*cos(AxB[2,0])-BxC[1,0]*sin(AxB[2,0])],
                       [0.0,    0.0,     1.0]])
        return J1

    def J_2oplus(AxB):
        """
        Jacobian of the pose compounding operation (:eq:`eq-oplus3dof`) with respect to the second pose:

        .. math::
            J_{2\\oplus}=\\frac{\\partial  ^Ax_B \\oplus ^Bx_C}{\\partial ^Bx_C} =
            \\begin{bmatrix}
                \\cos(^A\\psi_B) & -\\sin(^A\\psi_B) & 0  \\\\
                \\sin(^A\\psi_B) & \\cos(^A\\psi_B) & 0  \\\\
                0 & 0 & 1
            \\end{bmatrix}
            :label: eq-J2oplus3dof

        The method returns a numerical matrix containing the evaluation of the Jacobian for the :math:`1^{st} posepose *AxB* (the self object).

        :returns: Evaluation of the :math:`J_{2\\oplus}` Jacobian of the pose compounding operation with respect to the second pose (eq. :eq:`eq-J2oplus3dof`)
        """
        # TODO: To be completed by the student
        J2 = np.array([[cos(AxB[2,0]),    -sin(AxB[2,0]),   0.0],
                       [sin(AxB[2,0]),     cos(AxB[2,0]),   0.0],
                       [0.0,               0.0,             1.0]])
        return J2

    def ominus(AxB):
        """
        Inverse pose compounding of the *AxB* pose (the self objetc):

        .. math::
            ^Bx_A = \\ominus ^Ax_B =
            \\begin{bmatrix}
                -^Ax_B \\cos(^A\\psi_B) - ^Ay_B \\sin(^A\\psi_B) \\\\
                ^Ax_B \\sin(^A\\psi_B) - ^Ay_B \\cos(^A\\psi_B) \\\\
                -^A\\psi_B
            \\end{bmatrix}
            :label: eq-ominus3dof

        :returns: A-Frame pose expressed in B-Frame coordinates (eq. :eq:`eq-ominus3dof`)
        """
        # TODO: To be completed by the student
        AxB = np.array([[-AxB[0,0]*cos(AxB[2,0]) - AxB[1,0]*sin(AxB[2,0])],
                        [ AxB[0,0]*sin(AxB[2,0]) - AxB[1,0]*cos(AxB[2,0])],
                        [-AxB[2,0]]])
        return Pose3D(AxB)

    def J_ominus(AxB):
        """
        Jacobian of the inverse pose compounding operation (:eq:`eq-oplus3dof`) with respect the pose *AxB* (the self object):

        .. math::
            J_{\\ominus}=\\frac{\\partial  \\ominus ^Ax_B}{\\partial ^Ax_B} =
            \\begin{bmatrix}
                -\\cos(^A\\psi_B) & -\\sin(^A\\psi_B) &  ^Ax_B \\sin(^A\\psi_B) - ^Ay_B \\cos(^A\\psi_B) \\\\
                \\sin(^A\\psi_B) & -\\cos(^A\\psi_B) &  ^Ax_B \\cos(^A\\psi_B) + ^Ay_B \\sin(^A\\psi_B) \\\\
                0 & 0 & -1
            \\end{bmatrix}
            :label: eq-Jominus3dof

        Returns the numerical matrix containing the evaluation of the Jacobian for the pose *AxB* (the self object).

        :returns: Evaluation of the :math:`J_{\\ominus}` Jacobian of the inverse pose compounding operation with respect to the pose (eq. :eq:`eq-Jominus3dof`)
        """
        # TODO: To be completed by the student
        J = np.array([[-cos(AxB[2,0]),    -sin(AxB[2,0]),   AxB[0,0]*sin(AxB[2,0])-AxB[1,0]*cos(AxB[2,0])],
                      [ sin(AxB[2,0]),    -cos(AxB[2,0]),   AxB[0,0]*cos(AxB[2,0])+AxB[1,0]*sin(AxB[2,0])],
                      [ 0.0,               0.0,            -1.0]])
        return J
