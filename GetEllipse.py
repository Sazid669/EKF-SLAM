import  numpy as np

def GetEllipse(x, P, sigma=3):
    # computes the points of the ellipse contour for the 2 first DOFs
    R, D, V = np.linalg.svd(P[0:2, 0:2], full_matrices=False)
    S = np.sqrt(D)

    # prepare a list of points to draw the ellipse
    alpha = np.linspace(0, 2 * np.pi, 100)  # list of angles equally spaced between 0 and 2*pi
    unit_circle_points = np.array([np.cos(alpha), np.sin(alpha)])  # circle
    ellipse_points = sigma * R @ np.diag(S) @ unit_circle_points + x[0:2]  # ellipse
    return ellipse_points
