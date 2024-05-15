import cv2
import numpy as np
from scipy.stats import chi2


def plot_ellipse(image, mean, covariance, color, confidence=0.95, label=None):
    """Draw 95% confidence ellipse of a 2-D Gaussian distribution.
    Parameters
    ----------
    image: cv image
        The image where the ellipse will be drawn
    mean : array_like
        The mean vector of the Gaussian distribution (ndim=1).
    covariance : array_like
        The 2x2 covariance matrix of the Gaussian distribution.
    color : Scalar
        The color of the ellipse (RBG tuple)
    confidence : Optional[float]
        Confidence value in 0-1. By default it's set to 0.95 (95%)
    label : Optional[str]
        A text label that is placed at the center of the ellipse.
    """

    if len(mean) != 2 or covariance.shape != (2, 2):
        raise RuntimeError("Mean should be a 2x1 vector and " +
                           "Covariance a 2x2 matrix. You can " +
                           "provide array slices like cov[0:2, 0:2]")

    scale = chi2.ppf(confidence, 2)
    vals, vecs = np.linalg.eigh(scale * covariance)
    indices = vals.argsort()[::-1]
    vals, vecs = np.sqrt(vals[indices]), vecs[:, indices]

    center = int(mean[0] + .5), int(mean[1] + .5)
    axes = int(vals[0] + .5), int(vals[1] + .5)
    angle = int(180. * np.arctan2(vecs[1, 0], vecs[0, 0]) / np.pi)

    cv2.ellipse(image, center, axes, angle, 0, 360, color, 2)
    if label is not None:
        cv2.putText(image, label, center, cv2.FONT_HERSHEY_PLAIN,
                    1, color, 1)
