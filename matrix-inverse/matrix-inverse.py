import numpy as np

def matrix_inverse(A):
    try:
        A = np.array(A, dtype=float)

        # Validate 2D square
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            return None

        n = A.shape[0]

        # Check singularity
        if np.isclose(np.linalg.det(A), 0.0):
            return None

        A_inv = np.linalg.inv(A)

        # Verify accuracy: ||A A^-1 − I|| < 1e-7
        I = np.eye(n)
        if np.linalg.norm(A @ A_inv - I) >= 1e-7:
            return None

        return A_inv

    except Exception:
        return None