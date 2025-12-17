import numpy as np
class NewtonOptimizer:
    """
    Generic class to perform Newton's Method optimization.
    It takes the objective, gradient, and Hessian functions as arguments
    during initialization, making it flexible.
    """
    def __init__(self, objective_func, grad_func, hessian_func, tol=1e-12, max_iter=100):
        self._m = objective_func
        self._f = grad_func
        self._K = hessian_func
        self.tol = tol
        self.max_iter = max_iter
        
    def optimize(self, u0, *args):
        """
        Runs the Newton's Method optimization loop.
        
        Parameters:
            u0 (array-like): Initial guess (u).
            *args: Parameters passed to the objective, gradient, and hessian functions.
            
        Returns:
            dict: Dictionary containing the optimized parameters (u_star) and objective value (m_star).
        """
        u = np.array(u0, dtype=float)
        
        # We assume the functions accept u first, followed by *args
        # e.g., func(u, *args)

        # print(f"--- Starting Newton's Method ---")
        # print(f"Initial u: {u0}, Tolerance: {self.tol:.1e}")
        # print(f"Iter | u1      | u2      | m(u)      | ||f||")
        # print("-" * 50)

        for i in range(self.max_iter):
            # 1. Compute Gradient f = dm/du 
            f = np.array(self._f(u, *args)) 
            grad_norm = np.linalg.norm(f)
            
            # Report status
            current_m = self._m(u, *args)
            # print(f"{i:4d} | {u[0]:7.4f} | {u[1]:7.4f} | {current_m:9.6f} | {grad_norm:.3e}")
            
            # 2. Check for convergence 
            if grad_norm < self.tol:
                print("-" * 50)
                print(f"Converged (||f|| < {self.tol:.1e}) at iteration {i}")
                m_star = self._m(u, *args)
                return {"u_star": u.round(15), "m_star": m_star}
            
            # 3. Compute Hessian K 
            K = np.array(self._K(u, *args))
            
            # 4. Solve K h = -f 
            try:
                h = np.linalg.solve(K, -f)
            except np.linalg.LinAlgError:
                print("---")
                print("Error: Hessian K is singular. Stopping.")
                m_star = self._m(u, *args)
                return {"u_star": u.round(15), "m_star": m_star}

            # 5. Update u = u + h 
            u = u + h

        print("---")
        print(f"Reached maximum iterations ({self.max_iter}) without full convergence.")
        m_star = self._m(u, *args)
        return {"u_star": u.round(15), "m_star": m_star}
