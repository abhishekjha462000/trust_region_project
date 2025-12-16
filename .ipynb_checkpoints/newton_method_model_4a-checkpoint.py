import numpy as np

class NewtonOptimizer:
    # ... (init method remains the same) ...
    def __init__(self, objective_func, grad_func, hessian_func, tol=1e-12, max_iter=100):
        self._m = objective_func
        self._f = grad_func
        self._K = hessian_func
        self.tol = tol
        self.max_iter = max_iter
        
    def optimize(self, u0, *args):
        """
        Runs the Newton's Method optimization loop for Model 4A.
        
        Parameters:
            u0 (array-like): Initial guess (u).
            *args: Parameters (g, a, b, etc.) passed to the functions. 
                   For Model 4A, this should be (g,).
        """
        u = np.array(u0, dtype=float)
        
        # Determine the dimension from the initial guess
        N = len(u) 

        print(f"--- Starting Newton's Method for Model 4A (N={N}) ---")
        print(f"Initial ||u||: {np.linalg.norm(u):.4e}, Tolerance: {self.tol:.1e}")
        print(f"Iter | m(u)      | ||f||")
        print("-" * 30)

        for i in range(self.max_iter):
            # 1. Compute Gradient f = dm/du 
            f = np.array(self._f(u, *args)) 
            grad_norm = np.linalg.norm(f)
            
            # Report status
            current_m = self._m(u, *args)
            # We skip printing u vectors due to high dimension (N=80)
            print(f"{i:4d} | {current_m:9.6f} | {grad_norm:.3e}")
            
            # 2. Check for convergence 
            if grad_norm < self.tol:
                print("-" * 30)
                print(f"Converged (||f|| < {self.tol:.1e}) at iteration {i}")
                m_star = current_m
                return {"u_star": u, "m_star": m_star}
            
            # 3. Compute Hessian K 
            K = np.array(self._K(u, *args))
            
            # 4. Solve K h = -f 
            try:
                # np.linalg.solve is used for K h = -f
                h = np.linalg.solve(K, -f)
            except np.linalg.LinAlgError:
                print("---")
                print("Error: Hessian K is singular. Stopping.")
                return {"u_star": u, "m_star": current_m}

            # 5. Parameter Update (Pure Newton Step - Assuming stability for high-dim)
            # If instability occurs (as with Rosenbrock), a Line Search must be added here.
            u = u + h

        print("---")
        print(f"Reached maximum iterations ({self.max_iter}) without full convergence.")
        m_star = self._m(u, *args)
        return {"u_star": u, "m_star": m_star}