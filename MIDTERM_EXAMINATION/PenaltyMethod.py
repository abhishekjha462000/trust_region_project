from NewtonOptimizer import NewtonOptimizer
import numpy as np

class PenaltyEqualityRunner:
    def __init__(self, model_func, model_grad, model_hess, mu_start=10.0, mu_growth=10.0, max_mu=1e9):
        self.model_func = model_func
        self.model_grad = model_grad
        self.model_hess = model_hess
        self.mu = mu_start
        self.mu_growth = mu_growth
        self.max_mu = max_mu
        
        # Internal G vector (fixed parameter)
        self.G_vec = np.zeros(80)

    def solve(self, u_start):
        u = np.array(u_start, dtype=float)
        
        print(f"{'Outer':<5} {'Mu':<10} {'Obj':<15} {'|Constraint|':<15}")
        print("=" * 55)
        
        iteration = 0
        while self.mu <= self.max_mu:
            iteration += 1
            
            # --- A. Define Penalized Functions for THIS Mu ---
            # These are "closures" that capture 'self.mu'
            
            def phi_func(u, *args):
                # Phi = f(u) + (mu/2) * ||c(u)||^2
                f_val = self.model_func(u, self.G_vec)
                c_val = self._constraints(u)
                return f_val + (self.mu / 2.0) * np.dot(c_val, c_val)

            def phi_grad(u, *args):
                # Grad = grad_f + mu * J.T * c
                g_val = self.model_grad(u, self.G_vec)
                c_val = self._constraints(u)
                J_val = self._jac_constraints(u)
                return g_val + self.mu * (J_val.T @ c_val)

            def phi_hess(u, *args):
                # Hess = hess_f + mu * J.T * J
                H_val = self.model_hess(u, self.G_vec)
                J_val = self._jac_constraints(u)
                return H_val + self.mu * (J_val.T @ J_val)
            
            # --- B. Instantiate Your NewtonOptimizer ---
            # We create a new optimizer for every outer loop to handle the updated functions
            optimizer = NewtonOptimizer(phi_func, phi_grad, phi_hess, tol=1e-8, max_iter=50)
            
            # --- C. Run Optimization ---
            result = optimizer.optimize(u)
            u = result["u_star"]
            
            # --- D. Check Convergence ---
            c_final = self._constraints(u)
            max_viol = np.max(np.abs(c_final))
            obj_val = self.model_func(u, self.G_vec)
            
            print(f"{iteration:<5d} {self.mu:<10.1e} {obj_val:<15.6f} {max_viol:<15.4e}")
            
            if max_viol < 1e-8:
                print("-" * 55)
                print("Converged: Constraints satisfied.")
                return u
            
            # Increase Penalty
            self.mu *= self.mu_growth
            
        return u

    # --- Constraint Helpers (Same as before) ---
    def _constraints(self, u):
        c_vec = np.zeros(10)
        for k, j in enumerate(range(1, 11)):
            idx1 = (59 + 2*j) - 1
            idx2 = (60 + 2*j) - 1
            c_vec[k] = 0.02 * ((j - 1) + u[idx1]) - 0.1 - u[idx2]
        return c_vec

    def _jac_constraints(self, u):
        J = np.zeros((10, 80))
        for k, j in enumerate(range(1, 11)):
            idx1 = (59 + 2*j) - 1
            idx2 = (60 + 2*j) - 1
            J[k, idx1] = 0.02
            J[k, idx2] = -1.0
        return J