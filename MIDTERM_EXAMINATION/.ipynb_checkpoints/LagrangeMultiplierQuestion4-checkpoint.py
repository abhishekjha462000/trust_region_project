import numpy as np

class NewtonKKTSolverProblem4:
    def __init__(self, model_func, model_grad, model_hess, tol=1e-12, maxit=60):
        """
        Specialized Newton-KKT Solver for Model 4a with Linear Constraints.
        
        Constraints:
          0.02(j - 1 + u[59+2j]) - 0.1 - u[60+2j] = 0  for j in {1..10}
          
        (Indices are interpreted as 1-based, consistent with vector size 80).
        """
        self.model_func = model_func
        self.model_grad = model_grad
        self.model_hess = model_hess
        self.tol = tol
        self.maxit = maxit
        
        # Fixed Model Parameter g=0 (passed to model functions)
        self.G_vec = np.zeros(80)

    def solve(self, u_start):
        """
        Executes the solver.
        """
        # 1. Setup State
        u = np.array(u_start, dtype=float)
        # We have 10 constraints, so 10 Lagrange multipliers
        lam = np.zeros(10)
        
        print(f"{'Iter':<5} {'Alpha':<10} {'|f|^2':<15}")
        print("-" * 35)

        # 2. Initial KKT Assembly
        f, K = self._assemble_kkt(u, lam)
        f_norm_sq = np.dot(f, f)
        it = 0

        # 3. Main Optimization Loop
        while (f_norm_sq > self.tol) and (it < self.maxit):
            it += 1

            # Solve Linear System: K * [du; dlambda] = -f
            try:
                h_step = np.linalg.solve(K, -f)
            except np.linalg.LinAlgError:
                print("Error: KKT Matrix is singular.")
                return u, lam, it

            du = h_step[:80]
            dl = h_step[80:]

            # --- Damped Update (Patched Line Search) ---
            alpha = 1.0
            phi0 = 0.5 * f_norm_sq
            
            step_accepted = False
            
            # Line search (max 30 reductions)
            for _ in range(30):
                u_try = u + alpha * du
                lam_try = lam + alpha * dl
                
                # Compute new residual (Matrix not needed for check)
                f_try, _ = self._assemble_kkt(u_try, lam_try, calc_matrix=False)
                phi_try = 0.5 * np.dot(f_try, f_try)
                
                # Accept step if residual decreases
                if phi_try < phi0:
                    u = u_try
                    lam = lam_try
                    f = f_try
                    f_norm_sq = np.dot(f, f)
                    step_accepted = True
                    break
                
                alpha *= 0.5

            if not step_accepted:
                print(f"Warning: Line search failed to reduce residual at iter {it}. Stopping.")
                return u, lam, it

            # Recompute Matrix for next Newton step
            _, K = self._assemble_kkt(u, lam, calc_matrix=True)

            print(f"{it:<5d} {alpha:<10.3g} {f_norm_sq:<15.3e}")

        return u, lam, it

    def _assemble_kkt(self, u, lam, calc_matrix=True):
        """Internal helper to assemble Residual f and KKT Matrix K."""
        # 1. Constraint Derivatives
        cu = self._constraints(u)
        Jc = self._jac_constraints(u)
        
        # 2. Model Derivatives (g=0)
        # Note: We pass self.G_vec as the second argument as required
        gm = self.model_grad(u, self.G_vec)
        
        # 3. Lagrangian Gradient: gradL = grad_m + Jc.T * lambda
        gradL = gm + Jc.T @ lam
        
        # 4. Residual Vector f = [gradL; c(u)]
        f = np.concatenate([gradL, cu])
        
        K = None
        if calc_matrix:
            # Model Hessian
            Hm = self.model_hess(u, self.G_vec)
            
            # Hessian of Linear Constraints is ZERO.
            # (No need to loop summing zeros)
            
            # Hessian of Lagrangian (Regularized slightly for stability)
            HL = Hm + 1e-8 * np.eye(80)
            
            # Construct KKT Matrix
            # [ HL   Jc.T ]
            # [ Jc    0   ]
            top = np.hstack([HL, Jc.T])
            bot = np.hstack([Jc, np.zeros((10, 10))])
            K = np.vstack([top, bot])
            
        return f, K

    def _constraints(self, u):
        """
        Computes constraint vector c(u) for j in {1..10}.
        Eq: 0.02(j - 1 + u[59+2j]) - 0.1 - u[60+2j] = 0
        """
        c_vec = np.zeros(10)
        
        for k, j in enumerate(range(1, 11)):
            # Mapping 1-based formula indices to 0-based Python indices
            # u[59+2j] -> index (59 + 2j) - 1
            idx1 = (59 + 2*j) - 1
            idx2 = (60 + 2*j) - 1
            
            term1 = 0.02 * ((j - 1) + u[idx1])
            term2 = 0.1
            term3 = u[idx2]
            
            c_vec[k] = term1 - term2 - term3
            
        return c_vec

    def _jac_constraints(self, u):
        """
        Computes constraint Jacobian Jc.
        """
        J = np.zeros((10, 80))
        
        for k, j in enumerate(range(1, 11)):
            idx1 = (59 + 2*j) - 1
            idx2 = (60 + 2*j) - 1
            
            # Derivative w.r.t u[idx1] is 0.02
            J[k, idx1] = 0.02
            
            # Derivative w.r.t u[idx2] is -1.0
            J[k, idx2] = -1.0
            
        return J