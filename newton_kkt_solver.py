import numpy as np

class NewtonKKTSolver:
    def __init__(self, model_func, model_grad, model_hess, R=9.0, xR=4.5, yR=12.5, tol=1e-12, maxit=60):
        """
        Initializes the Newton-KKT Solver for Model 4a with constraints (Eq. 8.16).
        
        Parameters:
        model_func : Callable m(u, g) -> float
        model_grad : Callable grad_m(u, g) -> (80,)
        model_hess : Callable hess_m(u, g) -> (80, 80)
        R, xR, yR  : Constraint geometry parameters
        tol        : Convergence tolerance (for |f|^2)
        maxit      : Maximum iterations
        """
        self.model_func = model_func
        self.model_grad = model_grad
        self.model_hess = model_hess
        self.R = R
        self.xR = xR
        self.yR = yR
        self.tol = tol
        self.maxit = maxit
        
        # Fixed Model Parameter g=0
        self.G_vec = np.zeros(80)

    def solve(self, u_start):
        """
        Executes the solver.
        
        Returns:
        u (np.ndarray): Optimized variables (80,)
        lam (np.ndarray): Lagrange multipliers (10,)
        iters (int): Number of iterations
        """
        # 1. Setup State
        u = np.array(u_start, dtype=float)
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

            # Damped Update (Step-Halving Line Search)
            alpha = 1.0
            phi0 = 0.5 * f_norm_sq
            
            u_next = u
            lam_next = lam
            f_next = f
            
            # Line search (max 30 reductions)
            for _ in range(30):
                u_try = u + alpha * du
                lam_try = lam + alpha * dl
                
                # Compute new residual (Matrix not needed for check)
                f_try, _ = self._assemble_kkt(u_try, lam_try, calc_matrix=False)
                phi_try = 0.5 * np.dot(f_try, f_try)
                
                if phi_try < phi0:
                    u_next = u_try
                    lam_next = lam_try
                    f_next = f_try
                    break
                
                alpha *= 0.5

            # Update State
            u = u_next
            lam = lam_next
            f = f_next
            f_norm_sq = np.dot(f, f)

            # Recompute Matrix for next Newton step
            _, K = self._assemble_kkt(u, lam, calc_matrix=True)

            print(f"{it:<5d} {alpha:<10.3g} {f_norm_sq:<15.3e}")

        return u, lam, it

    def _assemble_kkt(self, u, lam, calc_matrix=True):
        """Internal helper to assemble Residual f and KKT Matrix K."""
        # 1. Constraint Derivatives
        cu = self._c_eq816(u)
        Jc = self._jac_c_eq816(u)
        
        # 2. Model Derivatives (Fixed call signature)
        gm = self.model_grad(u, self.G_vec)
        
        # 3. Lagrangian Gradient: gradL = grad_m + Jc.T * lambda
        gradL = gm + Jc.T @ lam
        
        # 4. Residual Vector f = [gradL; c(u)]
        f = np.concatenate([gradL, cu])
        
        K = None
        if calc_matrix:
            # Model Hessian
            Hm = self.model_hess(u, self.G_vec)
            
            # Constraint Hessians Summation
            Hc = np.zeros((80, 80))
            for k in range(10):
                Hc += lam[k] * self._hess_c_eq816(u, k)
                
            # Hessian of Lagrangian + Regularization
            HL = Hm + Hc + 1e-6 * np.eye(80)
            
            # Construct KKT Matrix
            top = np.hstack([HL, Jc.T])
            bot = np.hstack([Jc, np.zeros((10, 10))])
            K = np.vstack([top, bot])
            
        return f, K

    def _c_eq816(self, u):
        """Computes constraint vector c(u)."""
        cu = np.zeros(10)
        for k, j_node in enumerate(range(31, 41)):
            idx_x = 2 * (j_node - 1)
            idx_y = 2 * (j_node - 1) + 1
            dx = (self.xR - j_node + 31) - u[idx_x]
            dy = (self.yR - 4)           - u[idx_y]
            cu[k] = np.sqrt(dx**2 + dy**2) - self.R
        return cu

    def _jac_c_eq816(self, u):
        """Computes constraint Jacobian Jc."""
        J = np.zeros((10, 80))
        for k, j_node in enumerate(range(31, 41)):
            idx_x = 2 * (j_node - 1)
            idx_y = 2 * (j_node - 1) + 1
            dx = (self.xR - j_node + 31) - u[idx_x]
            dy = (self.yR - 4)           - u[idx_y]
            s = np.sqrt(dx**2 + dy**2)
            if s > 1e-14:
                J[k, idx_x] = -dx / s
                J[k, idx_y] = -dy / s
        return J

    def _hess_c_eq816(self, u, k_idx):
        """Computes Hessian of k-th constraint."""
        Hk = np.zeros((80, 80))
        j_node = 31 + k_idx
        idx_x = 2 * (j_node - 1)
        idx_y = 2 * (j_node - 1) + 1
        dx = (self.xR - j_node + 31) - u[idx_x]
        dy = (self.yR - 4)           - u[idx_y]
        s = np.sqrt(dx**2 + dy**2)
        if s > 1e-14:
            s3 = s**3
            Hxx = (dy**2) / s3
            Hyy = (dx**2) / s3
            Hxy = -(dx * dy) / s3
            Hk[idx_x, idx_x] = Hxx
            Hk[idx_y, idx_y] = Hyy
            Hk[idx_x, idx_y] = Hxy
            Hk[idx_y, idx_x] = Hxy
        return Hk