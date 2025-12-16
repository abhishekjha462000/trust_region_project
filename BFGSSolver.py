import numpy as np

class BFGSSolver:
    def __init__(self, func, grad, alpha_init=1.0, c1=1e-4, c2=0.9, r=0.5, tol=1e-15):
        """
        Initializes the BFGS Solver with Wolfe line search.
        
        Parameters:
        func (callable): Objective function m(u).
        grad (callable): Gradient function f(u).
        alpha_init (float): Initial step size.
        c1 (float): Armijo parameter.
        c2 (float): Curvature parameter.
        r (float): Decay factor for step size.
        tol (float): Tolerance for bisection method.
        """
        self.func = func
        self.grad = grad
        self.alpha_init = alpha_init
        self.c1 = c1
        self.c2 = c2
        self.r = r
        self.tol = tol
        
    def _get_m(self, u_vec):
        """Helper to safely call user function with 1D array."""
        val = self.func(u_vec.flatten())
        return float(val) if np.ndim(val) == 0 else float(val[0])

    def _get_f(self, u_vec):
        """Helper to safely call user gradient and return column vector."""
        g = self.grad(u_vec.flatten())
        return np.array(g).reshape(-1, 1)

    def solve(self, u0):
        """
        Executes the BFGS optimization.
        
        Parameters:
        u0 (array-like): Starting point.
        
        Returns:
        u_final (np.ndarray): The optimized coordinates.
        cnt (int): Number of iterations.
        """
        # %INITIALISATION
        u = np.array(u0, dtype=float).reshape(-1, 1)
        nvar = len(u)
        Le_new = np.eye(nvar)  # Inverse Hessian approximation
        
        # %QUASI-NEWTON ALGORITHM
        m_new = self._get_m(u)
        f2 = self._get_f(u)
        
        m_old = 1e100
        f_new = np.zeros_like(u)
        cnt = 0
        u_prev = u.copy()
        
        print(f"{'Iter':<5} {'m(u)':<20} {'|grad|':<20}")
        print("-" * 50)

        # Step 5
        while m_new < m_old:
            # Step 6
            m_old = m_new
            f_old = f_new.copy()
            Le_old = Le_new.copy()
            f_new = f2.copy()
            delta_f = f_new - f_old
            
            # %Determine search direction
            if cnt == 0:
                # Step 8: Steepest Descent
                h = -Le_old @ f_new
            else:
                # Step 10: BFGS Update
                delta_u = u - u_prev
                
                A = (delta_u.T @ delta_f)[0,0]
                B = delta_u @ delta_u.T
                C = (delta_f.T @ Le_old @ delta_f)[0,0]
                D = (Le_old @ delta_f) @ delta_u.T
                E = delta_u @ (delta_f.T @ Le_old)
                
                if abs(A) > 1e-14:
                    term1 = ((A + C) / A**2) * B
                    term2 = (D + E) / A
                    Le_new = Le_old + term1 - term2
                else:
                    Le_new = Le_old
                
                # Step 11
                h = -Le_new @ f_new

            # %Line search
            # Step 13
            signal1 = 0
            alpha3 = self.alpha_init
            u_x = u + alpha3 * h
            
            m3 = self._get_m(u_x)
            f3 = self._get_f(u_x)
            
            slope = (h.T @ f_new)[0,0]
            
            # Step 15
            if (m3 <= m_new + self.c1 * alpha3 * slope) and ((h.T @ f3)[0,0] >= self.c2 * slope):
                signal1 = 1
                
            # Step 18: Expansion
            while (m3 < m_new + self.c1 * alpha3 * slope) and (signal1 == 0):
                alpha3 = alpha3 / self.r
                u_x = u + alpha3 * h
                m3 = self._get_m(u_x)
                f3 = self._get_f(u_x)
                
                if (m3 <= m_new + self.c1 * alpha3 * slope) and ((h.T @ f3)[0,0] >= self.c2 * slope):
                    signal1 = 1
            
            # %Bisection method
            if signal1 == 0:
                signal2 = 0
                alpha1 = 0
                alpha2 = alpha3 / 2.0
                u_x = u + alpha2 * h
                m2 = self._get_m(u_x)
                f2 = self._get_f(u_x)
                
                while signal2 == 0:
                    # Step 30
                    if (alpha3 - alpha1) < self.tol:
                        signal2 = 1
                        m2 = m_new
                        f2 = f_new.copy()
                    
                    # Step 32
                    elif m2 > m_new + self.c1 * alpha2 * slope:
                        alpha3 = alpha2
                        m3 = m2
                        f3 = f2.copy()
                        alpha2 = (alpha1 + alpha2) / 2.0
                        u_x = u + alpha2 * h
                        m2 = self._get_m(u_x)
                        f2 = self._get_f(u_x)
                    
                    # Step 35
                    elif (h.T @ f2)[0,0] < self.c2 * slope:
                        alpha1 = alpha2
                        alpha2 = (alpha2 + alpha3) / 2.0
                        u_x = u + alpha2 * h
                        m2 = self._get_m(u_x)
                        f2 = self._get_f(u_x)
                        
                    else:
                        signal2 = 1

            # %Complete iteration
            u_prev = u.copy()
            u = u_x
            cnt += 1
            
            if signal1 == 1:
                m_new = m3
                f2 = f3.copy()
            else:
                m_new = m2
                
            print(f"{cnt:<5} {m_new:<20.10f} {np.linalg.norm(f2):<20.5e}")

        return u.flatten(), cnt