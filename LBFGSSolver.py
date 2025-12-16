import numpy as np
from collections import deque

class LBFGSSolver:
    def __init__(self, func, grad, alpha_init=1.0, c1=1e-4, c2=0.9, r=0.5, tol=1e-15, n_li=3):
        """
        L-BFGS Solver (Limited Memory).
        Accepts 'n_li' parameter for memory history size.
        """
        self.func = func
        self.grad = grad
        self.alpha_init = alpha_init
        self.c1 = c1
        self.c2 = c2
        self.r = r
        self.tol = tol
        self.n_li = n_li  # This is the parameter causing your error if missing
        
    def _get_m(self, u_vec):
        val = self.func(u_vec.flatten())
        return float(val) if np.ndim(val) == 0 else float(val[0])

    def _get_f(self, u_vec):
        g = self.grad(u_vec.flatten())
        return np.array(g).reshape(-1, 1)

    def _compute_direction(self, f_grad, storage):
        """L-BFGS Two-Loop Recursion"""
        q = f_grad.copy()
        alpha_storage = []
        
        # 1. Backward Loop
        for delta_u, delta_f, rho in reversed(storage):
            alpha = rho * (delta_u.T @ q)[0,0]
            alpha_storage.append(alpha)
            q = q - alpha * delta_f
            
        # 2. Scaling (Le0 = I)
        r_vec = q 

        # 3. Forward Loop
        for (delta_u, delta_f, rho), alpha in zip(storage, reversed(alpha_storage)):
            beta = rho * (delta_f.T @ r_vec)[0,0]
            r_vec = r_vec + (alpha - beta) * delta_u
            
        return -r_vec

    def solve(self, u0):
        # %INITIALISATION
        u = np.array(u0, dtype=float).reshape(-1, 1)
        storage = deque(maxlen=self.n_li)
        
        m_new = self._get_m(u)
        f2 = self._get_f(u)
        m_old = 1e100
        f_new = np.zeros_like(u)
        cnt = 0
        u_prev = u.copy()
        
        print(f"{'Iter':<5} {'m(u)':<20} {'|grad|':<20}")
        print("-" * 50)

        while m_new < m_old:
            m_old = m_new
            f_old = f_new.copy()
            f_new = f2.copy()
            cnt += 1
            
            if cnt == 1:
                h = -f_new
            else:
                h = self._compute_direction(f_new, storage)

            # %Line search
            signal1 = 0
            alpha3 = self.alpha_init
            u_x = u + alpha3 * h
            m3 = self._get_m(u_x)
            f3 = self._get_f(u_x)
            slope = (h.T @ f_new)[0,0]
            
            if (m3 <= m_new + self.c1 * alpha3 * slope) and ((h.T @ f3)[0,0] >= self.c2 * slope):
                signal1 = 1
                
            while (m3 < m_new + self.c1 * alpha3 * slope) and (signal1 == 0):
                alpha3 = alpha3 / self.r
                u_x = u + alpha3 * h
                m3 = self._get_m(u_x)
                f3 = self._get_f(u_x)
                if (m3 <= m_new + self.c1 * alpha3 * slope) and ((h.T @ f3)[0,0] >= self.c2 * slope):
                    signal1 = 1
            
            if signal1 == 0:
                signal2 = 0
                alpha1 = 0
                alpha2 = alpha3 / 2.0
                u_x = u + alpha2 * h
                m2 = self._get_m(u_x)
                f2 = self._get_f(u_x)
                
                while signal2 == 0:
                    if (alpha3 - alpha1) < self.tol:
                        signal2 = 1
                        m2 = m_new
                        f2 = f_new.copy()
                    elif m2 > m_new + self.c1 * alpha2 * slope:
                        alpha3 = alpha2
                        m3 = m2
                        f3 = f2.copy()
                        alpha2 = (alpha1 + alpha2) / 2.0
                        u_x = u + alpha2 * h
                        m2 = self._get_m(u_x)
                        f2 = self._get_f(u_x)
                    elif (h.T @ f2)[0,0] < self.c2 * slope:
                        alpha1 = alpha2
                        alpha2 = (alpha2 + alpha3) / 2.0
                        u_x = u + alpha2 * h
                        m2 = self._get_m(u_x)
                        f2 = self._get_f(u_x)
                    else:
                        signal2 = 1

            u_prev = u.copy()
            u = u_x
            
            if signal1 == 1:
                m_new = m3
                f2 = f3.copy()
            else:
                m_new = m2
            
            delta_u = u - u_prev
            delta_f = f2 - f_new
            denom = (delta_f.T @ delta_u)[0,0]
            
            if abs(denom) > 1e-14:
                rho = 1.0 / denom
                storage.append((delta_u, delta_f, rho))
            
            print(f"{cnt:<5} {m_new:<20.10f} {np.linalg.norm(f2):<20.5e}")

        return u.flatten(), cnt