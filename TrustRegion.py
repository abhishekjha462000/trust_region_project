import numpy as np
from scipy.sparse import issparse

class TrustRegion:
    """
     Trust-Region implementation faithful to the textbook Algorithms 3 and 4.
    - Uses (dense) Cholesky factorization for the preconditioner Mit = L L^T
      and diagonal inflation if necessary (Eq. 2.147).
    - Implements the preconditioned Steihaug-Toint conjugate gradient exactly.
    - Implements Algorithm 4 (rho and TR updates) exactly as in the text.
    """

    def __init__(self, obj_func, grad_func, hess_func, u0, g_extra=None,
                 R_TR_initial=1.0, R_TR_max=100.0, max_iter=500,
                 gtol=1e-6, verbose=True, max_precond_attempts=12):

        self.obj_func = obj_func
        self.grad_func = grad_func
        self.hess_func = hess_func
        self.u = np.array(u0, dtype=float).copy()
        self.g_extra = g_extra
        self.R_TR = float(R_TR_initial)
        self.R_TR_max = float(R_TR_max)
        self.max_iter = int(max_iter)
        self.gtol = float(gtol)
        self.verbose = bool(verbose)
        self.max_precond_attempts = int(max_precond_attempts)

        # history
        self.history = {'iter': [], 'obj': [], 'grad_norm': [], 'R_TR': [],
                        'rho': [], 'step_norm': []}

    # ---------------------------
    # Utilities for preconditioner
    # ---------------------------
    def _to_dense(self, A):
        if issparse(A):
            return A.toarray()
        return np.array(A, dtype=float, copy=True)

    def build_preconditioner_cholesky(self, K):
        """
        Build preconditioner Mit ~ K using (dense) Cholesky with diagonal inflation
        following Eq. (2.147) in the thesis.
        Returns:
          L (lower-triangular numpy array) such that Mit = L @ L.T
        or None if we fail after attempts.
        Notes:
          - If any diagonal entries of K are negative, we set them to abs(diag)
            before attempting factorization (as described in the thesis).
          - On failure we multiply diagonal by (1 + 10^(i-8)) repeatedly.
        """
        Kd = self._to_dense(K)
        n = Kd.shape[0]
        # Ensure symmetry (average)
        Kd = 0.5 * (Kd + Kd.T)

        # Fix negative diagonal entries by absolute value first (text mentions this)
        diag = np.diag(Kd).copy()
        if np.any(diag <= 0):
            diag = np.abs(diag)
            np.fill_diagonal(Kd, diag)

        # Attempt Cholesky with progressive diagonal inflation
        for i in range(self.max_precond_attempts):
            try:
                # small jitter to assist numeric stability (only on repeats)
                if i > 0:
                    inflation = 1.0 + 10.0**(i - 8)   # Eq. (2.147)
                    d_new = np.diag(Kd) * inflation
                    np.fill_diagonal(Kd, d_new)

                # Try Cholesky
                L = np.linalg.cholesky(Kd)        # returns lower-triangular L s.t. Kd = L @ L.T
                return L
            except np.linalg.LinAlgError:
                # not positive definite yet -> try again with bigger inflation
                continue

        # If we reach here, failed to factor
        return None

    # ---------------------------
    # Preconditioner apply and solve
    # ---------------------------
    def _make_precond_ops(self, L):
        """
        Given L (lower triangular) for Mit = L @ L.T,
        return:
          - solve_M_inv(vec): solves Mit * x = vec  => returns x = Mit^{-1} vec
          - apply_M(vec): returns Mit @ vec
          - inner_M(x,y): returns x^T Mit y
        """

        def solve_M_inv(b):
            # Solve L L^T x = b  => first solve L y = b, then L^T x = y
            y = np.linalg.solve(L, b)
            x = np.linalg.solve(L.T, y)
            return x

        def apply_M(v):
            # Mit @ v = L @ (L.T @ v)
            return L @ (L.T @ v)

        def inner_M(x, y):
            # x^T Mit y
            temp = apply_M(y)
            return float(np.dot(x, temp))

        return solve_M_inv, apply_M, inner_M

    # ---------------------------
    # Algorithm 3: preconditioned Steihaug-Toint (text exact)
    # ---------------------------
    def steihaug_toint_precond(self, f_it, K_it, R_TR):
        """
        Implements Algorithm 3 exactly (lines referenced to the thesis).
        Inputs:
          f_it : gradient at current iterate (vector)
          K_it : Hessian at current iterate (matrix or array)
          R_TR : trust-region radius (scalar)
        Returns:
          h_star (vector), boundary_reached (bool)
        """
        n = f_it.shape[0]
        # Build preconditioner Mit ~ K_it
        L = self.build_preconditioner_cholesky(K_it)
        if L is None:
            # fall back to identity preconditioner (unpreconditioned)
            solve_M_inv = lambda x: x.copy()
            def apply_M(v): return v.copy()
            def inner_M(x,y): return float(np.dot(x,y))
        else:
            solve_M_inv, apply_M, inner_M = self._make_precond_ops(L)

        # Line 1: r = -f_it, Solve Mit·p = r for p, q = p, h* = 0
        r = -f_it.copy()
        p = solve_M_inv(r.copy())
        q = p.copy()
        h_star = np.zeros_like(f_it)
        boundary_reached = False

        # Main loop (lines 2-26)
        while True:
            # compute K p
            Kp = K_it.dot(p) if issparse(K_it) else np.dot(K_it, p)
            pKp = float(np.dot(p, Kp))

            # Line 3: check negative curvature
            if pKp <= 0.0:
                # Lines 4-8: negative curvature
                boundary_reached = True

                # Compute quadratic coefficients in M-inner-product:
                # a = p^T M p, b = 2 p^T M h_star, c = h_star^T M h_star - R_TR^2
                a = inner_M(p, p)
                b = 2.0 * inner_M(p, h_star)
                c = inner_M(h_star, h_star) - R_TR**2

                # Solve for positive root alpha
                disc = b*b - 4.0*a*c
                if a <= 0 or disc < 0:
                    alpha = 0.0
                else:
                    alpha = (-b + np.sqrt(disc)) / (2.0 * a)

                h_star = h_star + alpha * p
                break

            # Line 10: alpha = (r^T q) / (p^T K p)
            rq = float(np.dot(r, q))
            alpha = rq / pKp

            # Line 11: check if (h* + alpha p)^T M (h* + alpha p) > R_TR^2
            h_new = h_star + alpha * p
            norm_sq_new = inner_M(h_new, h_new)

            if norm_sq_new > R_TR**2:
                # Lines 12-16: step would exceed TR, find alpha on boundary
                boundary_reached = True

                a = inner_M(p, p)
                b = 2.0 * inner_M(p, h_star)
                c = inner_M(h_star, h_star) - R_TR**2

                disc = b*b - 4.0*a*c
                if a <= 0 or disc < 0:
                    alpha = 0.0
                else:
                    alpha = (-b + np.sqrt(disc)) / (2.0 * a)

                h_star = h_star + alpha * p
                break

            # Line 18: h* = h* + alpha p
            h_star = h_new.copy()

            # Line 19: phi = r^T p  (store before modifying r)
            phi = float(np.dot(r, p))

            # Line 20: r = r - alpha K p
            r = r - alpha * Kp

            # Line 21: check convergence
            r_norm = np.linalg.norm(r)
            f_norm = np.linalg.norm(f_it)
            if r_norm < max(1e-15, 1e-5 * f_norm):
                # converged enough
                break

            # Line 24: Solve Mit · q = r for q
            q = solve_M_inv(r.copy())

            # Line 25: p = q + (r^T q / phi) * p
            numerator = float(np.dot(r, q))
            beta = numerator / phi
            p = q + beta * p

        return h_star, boundary_reached

    # ---------------------------
    # Algorithm 4: TR update
    # ---------------------------
    def trust_region_update(self, h_star, f_it, K_it, flag_boundary, u_current, obj_current):
        """
        Implements Algorithm 4 exactly from the text.
        Returns: u_next, R_TR_next, rho, step_accepted
        """
        u_candidate = u_current + h_star
        # Evaluate gradient at candidate
        try:
            f_it_hstar = self.grad_func(u_candidate, self.g_extra)
            if np.any(np.isnan(f_it_hstar)):
                rho = 0.0
            else:
                numerator = float(np.dot(h_star, (f_it + f_it_hstar)))
                Kh = K_it.dot(h_star) if issparse(K_it) else np.dot(K_it, h_star)
                denominator = 2.0 * float(np.dot(h_star, f_it)) + float(np.dot(h_star, Kh))
                if abs(denominator) > 1e-15:
                    rho = numerator / denominator
                else:
                    rho = 0.0
        except Exception:
            rho = 0.0
            f_it_hstar = None

        if rho < 0.25:
            # reject and shrink
            u_next = u_current.copy()
            R_TR_next = 0.25 * self.R_TR
            step_accepted = False
        else:
            # accept
            u_next = u_current + h_star
            step_accepted = True
            R_TR_next = self.R_TR
            if rho > 0.75 and flag_boundary:
                R_TR_next = min(2.0 * self.R_TR, self.R_TR_max)

        return u_next, R_TR_next, rho, step_accepted

    # ---------------------------
    # Main optimizer
    # ---------------------------
    def evaluate_functions(self, u):
        """Return obj, grad, hess at u (pass g_extra if provided)."""
        if self.g_extra is None:
            obj = self.obj_func(u)
            grad = self.grad_func(u)
            hess = self.hess_func(u)
        else:
            obj = self.obj_func(u, self.g_extra)
            grad = self.grad_func(u, self.g_extra)
            hess = self.hess_func(u, self.g_extra)
        return obj, grad, hess

    def optimize(self):
        if self.verbose:
            print(f"{'Iter':>4} {'Objective':>12} {'GradNorm':>12} {'StepNorm':>10} {'R_TR':>8} {'rho':>8} {'Status':>12}")
            print("-" * 80)

        obj_current, grad_current, hess_current = self.evaluate_functions(self.u)
        grad_norm = np.linalg.norm(grad_current)

        for it in range(self.max_iter):
            # record
            self.history['iter'].append(it)
            self.history['obj'].append(obj_current)
            self.history['grad_norm'].append(grad_norm)
            self.history['R_TR'].append(self.R_TR)

            # convergence check
            if grad_norm < self.gtol:
                if self.verbose:
                    print(f"{it:4d} {obj_current:12.6e} {grad_norm:12.6e} {'-':>10} {self.R_TR:8.3e} {'-':>8} Converged")
                break

            # Algorithm 3: solve TR subproblem
            try:
                h_star, flag_boundary = self.steihaug_toint_precond(grad_current, hess_current, self.R_TR)
            except Exception as e:
                # If algorithm 3 fails unexpectedly, shrink TR and continue
                if self.verbose:
                    print(f"Iter {it}: Algorithm 3 failed: {e}")
                self.R_TR *= 0.25
                continue

            step_norm = np.linalg.norm(h_star)

            # Algorithm 4: update
            u_next, R_TR_next, rho, step_accepted = self.trust_region_update(
                h_star, grad_current, hess_current, flag_boundary, self.u, obj_current
            )

            # apply updates
            self.u = u_next.copy()
            self.R_TR = R_TR_next

            if step_accepted:
                obj_current, grad_current, hess_current = self.evaluate_functions(self.u)
                grad_norm = np.linalg.norm(grad_current)
            # else: keep same obj_current, grad_current

            # store
            self.history['rho'].append(rho)
            self.history['step_norm'].append(step_norm)

            # status string
            if step_accepted:
                status = "Accept"
                if rho > 0.75 and flag_boundary:
                    status = "Accept+Expand"
            else:
                status = "Reject"

            if self.verbose:
                print(f"{it:4d} {obj_current:12.6e} {grad_norm:12.6e} {step_norm:10.3e} {self.R_TR:8.3e} {rho:8.3f} {status:>12}")

            # safety stop
            if self.R_TR < 1e-12:
                if self.verbose:
                    print("Trust region radius too small; stopping.")
                break

        # final
        final_obj, final_grad, _ = self.evaluate_functions(self.u)
        if self.verbose:
            print("\n" + "="*80)
            print("OPTIMIZATION COMPLETE")
            print("="*80)
            print(f"Final objective: {final_obj:.10e}")
            print(f"Final gradient norm: {np.linalg.norm(final_grad):.6e}")
            print(f"Iterations: {len(self.history['iter'])}")
            print(f"Final trust-region radius: {self.R_TR:.3e}")
            print("="*80)

        return self.u, self.history