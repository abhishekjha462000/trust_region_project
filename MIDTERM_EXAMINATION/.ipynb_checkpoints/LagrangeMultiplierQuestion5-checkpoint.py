import numpy as np
from LagrangeMultiplierQuestion4 import NewtonKKTSolverProblem4

class NewtonInequalitySolver(NewtonKKTSolverProblem4):
    def solve(self, u_start):
        """
        Solves the optimization problem subject to c(u) >= 0.
        """
        u = np.array(u_start, dtype=float)
        n_vars = len(u)
        n_con = 10 
        
        # 1. Initialize Active Set
        # active_mask[j] = True means constraint j is treated as equality c_j(u) = 0
        active_mask = np.zeros(n_con, dtype=bool)
        
        # Initial check: Activate any constraint that violates c(u) >= 0 (i.e., c < 0)
        c_vals = self._constraints(u)
        active_mask[c_vals < -1e-6] = True
        
        print(f"{'Outer':<5} {'Active Set Indices':<30} {'Max Lambda':<10}")
        print("=" * 60)

        # Outer Loop: Adjusts the Active Set
        for outer_iter in range(30): 
            
            # --- A. Solve Equality Problem for Current Active Set ---
            active_indices = np.where(active_mask)[0]
            n_active = len(active_indices)
            
            if n_active == 0:
                # If no constraints are active, solve unconstrained Newton step
                # (Simplification: solving H * du = -g)
                # We use the assemble helper but ignore constraint parts
                f_full, K_full = self._assemble_kkt(u, np.zeros(n_con))
                Hm = K_full[:n_vars, :n_vars]
                gm = f_full[:n_vars]
                
                try:
                    du = np.linalg.solve(Hm, -gm)
                except np.linalg.LinAlgError:
                    print("Singular Hessian in unconstrained step.")
                    return u, np.zeros(n_con), outer_iter
                
                lam_full = np.zeros(n_con) # No multipliers
            else:
                # Build Reduced KKT System for active constraints
                f_full, K_full = self._assemble_kkt(u, np.zeros(n_con))
                
                # Slicing the Full KKT matrix
                Hm = K_full[:n_vars, :n_vars]
                J_full = K_full[n_vars:, :n_vars]
                grad_m = f_full[:n_vars]
                c_full = f_full[n_vars:]
                
                J_active = J_full[active_indices]
                c_active = c_full[active_indices]
                
                # Construct KKT [H  J.T]
                #               [J   0 ]
                top = np.hstack([Hm, J_active.T])
                bot = np.hstack([J_active, np.zeros((n_active, n_active))])
                K_active = np.vstack([top, bot])
                
                # RHS = [-grad; -c] (Driving residuals to 0)
                rhs = np.concatenate([-grad_m, -c_active])
                
                try:
                    sol = np.linalg.solve(K_active, rhs)
                except np.linalg.LinAlgError:
                    print("Singular KKT matrix in active set step.")
                    break
                    
                du = sol[:n_vars]
                dlam_active = sol[n_vars:]
                
                # Map active lambdas to full vector
                lam_full = np.zeros(n_con)
                lam_full[active_indices] = dlam_active

            # --- B. Update Solution 'u' ---
            # (A simple full step is taken here. For robust production code, 
            # a line search checking for inactive constraint crossings is ideal)
            u = u + du
            
            # --- C. Check Active Set Changes ---
            set_changed = False
            
            # 1. REMOVE constraints?
            # For c(u) >= 0, binding lambda must be NEGATIVE.
            # If lambda is POSITIVE, the constraint is pushing the wrong way -> Inactive.
            if n_active > 0:
                max_lam = np.max(lam_full[active_indices])
                lam_print = f"{max_lam:.2e}"
                
                if max_lam > 1e-6:
                    # Remove the one with the largest positive lambda
                    # (It wants to leave the boundary the most)
                    idx_to_remove = active_indices[np.argmax(lam_full[active_indices])]
                    active_mask[idx_to_remove] = False
                    set_changed = True
                    print(f"{outer_iter:<5} {str(active_indices):<30} {lam_print:<10} -> Removing {idx_to_remove}")
            else:
                lam_print = "-"

            # 2. ADD constraints?
            # Check for violations: c(u) < 0
            c_check = self._constraints(u)
            violated_indices = np.where((~active_mask) & (c_check < -1e-6))[0]
            
            if not set_changed and len(violated_indices) > 0:
                # Add the most violated constraint
                idx_to_add = violated_indices[np.argmin(c_check[violated_indices])]
                active_mask[idx_to_add] = True
                set_changed = True
                print(f"{outer_iter:<5} {str(active_indices):<30} {lam_print:<10} -> Adding {idx_to_add} (Val: {c_check[idx_to_add]:.2e})")

            # --- D. Termination ---
            if not set_changed:
                print(f"{outer_iter:<5} {str(active_indices):<30} {lam_print:<10} -> Converged.")
                return u, lam_full, outer_iter
                
            if n_active == 0 and not set_changed:
                 print(f"{outer_iter:<5} (None) -> Converged unconstrained.")
                 return u, lam_full, outer_iter

        print("Max outer iterations reached.")
        return u, lam_full, 30