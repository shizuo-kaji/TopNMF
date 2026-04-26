import torch
import numpy as np
from typing import Tuple, Optional

class NMF:
    def __init__(self, n_components: int, max_iter: int = 200, tol: float = 1e-4,
                 max_line_search_iter: int = 20, alpha: float = 0.1, beta: float = 0.5):
        """
        Initialize NMF with second-order optimization and line search.
        
        Args:
            n_components: Number of components for factorization
            max_iter: Maximum number of iterations
            tol: Tolerance for convergence
            max_line_search_iter: Maximum number of line search iterations
            alpha: Sufficient decrease parameter (Armijo condition)
            beta: Step size reduction factor
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.max_line_search_iter = max_line_search_iter
        self.alpha = alpha
        self.beta = beta
        
    def _objective(self, X: torch.Tensor, W: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        """
        Compute the Frobenius norm objective function.
        
        Args:
            X: Input data matrix
            W: Weight matrix
            H: Feature matrix
            
        Returns:
            Objective value
        """
        reconstruction = torch.mm(W, H)
        return 0.5 * torch.norm(X - reconstruction) ** 2
    
    def _compute_hessian(self, W: torch.Tensor, H: torch.Tensor, fixed: str) -> torch.Tensor:
        """
        Compute the Hessian matrix for either W or H update.
        
        Args:
            W: Weight matrix
            H: Feature matrix
            fixed: Which matrix to compute Hessian for ('W' or 'H')
            
        Returns:
            Hessian matrix
        """
        if fixed == 'H':
            return 2 * torch.mm(H, H.t())
        else:
            return 2 * torch.mm(W.t(), W)
            
    def _line_search(self, X: torch.Tensor, current_matrix: torch.Tensor, 
                    direction: torch.Tensor, grad: torch.Tensor, 
                    fixed_matrix: torch.Tensor, fixed: str) -> Tuple[torch.Tensor, float]:
        """
        Perform backtracking line search using Armijo condition.
        
        Args:
            X: Input data matrix
            current_matrix: Current W or H matrix
            direction: Search direction
            grad: Gradient
            fixed_matrix: The fixed matrix (H when updating W, or W when updating H)
            fixed: Which matrix is being updated ('W' or 'H')
            
        Returns:
            Updated matrix and step size
        """
        step_size = 1.0
        initial_obj = self._objective(X, 
                                    W=current_matrix if fixed == 'H' else fixed_matrix,
                                    H=fixed_matrix if fixed == 'H' else current_matrix)
        
        grad_direction = torch.sum(grad * direction)
        
        for _ in range(self.max_line_search_iter):
            # Compute candidate update
            candidate = current_matrix + step_size * direction
            candidate = torch.clamp(candidate, min=0)  # Project to non-negative orthant
            
            # Compute new objective
            new_obj = self._objective(X,
                                    W=candidate if fixed == 'H' else fixed_matrix,
                                    H=fixed_matrix if fixed == 'H' else candidate)
            
            # Check Armijo condition
            if new_obj <= initial_obj + self.alpha * step_size * grad_direction:
                return candidate, step_size
            
            step_size *= self.beta
        
        # If line search fails, return minimal update
        return current_matrix + 1e-10 * direction, step_size
    
    def _newton_update(self, X: torch.Tensor, W: torch.Tensor, H: torch.Tensor, 
                      fixed: str) -> torch.Tensor:
        """
        Perform Newton update with line search for either W or H.
        
        Args:
            X: Input data matrix
            W: Weight matrix
            H: Feature matrix
            fixed: Which matrix to update ('W' or 'H')
            
        Returns:
            Updated matrix
        """
        if fixed == 'H':
            # Update W
            current = W
            fixed_matrix = H
            grad = -2 * torch.mm(X, H.t()) + 2 * torch.mm(W, torch.mm(H, H.t()))
        else:
            # Update H
            current = H
            fixed_matrix = W
            grad = -2 * torch.mm(W.t(), X) + 2 * torch.mm(W.t(), torch.mm(W, H))
        
        hessian = self._compute_hessian(W, H, fixed)
        hessian = hessian + torch.eye(hessian.shape[0], device=X.device) * 1e-10
        
        try:
            direction = torch.solve(-grad, hessian)[0]
        except:
            # Fallback to gradient descent if Hessian is ill-conditioned
            direction = -grad
        
        # Perform line search
        updated_matrix, step_size = self._line_search(X, current, direction, grad, 
                                                    fixed_matrix, fixed)
        
        return updated_matrix
    
    def fit_transform(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fit the NMF model and return the factorization.
        
        Args:
            X: Input data matrix of shape (n_samples, n_features)
            
        Returns:
            W: Weight matrix of shape (n_samples, n_components)
            H: Feature matrix of shape (n_components, n_features)
        """
        n_samples, n_features = X.shape
        
        # Initialize W and H with random non-negative values
        W = torch.rand(n_samples, self.n_components, device=X.device)
        H = torch.rand(self.n_components, n_features, device=X.device)
        
        # Normalize W and H
        W = W / torch.norm(W, dim=0, keepdim=True)
        H = H / torch.norm(H, dim=1, keepdim=True)
        
        prev_error = float('inf')
        
        for iteration in range(self.max_iter):
            # Update W with H fixed
            W = self._newton_update(X, W, H, fixed='H')
            
            # Update H with W fixed
            H = self._newton_update(X, W, H, fixed='W')
            
            # Compute reconstruction error
            error = torch.norm(X - torch.mm(W, H))
            
            # Check convergence
            if abs(prev_error - error) < self.tol:
                break
                
            prev_error = error
            
        return W, H
    
    def fit(self, X: torch.Tensor) -> 'NMF':
        """
        Fit the NMF model.
        
        Args:
            X: Input data matrix
            
        Returns:
            self: The fitted model
        """
        self.W_, self.H_ = self.fit_transform(X)
        return self
    
    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """
        Transform new data using the fitted model.
        
        Args:
            X: Input data matrix
            
        Returns:
            Transformed data matrix
        """
        if not hasattr(self, 'H_'):
            raise RuntimeError("Model must be fitted before transform")
        
        W = torch.rand(X.shape[0], self.n_components, device=X.device)
        for _ in range(self.max_iter):
            W = self._newton_update(X, W, self.H_, fixed='H')
        
        return W
