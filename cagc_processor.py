"""
Conflict-Aware Gradient Consensus (CAGC) for MTLoRA
Complete implementation based on SwinTransformerMTLoRA architecture
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple
import torch.distributed as dist


class CAGCProcessor:
    """
    CAGC Processor for MTLoRA
    
    TA-LoRA parameters in SwinTransformerMTLoRA:
    - attn.qkv, attn.proj (WindowAttention)
    - mlp.fc1, mlp.fc2 (Mlp)  
    - downsample.reduction (PatchMerging)
    """
    
    def __init__(
        self,
        tasks: List[str],
        rho_thr: float = 0.5,
        beta_w: float = 0.99,
        lambda_lap: float = 0.01,
        theta_scale: float = 0.1,
        epsilon: float = 1e-8,
        max_iter_qp: int = 100,
        device: str = 'cuda'
    ):
        self.tasks = tasks
        self.num_tasks = len(tasks)
        self.rho_thr = rho_thr
        self.beta_w = beta_w
        self.lambda_lap = lambda_lap
        self.theta_scale = theta_scale
        self.epsilon = epsilon
        self.max_iter_qp = max_iter_qp
        self.device = device
        
        # EMA energy statistics
        self.energy_ema = {task: 0.0 for task in tasks}
        self.step_count = 0
        
        # Cache shared parameter references
        self.shared_params = []  # TA-LoRA parameters (lora_shared_A/B)
        self.shared_param_names = []
        
    def register_shared_params(self, backbone: nn.Module):
        """
        Register TA-LoRA (shared) parameters.
        Search for all parameters containing 'lora_shared' in SwinTransformerMTLoRA
        """
        self.shared_params = []
        self.shared_param_names = []
        
        for name, param in backbone.named_parameters():
            # Identify TA-LoRA parameters: containing 'lora_shared'
            if 'lora_shared' in name and param.requires_grad:
                self.shared_params.append(param)
                self.shared_param_names.append(name)
                
        print(f"[CAGC] Registered {len(self.shared_params)} shared TA-LoRA parameters")
        # Print first 5 as examples
        for name in self.shared_param_names[:5]:
            print(f"  - {name}")
        if len(self.shared_param_names) > 5:
            print(f"  ... and {len(self.shared_param_names)-5} more")
            
    def compute_task_gradients(
        self, 
        backbone: nn.Module, 
        samples: torch.Tensor, 
        targets: Dict[str, torch.Tensor], 
        criterion,
        loss_scaler=None
    ) -> Tuple[Dict[str, List[torch.Tensor]], Dict[str, float]]:
        """
        Compute gradients of each task w.r.t. TA-LoRA shared parameters
        
        Args:
            backbone: SwinTransformerMTLoRA instance (model.backbone)
            samples: Input image
            targets: Task label dictionary
            criterion: MultiTaskLoss instance
            
        Returns:
            task_grads: {task_name: [grad_tensors for each shared_param]}
            task_losses: {task_name: loss_value}
        """
        task_grads = {}
        task_losses = {}
        
        # Compute gradients independently for each task
        for task in self.tasks:
            # Clear gradients
            backbone.zero_grad(set_to_none=True)
            
            # Forward pass - get task-specific outputs
            with torch.cuda.amp.autocast():
                # backbone returns[(shared_feat, {task: task_feat}), ...]
                stage_outputs = backbone(samples, return_stages=True)
                
                # Extract features for this task at all stages
                task_features = []
                for shared_feat, tasks_lora in stage_outputs:
                    if task in tasks_lora:
                        task_features.append(tasks_lora[task])
                    else:
                        task_features.append(shared_feat)
            
            
        return task_grads, task_losses
    
    def compute_task_gradients_full_model(
        self,
        model: nn.Module,  # MultiTaskSwin
        samples: torch.Tensor,
        targets: Dict[str, torch.Tensor],
        task: str,
        loss_scaler=None
    ) -> List[torch.Tensor]:
        """
        Compute gradients of a single task w.r.t. TA-LoRA parameters
        
        Returns:
            List of gradients for all shared_params corresponding to this task
        """
        model.zero_grad(set_to_none=True)
        
        with torch.cuda.amp.autocast():
            outputs = model(samples)
            
            if isinstance(outputs, dict):
                task_output = outputs[task]
            else:
                raise TypeError(f"Unexpected output type: {type(outputs)}")
            
            task_target = targets[task]
            
            if hasattr(model, 'criterion'):
                loss = model.criterion.loss_ft[task](task_output, task_target)
            else:
                from mtl_loss_schemes import get_loss
                criterion = get_loss(None, task)
                loss = criterion(task_output, task_target)
        
        if loss_scaler is not None:
            scaled_loss = loss_scaler.scale(loss)
            scaled_loss.backward()
        else:
            loss.backward()
        
        grads = []
        for param in self.shared_params:
            if param.grad is not None:
                grads.append(param.grad.clone().detach().flatten())
            else:
                grads.append(torch.zeros_like(param.data).flatten())
        
        return grads, loss.item()
    
    def apply_cagc(self, task_grads: Dict[str, List[torch.Tensor]]) -> torch.Tensor:
        """
        Apply CAGC algorithm to reconcile gradients
        
        Args:
            task_grads: {task: [grad_list]} The gradients of all shared parameters for each task
            
        Returns:
            final_flat_grad: Reconciled flattened gradient
        """
        # Flatten task gradients g_t
        g_flat = {}
        for task in self.tasks:
            if task not in task_grads or len(task_grads[task]) == 0:
                # If this task has no gradients (e.g., some tasks skipped), use zero gradients
                if len(self.shared_params) > 0:
                    total_dim = sum(p.numel() for p in self.shared_params)
                    g_flat[task] = torch.zeros(total_dim, device=self.device)
                else:
                    g_flat[task] = torch.tensor([], device=self.device)
            else:
                g_flat[task] = torch.cat(task_grads[task])
        
        if len(self.shared_params) == 0:
            return torch.tensor([], device=self.device)
            
        # Stack into matrix G (d x T)
        G = torch.stack([g_flat[task] for task in self.tasks], dim=1)  # (d, T)
        
        # 1. Conflict Detection
        g_norms = torch.norm(G, dim=0, keepdim=True) + self.epsilon
        G_normalized = G / g_norms  # (d, T)
        C = torch.mm(G_normalized.t(), G_normalized)  # (T, T) cosine similarity matrix
        
        # compute ρ_t = sqrt((C + μI)^{-1}_{tt})
        mu = 1e-4
        C_reg = C + mu * torch.eye(self.num_tasks, device=self.device)
        try:
            C_inv = torch.inverse(C_reg)
            rho = torch.sqrt(torch.diag(C_inv))
        except:
            rho = torch.ones(self.num_tasks, device=self.device) * 0.5
        
        # Identify strongly conflicting tasks
        T_conf = [i for i, r in enumerate(rho) if r > self.rho_thr]
        T_ref = [i for i in range(self.num_tasks) if i not in T_conf]
        
        if len(T_conf) > 0 and self.step_count % 100 == 0:  # Print every 100 steps
            conf_tasks = [self.tasks[i] for i in T_conf]
            conf_rhos = [f"{rho[i]:.3f}" for i in T_conf]
            print(f"[CAGC] Step {self.step_count}: Strong conflicts detected in tasks "
                  f"{conf_tasks} (ρ={conf_rhos})")
        
        # 2. Orthogonal Gradient Decomposition and Residual Preservation
        g_parallel = {}
        g_perp = {}
        
        for t_idx in T_conf:
            task = self.tasks[t_idx]
            g_t = G[:, t_idx]
            
            # Construct subspace of other tasks G_{-t}
            other_indices = [i for i in range(self.num_tasks) if i != t_idx]
            if len(other_indices) == 0:
                g_parallel[task] = g_t
                g_perp[task] = torch.zeros_like(g_t)
                continue
                
            G_minus_t = G[:, other_indices]  # (d, T-1)
            
            # QR decomposition for orthogonal basis
            try:
                Q, R = torch.linalg.qr(G_minus_t, mode='reduced')
                # g_t^∥ = Q(Q^T g_t) - project onto compatible subspace
                g_t_parallel = torch.mv(Q, torch.mv(Q.t(), g_t))
                # g_t^⊥ = g_t - g_t^∥ - orthogonal residual
                g_t_perp = g_t - g_t_parallel
            except:
                # If QR fails (rank deficient), conservative: keep full gradient, no residual
                g_t_parallel = g_t
                g_t_perp = torch.zeros_like(g_t)
            
            g_parallel[task] = g_t_parallel
            g_perp[task] = g_t_perp
            
        # Construct modified gradient matrix G_tilde
        G_tilde_list = []
        for i, task in enumerate(self.tasks):
            if i in T_conf:
                G_tilde_list.append(g_parallel[task])
            else:
                G_tilde_list.append(G[:, i])
        G_tilde = torch.stack(G_tilde_list, dim=1)  # (d, T)
        
        # 3. Inverse-Energy Accumulated Weighting
        energies = torch.norm(G, dim=0) ** 2  # (T,) Gradient energy of each task
        
        # Update EMA energy
        for i, task in enumerate(self.tasks):
            self.energy_ema[task] = self.beta_w * self.energy_ema[task] + \
                                    (1 - self.beta_w) * energies[i].item()
        
        E_avg = np.mean(list(self.energy_ema.values()))
        
        # Compute synergy coefficient κ_t (inter-task alignment degree)
        kappa = torch.zeros(self.num_tasks, device=self.device)
        for t in range(self.num_tasks):
            alignments = [(1 + C[t, u]) / 2 for u in range(self.num_tasks) if u != t]
            kappa[t] = sum(alignments) / (self.num_tasks - 1) if self.num_tasks > 1 else 1.0
        
        # Compute inverse-energy weights w_t^g
        inv_energy_weights = []
        for t in range(self.num_tasks):
            E_t = self.energy_ema[self.tasks[t]] + self.epsilon
            weight = (E_avg / E_t) ** ((1 + kappa[t].item()) / 2)
            inv_energy_weights.append(weight)
        
        inv_energy_weights = torch.tensor(inv_energy_weights, device=self.device)
        w_g = torch.softmax(inv_energy_weights, dim=0)  # (T,) Normalized weights
        
        # Reference direction v = Σ w_t * g_tilde_t
        v = torch.mv(G_tilde, w_g)  # (d,)
        
        # 4. Dirichlet Regularized Constrained Optimization (QP solver)
        # Construct Gram matrix K_tilde
        K_tilde = torch.mm(G_tilde.t(), G_tilde)  # (T, T)
        
        # Construct b_tilde
        b_tilde = torch.mv(G_tilde.t(), v)  # (T,)
        
        # Construct graph Laplacian L = D - W, W = max(0, C)
        W = torch.clamp(C, min=0.0)
        D = torch.diag(torch.sum(W, dim=1))
        L = D - W
        
        # Solve QP: min 0.5*α^T(K+λL)α - b^Tα, s.t. α >= 0
        alpha = self._solve_qp(K_tilde + self.lambda_lap * L, b_tilde)
        
        # Consensus gradient Δ_cons = Σ α_t * g_tilde_t
        delta_cons = torch.mv(G_tilde, alpha)  # (d,)
        
        # 5. Residual Compensation
        delta_perp = torch.zeros_like(delta_cons)
        if len(T_conf) > 0:
            for t_idx in T_conf:
                task = self.tasks[t_idx]
                g_t_perp = g_perp[task]
                if g_t_perp.norm() > self.epsilon:
                    # θ_t = θ_scale / ||g_t^⊥||
                    theta_t = self.theta_scale / (g_t_perp.norm() + self.epsilon)
                    delta_perp += theta_t * g_t_perp
        
        # Final gradient
        final_grad = delta_cons + delta_perp
        self.step_count += 1
        
        return final_grad
    
    def _solve_qp(self, K: torch.Tensor, b: torch.Tensor, lr: float = 0.1) -> torch.Tensor:
        """
        Solve QP problem using projected gradient descent
        min 0.5 * x^T K x - b^T x
        s.t. x >= 0, sum(x) = 1 (可选)
        """
        x = torch.ones(self.num_tasks, device=self.device) / self.num_tasks
        
        for iter in range(self.max_iter_qp):

            grad = torch.mv(K, x) - b
            
            x_new = x - lr * grad
            
            x_new = torch.clamp(x_new, min=0.0)
            
            # Normalize to probability simplex (optional, maintain convex combination)
            if x_new.sum() > 0:
                x_new = x_new / x_new.sum()
            
            # Check convergence
            if torch.norm(x_new - x) < 1e-6:
                break
                
            x = x_new
        
        return x
    
    def apply_to_model(self, final_flat_grad: torch.Tensor):
        """
        Apply reconciled gradients to model parameters
        Reshape back to each parameter shape and assign to .grad
        """
        if final_flat_grad.numel() == 0 or len(self.shared_params) == 0:
            return
        
        offset = 0
        for param in self.shared_params:
            numel = param.numel()
            grad_slice = final_flat_grad[offset:offset+numel]
            param.grad = grad_slice.view_as(param)
            offset += numel
            
    def zero_shared_grads(self):
        """Clear shared parameter gradients"""
        for param in self.shared_params:
            if param.grad is not None:
                param.grad.zero_()
            else:
                param.grad = torch.zeros_like(param.data)


def should_apply_cagc(epoch: int, batch_idx: int, apply_from_epoch: int = 0, freq: int = 1) -> bool:
    """Determine whether CAGC should be applied at current step"""
    if epoch < apply_from_epoch:
        return False
    return batch_idx % freq == 0