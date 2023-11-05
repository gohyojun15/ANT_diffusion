import numpy as np
import torch

try:
    import cvxpy as cp
except ModuleNotFoundError:
    from pip._internal import main as pip

    pip(["install", "--user", "cvxpy"])
    import cvxpy as cp


class NashMTL:
    def __init__(self, num_tasks, model_module, device, opt):
        self.device = device
        self.model = model_module
        self.task_num = num_tasks
        self.step = 0
        self.prvs_alpha_param = None
        self.init_gtg = np.eye(self.task_num)
        self.prvs_alpha = np.ones(self.task_num, dtype=np.float32)
        self.normalization_factor = np.ones((1,))
        self.opt = opt

        self.update_weights_every = 25
        self.optim_niter = 20
        self.max_norm = 1

    def _compute_grad_dim(self):
        self.grad_index = []
        for param in self.model.parameters():
            if param.requires_grad:
                self.grad_index.append(param.data.numel())
        self.grad_dim = sum(self.grad_index)

    def compute_grad(self, losses):
        grads = torch.zeros(self.task_num, self.grad_dim).to(self.device)
        for tn in range(self.task_num):
            self.opt.zero_grad()
            losses[tn].backward(retain_graph=True)
            grad = []
            for param in self.model.parameters():
                if param.requires_grad:
                    grad.append(param.grad.data.view(-1))
            grads[tn] = torch.cat(grad, dim=0)
        return grads

    def _calc_phi_alpha_linearization(self):
        G_prvs_alpha = self.G_param @ self.prvs_alpha_param
        prvs_phi_tag = 1 / self.prvs_alpha_param + (1 / G_prvs_alpha) @ self.G_param
        phi_alpha = prvs_phi_tag @ (self.alpha_param - self.prvs_alpha_param)
        return phi_alpha

    def _init_optim_problem(self):
        self.alpha_param = cp.Variable(shape=(self.task_num,), nonneg=True)
        self.prvs_alpha_param = cp.Parameter(shape=(self.task_num,), value=self.prvs_alpha)
        self.G_param = cp.Parameter(shape=(self.task_num, self.task_num), value=self.init_gtg)
        self.normalization_factor_param = cp.Parameter(shape=(1,), value=np.array([1.0]))

        self.phi_alpha = self._calc_phi_alpha_linearization()

        G_alpha = self.G_param @ self.alpha_param
        constraint = []
        for i in range(self.task_num):
            constraint.append(
                -cp.log(self.alpha_param[i] * self.normalization_factor_param) - cp.log(G_alpha[i])
                <= 0
            )
        obj = cp.Minimize(cp.sum(G_alpha) + self.phi_alpha / self.normalization_factor_param)
        self.prob = cp.Problem(obj, constraint)

    def _stop_criteria(self, gtg, alpha_t):
        return (
            (self.alpha_param.value is None)
            or (np.linalg.norm(gtg @ alpha_t - 1 / (alpha_t + 1e-10)) < 1e-3)
            or (np.linalg.norm(self.alpha_param.value - self.prvs_alpha_param.value) < 1e-6)
        )

    def solve_optimization(self, gtg: np.array):
        self.G_param.value = gtg
        self.normalization_factor_param.value = self.normalization_factor

        alpha_t = self.prvs_alpha
        for _ in range(self.optim_niter):
            self.alpha_param.value = alpha_t
            self.prvs_alpha_param.value = alpha_t

            try:
                self.prob.solve(solver=cp.ECOS, warm_start=True, max_iters=100)
            except:
                self.alpha_param.value = self.prvs_alpha_param.value

            if self._stop_criteria(gtg, alpha_t):
                break

            alpha_t = self.alpha_param.value

        if alpha_t is not None:
            self.prvs_alpha = alpha_t

        return self.prvs_alpha

    def __call__(self, losses, logger):
        if self.step == 0:
            self._init_optim_problem()
        if (self.step % self.update_weights_every) == 0:
            self._compute_grad_dim()
            grads = self.compute_grad(losses)
            GTG = torch.mm(grads, grads.t())
            self.normalization_factor = torch.norm(GTG).detach().cpu().numpy().reshape((1,))
            GTG = GTG / self.normalization_factor.item()
            alpha = self.solve_optimization(GTG.cpu().detach().numpy())
            logger.info(f"nash_weight: {alpha}")
        alpha = torch.from_numpy(self.prvs_alpha).to(torch.float32).to(self.device)

        loss = 0
        for i in range(len(losses)):
            loss += alpha[i] * losses[i]
        self.step += 1
        return loss / len(losses)
