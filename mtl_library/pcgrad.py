import copy
import torch
import random


class PCgrad:
    def __init__(self, num_tasks, model_module):
        self.num_tasks = num_tasks
        self.grad_index = []
        self.model = model_module
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.grad_index.append(param.data.numel())
            else:
                print(name)
        self.grad_dim = sum(self.grad_index)

    def do_pc_grad(self, grads):
        with torch.no_grad():
            pc_grads = copy.deepcopy(grads)
            for tn_i in range(self.num_tasks):
                task_index = list(range(self.num_tasks))
                random.shuffle(task_index)
                for tn_j in task_index:
                    g_ij = torch.dot(pc_grads[tn_i], grads[tn_j])
                    if g_ij < 0:
                        pc_grads[tn_i].add_(grads[tn_j], alpha=-g_ij / (grads[tn_j].norm().pow(2)))
            new_grads = pc_grads.sum(0)
        return new_grads

    def _reset_grad(self, new_grads):
        count = 0
        for param in self.model.parameters():
            if param.requires_grad:
                beg = 0 if count == 0 else sum(self.grad_index[:count])
                end = sum(self.grad_index[: (count + 1)])
                param.grad = new_grads[beg:end].view(param.data.size())
                count += 1
