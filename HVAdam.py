import math
import torch
import torch.optim


from torch.optim.optimizer import Optimizer

version_higher = (torch.__version__ >= "1.5.0")


class HVAdam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-16, gamma=0.,
                 weight_decay=0, amsgrad=False, weight_decouple=False, fixed_decay=False, rectify=False,
                 degenerated_to_sgd=False):


        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= gamma:
            raise ValueError("Invalid epsilon value: {}".format(gamma))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        self.degenerated_to_sgd = degenerated_to_sgd
        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):
            for param in params:
                if 'betas' in param and (param['betas'][0] != betas[0] or param['betas'][1] != betas[1]):
                    param['buffer'] = [[None, None, None] for _ in range(10)]

        defaults = dict(lr=lr, betas=betas, eps=eps, gamma=gamma,
                        weight_decay=weight_decay, amsgrad=amsgrad, buffer=[[None, None, None] for _ in range(10)])
        super(HVAdam, self).__init__(params, defaults)

        self.degenerated_to_sgd = degenerated_to_sgd
        self.weight_decouple = weight_decouple
        self.rectify = rectify
        self.fixed_decay = fixed_decay
        if self.weight_decouple:
            print('Weight decoupling enabled in HVAdam')
            if self.fixed_decay:
                print('Weight decay fixed')
        if self.rectify:
            print('Rectification enabled in HVAdam')
        if amsgrad:
            print('AMSGrad enabled in HVAdam')

    def __setstate__(self, state):
        super(HVAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def reset(self):
        for group in self.param_groups:
            group['flag'] = -1
            for p in group['params']:
                state = self.state[p]
                amsgrad = group['amsgrad']

                # State initialization
                state['step'] = 0
                # Exponential moving average of gradient values
                state['exp_avg'] = torch.zeros_like(p.data, memory_format=torch.preserve_format) \
                    if version_higher else torch.zeros_like(p.data)

                # Exponential moving average of squared gradient values
                state['exp_avg_var'] = torch.zeros_like(p.data, memory_format=torch.preserve_format) \
                    if version_higher else torch.zeros_like(p.data)
                
                state['last_v'] = torch.zeros_like(p.data, memory_format=torch.preserve_format) \
                    if version_higher else torch.zeros_like(p.data)

                if amsgrad:
                    # Maintains max of all exp. moving avg. of sq. grad. values
                    state['max_exp_avg_var'] = torch.zeros_like(p.data, memory_format=torch.preserve_format) \
                        if version_higher else torch.zeros_like(p.data)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            if 'flag' not in group:
                group['flag'] = -1
            mlv = 0.
            mm = 0.
            lvlv = 0.

            for p in group['params']:
                if p.grad is None:
                    continue
                # grad = p.grad.data.float()
                grad = p.grad
                # group['lrxz']+=len(p.grad)
                if grad.is_sparse:
                    raise RuntimeError(
                        'HVAdam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                beta1, beta2 = group['betas']

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data, memory_format=torch.preserve_format) \
                        if version_higher else torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_var'] = torch.zeros_like(p.data, memory_format=torch.preserve_format) \
                        if version_higher else torch.zeros_like(p.data)
                    state['last_v'] = torch.zeros_like(p.data, memory_format=torch.preserve_format) \
                        if version_higher else torch.zeros_like(p.data)

                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_var'] = torch.zeros_like(p.data, memory_format=torch.preserve_format) \
                            if version_higher else torch.zeros_like(p.data)

                # perform weight decay, check if decoupled weight decay
                if self.weight_decouple:
                    if not self.fixed_decay:
                        p.data.mul_(1.0 - group['lr'] * group['weight_decay'])
                    else:
                        p.data.mul_(1.0 - group['weight_decay'])
                else:
                    if group['weight_decay'] != 0:
                        grad.add_(p.data, alpha=group['weight_decay'])

                # get current state variable
                exp_avg, exp_avg_var = state['exp_avg'], state['exp_avg_var']
                eps = group['eps']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step = state['step']

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                temp1 = (grad - state['last_v']) ** 2
                temp2 = (grad - exp_avg) ** 2
                grad_residual = (temp1 / (temp2 + group['gamma'] * temp1 + eps)) * temp1

                exp_avg_bc = exp_avg / bias_correction1

                exp_avg_var.mul_(beta2).add_(grad_residual * (1 - beta2))
                last_v = state['last_v']
                mm += exp_avg_bc.norm() ** 2
                mlv += (exp_avg_bc * last_v).sum()
                lvlv += last_v.norm() ** 2


                if amsgrad:
                    max_exp_avg_var = state['max_exp_avg_var']
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_var, exp_avg_var.add_(group['eps']), out=max_exp_avg_var)

                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_var.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_var.add_(group['eps']).sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                # update
                if not self.rectify:
                    # Default update
                    step_size = group['lr'] / bias_correction1
                    p.data.addcdiv_(exp_avg, denom, value=-step_size)
                else:  # Rectified update, forked from RAdam
                    buffered = group['buffer'][int(state['step'] % 10)]
                    if state['step'] == buffered[0]:
                        N_sma, step_size = buffered[1], buffered[2]
                    else:
                        buffered[0] = state['step']
                        beta2_t = beta2 ** state['step']
                        N_sma_max = 2 / (1 - beta2) - 1
                        N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                        buffered[1] = N_sma

                        # more conservative since it's an approximated value
                        if N_sma >= 5:
                            step_size = math.sqrt(
                                (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (
                                        N_sma_max - 2)) / (1 - beta1 ** state['step'])
                        elif self.degenerated_to_sgd:
                            step_size = 1.0 / (1 - beta1 ** state['step'])
                        else:
                            step_size = -1
                        buffered[2] = step_size

                    if N_sma >= 5:
                        denom = exp_avg_var.sqrt().add_(group['eps'])
                        p.data.addcdiv_(exp_avg, denom, value=-step_size * group['lr'])
                    elif step_size > 0:
                        p.data.add_(exp_avg, alpha=-step_size * group['lr'])

            if mm + lvlv - 2 * mlv == 0:
                k = 0.
            else:
                k = (mm - mlv) / (mm + lvlv - 2 * mlv)


            if group['flag'] != -1:
                beta1, beta2 = group['betas']
                vv = abs(k ** 2 * lvlv + (1 - k) ** 2 * mm + 2 * k * (1 - k) * mlv)
                delta = (k * mlv + (1 - k) * mm) / (vv * mm) ** 0.5
                if abs(delta) > 1.:
                    continue
                beta1, beta2 = group['betas']
                group['exp_delta'] = float(delta) * (1. - beta2) + group['exp_delta'] * beta2

                group['step2'] += 1
                bias_correction2 = 1 - beta2 ** group['step2']

                if group['exp_delta'] / bias_correction2 < 0.1:
                    group['flag'] = -1
                    continue


                group['flag'] = 0
                lr_f = 10 ** (group['exp_delta'] * 6 - 3)


                for p in group['params']:
                    if p.grad is None:
                        continue
                    beta1, beta2 = group['betas']
                    state = self.state[p]
                    exp_avg_bc = state['exp_avg'] / (1 - beta1 ** state['step'])
                    last_v = state['last_v']
                    state['last_v'] = float(k) * last_v + float(1 - k) * exp_avg_bc
                    step_size = group['lr'] * lr_f
                    step1 = (step_size) * state['last_v']
                    p.data.add_(- step1)
            else:
                group['step2'] = 0
                group['flag'] = 0
                group['exp_delta'] = 0.
                for p in group['params']:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    state['last_v'] = state['exp_avg'] / (1 - beta1 ** state['step'])

        return loss


