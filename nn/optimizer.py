import torch

from nn.base import Optimizer


class GradientDescend(Optimizer):
    """
    Градиентый спуск с возможностью использования Momentum и моментов Нестерова.

    Атрибуты
    --------
    `config` : dict
        Словарь c гиперпараметрами оптимизатора. Для SGD следующие параметры:
        `learning_rate`, `momentum` для SGD с Momentum, `is_nesterov` для использования момента Нестерова.
    `state` :  dict
        Словарь cо состоянием оптимизатора. Нужен, чтобы сохранять старые значения градиентов.
    """

    def __init__(self, lr=0.01, momentum=0., is_nesterov=False):
        super(GradientDescend, self).__init__()
        self.config['lr'] = lr
        self.config['momentum'] = momentum
        self.config['is_nesterov'] = is_nesterov
        self.state.setdefault('accumulated_grads', {})

    def step(self, params, params_grad):
        var_index = 0  # Для каждого параметра будет своя ячейка с аккумулированными градиентами

        for current_layer_vars, current_layer_grads in zip(params, params_grad):
            for current_var, current_grad in zip(current_layer_vars, current_layer_grads):
                if current_var is not None:
                    # Так как словарь мог ещё не создаться, нужно это проверить. В целом можно это написать и просто
                    # через cur_epoch > 0, но не хочется портить параметры step
                    try:
                        self.state['accumulated_grads'][var_index]
                    except KeyError:
                        self.state['accumulated_grads'].setdefault(var_index, torch.zeros(current_grad.shape))

                    self.state['accumulated_grads'][var_index] = \
                        self.config['momentum'] * self.state['accumulated_grads'][var_index] + \
                        self.config['lr'] * current_grad
                    current_var -= self.state['accumulated_grads'][var_index]

                    var_index += 1


class Adam(Optimizer):
    """
    Алгоритм оптимизации Adam.

    Атрибуты
    --------
    `config` : dict
        Словарь c гиперпараметрами оптимизатора. Для Адама следующие параметры:
         `learning_rate`, `beta1` и `beta2` для экспоненциального сглаживания,
         `eps` для добавления в знаменатель при делении.
    `state` :  dict
        Словарь cо состоянием оптимизатора. Нужен, чтобы сохранять значения параметров m_t, v_t, а также степеней
        \beta_1, \beta_2 (обозначения как в оригинальной статье).
    """

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        super(Adam, self).__init__()
        self.config['lr'] = lr
        self.config['beta1'] = beta1
        self.config['beta2'] = beta2
        self.config['eps'] = eps
        self.state.setdefault('m_t', {})
        self.state.setdefault('v_t', {})

    def step(self, params, params_grad):
        var_index = 0  # Для каждого параметра будет своя ячейка с аккумулированными градиентами

        for current_layer_vars, current_layer_grads in zip(params, params_grad):
            for current_var, current_grad in zip(current_layer_vars, current_layer_grads):
                if current_var is not None:
                    # Так как словарь мог ещё не создаться, нужно это проверить.
                    try:
                        self.state['m_t'][var_index]
                    except KeyError:
                        self.state['m_t'].setdefault(var_index, torch.zeros(current_grad.shape))
                        self.state['v_t'].setdefault(var_index, torch.zeros(current_grad.shape))
                        self.state['beta_1_t'] = self.config['beta_1']
                        self.state['beta_2_t'] = self.config['beta_2']

                    self.state['m_t'][var_index] = self.config['beta_1'] * self.state['m_t'][var_index] + \
                                                   (1 - self.config['beta_1']) * current_grad
                    self.state['v_t'][var_index] = self.config['beta_2'] * self.state['v_t'][var_index] + \
                                                   (1 - self.config['beta_2']) * torch.square(current_grad)
                    widehat_m_t = self.state['m_t'][var_index] / (1 - self.state['beta_1_t'])
                    widehat_v_t = self.state['v_t'][var_index] / (1 - self.state['beta_2_t'])
                    current_var -= self.config['lr'] * torch.mul(widehat_m_t, 1. / (self.config['eps'] + widehat_v_t))

                    var_index += 1
                    self.state['beta_1_t'] *= self.config['beta_1']
                    self.state['beta_2_t'] *= self.config['beta_2']
