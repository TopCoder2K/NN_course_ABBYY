import torch

from nn.base import Optimizer


class GradientDescend(Optimizer):
    """
    Градиентый спуск с возможностью использования Momentum и моментов Нестерова.

    Атрибуты
    --------
    `config` : dict
        Словарь c гиперпараметрами оптимизатора.
    `state` : dict
        Словарь cо состоянием оптимизатора. Нужен, чтобы сохранять старые значения градиентов.
    """

    def __init__(self, lr=0.01, l1=None, l2=None, momentum=0., is_nesterov=False):
        """
        Параметры
        ---------
        `lr` : float
            Шаг обучения.
        `l1` : float
            Параметр L1-регуляризации.
        `l2` : float
            Параметр L2-регуляризации.
        `momentum` : float
            Параметр для метода 'Momentum'
        `is_nesterov` : boolean
            Использовать ли момент Нестерова?
        """

        super(GradientDescend, self).__init__(lr, l1, l2)
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

                    # Добавляем регуляризацию, если нужно
                    self._add_regularization_grad(current_var, current_grad)
                    # Сохраняем старую скорость для Нестерова
                    velocity_prev = self.state['accumulated_grads'][var_index]

                    self.state['accumulated_grads'][var_index] = \
                        self.config['momentum'] * self.state['accumulated_grads'][var_index] - \
                        self.config['lr'] * current_grad
                    current_var += self.state['accumulated_grads'][var_index]

                    if self.config['is_nesterov']:
                        current_var += self.config['momentum'] * \
                                       (-velocity_prev + self.state['accumulated_grads'][var_index])

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
        self.state.setdefault('beta_1_t', {})
        self.state.setdefault('beta_2_t', {})

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
                        self.state['beta_1_t'].setdefault(var_index, self.config['beta_1'])
                        self.state['beta_2_t'].setdefault(var_index, self.config['beta_2'])

                    self.state['m_t'][var_index] = self.config['beta_1'] * self.state['m_t'][var_index] + \
                                                   (1 - self.config['beta_1']) * current_grad
                    self.state['v_t'][var_index] = self.config['beta_2'] * self.state['v_t'][var_index] + \
                                                   (1 - self.config['beta_2']) * torch.square(current_grad)
                    widehat_m_t = self.state['m_t'][var_index] / (1 - self.state['beta_1_t'][var_index])
                    widehat_v_t = self.state['v_t'][var_index] / (1 - self.state['beta_2_t'][var_index])
                    current_var -= self.config['lr'] * torch.div(widehat_m_t, self.config['eps'] + widehat_v_t.sqrt())

                    self.state['beta_1_t'][var_index] *= self.config['beta_1']
                    self.state['beta_2_t'][var_index] *= self.config['beta_2']
                    var_index += 1
