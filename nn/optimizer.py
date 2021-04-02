from nn.base import Optimizer


class GradientDescend(Optimizer):
    """
    Градиентый спуск с возможностью различных модификаций.

    Атрибуты
    --------
    `config` : dict
        Словарь c гиперпараметрами оптимизатора. Например, `learning_rate` и `momentum` для SGD с Momentum.
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
        var_index = 0
        for current_layer_vars, current_layer_grads in zip(variables, gradients):
            for current_var, current_grad in zip(current_layer_vars, current_layer_grads):
                state['accumulated_grads'].setdefault(var_index, np.zeros_like(current_grad))
                state['accumulated_grads'][var_index] = \
                    config['momentum'] * state['accumulated_grads'][var_index] + \
                    config['learning_rate'] * current_grad

                current_var -= state['accumulated_grads'][var_index]

                var_index += 1
