import torch


class Conv:
    """ Наивная реализация свёртки, медленная. """

    def __init__(self, nb_filters: int, filter_size: int, nb_channels: int,
                 init=None, stride: int = 1, padding: int = 0):
        """
        Конструктор свёрточного слоя.

        Параметры
        ---------
        nb_filters : int
            Число фильтров (каналов) на входе.
        filter_size : int
            Размер фильтра (ядра).
        nb_channels : int
            Число каналов на выходе слоя.
        init : tuple of torch.tensors, optional
            Содержит два элемента: тензор, которым проинициализировать матрицу
            весов W, и тензор, которым проинициализировать bias.
        stride : int
            Шаг, с которым берутся свёртки.
        padding : int
            Насколько нужно расширить входной тензор нулями.
        """

        self.num_filters = nb_filters
        self.f = filter_size
        self.n_C = nb_channels
        self.s = stride
        self.p = padding

        self.cache = None  # Для хранения данных прямого прохода сети,
        # которые потребуются при обратном проходе

        if init is not None:
            self.W = init[0]  # (n_C, n_C_prev, filter_size, filter_size)
            self.b = init[1]  # (n_C)
        else:
            self.W = torch.zeros((self.n_C, self.num_filters, self.f, self.f))
            self.b = torch.zeros((self.n_C,))
            self.W = torch.nn.init.uniform_(self.W)
            self.b = torch.nn.init.uniform_(self.b)

        self.dW = torch.zeros(self.W.shape)
        self.db = torch.zeros(self.b.shape)

    @staticmethod
    def single_conv(x_slice, W, b):
        """
        Выполняет скалярное произведение двух матриц.
        Нужна для операции свёртки.

        Параметры
        ---------
        x_slice : torch.tensor, shape = (n_C_prev, filter_size, filter_size)
            Часть входных данных, над которыми сейчас идёт свёртка
        W : torch.tensor, shape = (n_C_prev, filter_size, filter_size)
            Параметры свёртки
        b : torch.tensor, shape = (1, 1, 1)
            Параметр смещения для данного фильтра

        Возвращает
        ----------
        float
            Результат применения операции свёртки.
        """

        # Поэлементное произведение
        s = torch.mul(x_slice, W)
        # Сумма произведений
        g = torch.sum(s)
        # Сдвиг
        g = g + b
        return g

    @staticmethod
    def dilate(X, stride):
        """
        Вставляет в каждом фильтре после каждой строки/столбца
        stride-1 нулевых строк/столбцов. Нужна для backpropagation.

        Параметры
        ---------
        X : torch.tensor, shape = (n_C, H, W)
            Тензор для преобразования.
        stride : int
            Размер страйда, используемый в свёртке

        Возвращает
        ----------
        torch.tensor, shape = (n_C, new_H, new_W)
            Преобразованный тензор.
        """

        n_C, H, W = X.shape

        new_H, new_W = H + (H - 1) * (stride - 1), W + (W - 1) * (stride - 1)
        X_dilated = torch.zeros((n_C, new_H, new_W))

        for i in range(0, new_H, stride):
            for j in range(0, new_W, stride):
                X_dilated[:, i, j] = X[:, i // stride, j // stride]

        return X_dilated

    @staticmethod
    def rotate180(X):
        """
        Поворачивает тензор на 180 градусов с помощью двух симметричных
        отображений.

        Параметры
        ---------
        X : torch.tensor, shape = (H, W)
            Тензор для преобразования.

        Возвращает
        ----------
        torch.tensor
            Повёрнутый на 180 градусов тензор.
        """

        H, W = X.shape

        X_rotated = torch.zeros(X.shape)
        for i in range(H):
            for j in range(W):
                X_rotated[i, j] = X[H - 1 - i, W - 1 - j]

        return X_rotated

    def forward(self, X):
        """
        Прямой проход для свёрточного слоя.

        Параметры
        ---------
        X : torch.tensor, shape = (m, n_C_prev, n_H_prev, n_W_prev).
            Выход предыдущего свёрточного слоя.

        Возвращает
        ----------
        torch.tensor, shape = (m, n_C, n_H, n_W)
        """

        m, n_C_prev, n_H_prev, n_W_prev = X.shape

        # По формулам из доков Conv2d, dilation = 0
        n_H = int((n_H_prev + 2 * self.p - self.f) / self.s) + 1
        n_W = int((n_W_prev + 2 * self.p - self.f) / self.s) + 1

        Z = torch.zeros((m, self.n_C, n_H, n_W))
        # Так как ничего не указано, заполняем нулями
        X_padded = torch.nn.functional.pad(
            X, (self.p, self.p, self.p, self.p), mode='constant', value=0
        )

        for i in range(m):
            x_padded = X_padded[i]
            for h in range(n_H):
                for w in range(n_W):
                    # Вычислим положение текущего окна
                    top_left_h = h * self.s
                    top_left_w = w * self.s
                    bottom_right_h = top_left_h + self.f
                    bottom_right_w = top_left_w + self.f

                    # Текущее окно
                    window = x_padded[:, top_left_h:bottom_right_h,
                             top_left_w:bottom_right_w]
                    for c in range(self.n_C):
                        # Применим свёртку
                        Z[i, c, h, w] = Conv.single_conv(
                            window, self.W[c], self.b[c]
                        )

        # Сохраняем то, что пригодится для обратного прохода
        self.cache = {'input': X, 'output': Z}

        return Z

    def backward(self, dZ):
        """
        Распространяет градиент ошибки от предыдущего слоя в текущий
        свёрточный слой.

        Параметры
        ---------
        dZ : torch.tensor, shape = (m, n_C, n_H, n_W)
            Градиент, пришедший от следующего слоя.

        Возвращает
        ----------
        dX : torch.tensor, shape = (m, n_C_prev, n_H_prev, n_W_prev)
            Ошибка текущего свёрточного слоя.
        self.dW : torch.tensor, shape = (n_C, n_C_prev, filter_size, filter_size)
            Градиент по весам.
        self.db : torch.tensor, shape = (n_C)
            Градиент по сдвигам.
        """

        m, n_C_prev, n_H_prev, n_W_prev = self.cache['input'].shape
        m, n_C, n_H, n_W = self.cache['output'].shape

        # Форма пришедшего градиента должна совпадать с формой выхода слоя
        assert dZ.shape == self.cache['output'].shape

        dX = torch.zeros(
            (m, n_C_prev, n_H_prev + 2 * self.p, n_W_prev + 2 * self.p)
        )

        # Сначала посчитаем градиент по входу
        W_rotated = self.W.detach().clone()
        for c in range(n_C):
            for old_c in range(n_C_prev):
                W_rotated[c, old_c] = Conv.rotate180(self.W[c, old_c])

        for i in range(m):
            dx = dX[i]
            # Как показывается в последней статье, нужно добавить
            # padding размера filter_size - 1 со всех сторон, а также
            # расширить каждую строку/столбец на stride - 1
            dz_transformed = torch.nn.functional.pad(
                Conv.dilate(dZ[i], self.s),
                (self.f - 1, self.f - 1, self.f - 1, self.f - 1),
                mode='constant', value=0.
            )
            # Бывают случаи, когда какие-то столбцы/строки входа
            # не участвуют вообще. В таком случае градиенты по ним остаются
            # нулевыми. Так что градиенты нужны не по всем элементам входа.
            # Но формулу выводить сложно, поэтому будем отслеживать такие
            # моменты находу.
            for h in range(dX.shape[2]):
                for w in range(dX.shape[3]):
                    window = dz_transformed[:, h: h + self.f, w: w + self.f]
                    for c in range(n_C_prev):
                        if window.shape != W_rotated[:, c, :, :].shape:
                            continue
                        dx[c, h, w] = Conv.single_conv(
                            window, W_rotated[:, c, :, :], 0.
                        )
        # Так как мы могли делать padding при forward pass, нужно сейчас
        # вернуть к прежним размерам градиент
        if self.p > 0:
            dX = dX[:, :, self.p:-self.p, self.p:-self.p]
        assert dX.shape == self.cache['input'].shape

        # Теперь посчитаем градиенты по параметрам
        # layer_input_padded = torch.nn.functional.pad(
        #     self.cache['input'], (self.p, self.p, self.p, self.p),
        #     mode='constant', value=0
        # )
        X_padded = torch.nn.functional.pad(
            self.cache['input'], (self.p, self.p, self.p, self.p),
            mode='constant', value=0.
        )
        for i in range(m):
            x = X_padded[i]
            # (n_C, dilated_n_H, dilated_n_W)
            dz_transformed = Conv.dilate(dZ[i], self.s)
            window_h, window_w = dz_transformed[0].shape
            for h in range(self.f):
                for w in range(self.f):
                    # Выбираем конкретный двумерный фильтр
                    for c in range(n_C_prev):
                        window = x[c, h:h + window_h, w:w + window_w]
                        for new_c in range(n_C):
                            # Теперь по формуле из ссылки на вторую часть
                            # последней статьи посчитаем значение каждого
                            # элемента в двумерном фильтре
                            self.dW[new_c, c, h, w] += Conv.single_conv(
                                window, dz_transformed[new_c], 0.
                            )

        self.db = dZ.sum(dim=(0, 2, 3))  # Оставляем размерность каналов

        return dX, self.dW, self.db


class AvgPool:
    def __init__(self, filter_size, stride=1, padding=0):
        self.f = filter_size
        self.p = padding
        self.s = stride
        self.cache = {}

    def forward(self, X):
        """
        Прямой проход для усредняющего пулинга.

        Аргументы
        ---------
        X : torch.tensor, shape = (m, n_C_prev, n_H_prev, n_W_prev)

        Возвращает
        ----------
        torch.tensor, shape = (m, n_C_prev, n_H, n_W)
        """

        m, n_C_prev, n_H_prev, n_W_prev = self.cache['input_shape'] = X.shape

        # Посчитаем новый размеры по формуле из доков AvgPool2d
        n_H = int(1 + (n_H_prev + 2 * self.p - self.f) / self.s)
        n_W = int(1 + (n_W_prev + 2 * self.p - self.f) / self.s)

        Z = torch.zeros((m, n_C_prev, n_H, n_W))
        X_padded = torch.nn.functional.pad(
            X, (self.p, self.p, self.p, self.p), mode='constant', value=0.
        )

        for i in range(m):
            cur_batch_elem = X_padded[i]
            for h in range(n_H):
                for w in range(n_W):
                    # Определим границы текущего окна
                    top_left_h = h * self.s
                    top_left_w = w * self.s
                    bottom_right_h = top_left_h + self.f
                    bottom_right_w = top_left_w + self.f

                    for c in range(n_C_prev):
                        window = cur_batch_elem[c, top_left_h:bottom_right_h,
                                 top_left_w:bottom_right_w]
                        Z[i, c, h, w] = torch.mean(window)

        return Z

    def backward(self, dZ):
        """
        Обратный проход.

        Параметры
        ---------
        dZ : torch.tensor, shape = (m, n_C, n_H, n_W)
            Градиент лосса по выходу авгпулинга.

        Возвращает
        ----------
        dX : torch.tensor, shape = (m, n_C, n_H_prev, n_W_prev)
            Градиент лосса по входу авгпулинга.
        """

        m, n_C, n_H_prev, n_W_prev = self.cache['input_shape']
        dX = torch.zeros((m, n_C, n_H_prev + 2 * self.p, n_W_prev + 2 * self.p))
        W = torch.ones((n_C, self.f, self.f)) / (self.f ** 2)

        for i in range(m):
            cur_grad_dilated = torch.nn.functional.pad(
                Conv.dilate(dZ[i], self.s),
                (self.f - 1, self.f - 1, self.f - 1, self.f - 1),
                mode='constant', value=0.
            )
            for c in range(n_C):
                # Бывают случаи, когда какие-то столбцы/строки входа
                # не участвуют вообще. В таком случае градиенты по ним остаются
                # нулевыми. Так что градиенты нужны не по всем элементам входа.
                # Но формулу выводить сложно, поэтому будем отслеживать такие
                # моменты находу.
                for h in range(dX.shape[2]):
                    for w in range(dX.shape[3]):
                        window = cur_grad_dilated[c, h:h + self.f, w:w + self.f]
                        if window.shape != W[c].shape:
                            continue

                        dX[i, c, h, w] = Conv.single_conv(
                            window, W[c], 0.
                        )

        # Так как мы могли делать padding при forward pass, нужно сейчас
        # вернуть к прежним размерам градиент
        if self.p > 0:
            dX = dX[:, :, self.p:-self.p, self.p:-self.p]
        assert dX.shape == self.cache['input_shape']

        return dX
