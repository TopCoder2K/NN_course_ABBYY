Пояснение по слоям
1) Линейный слой.
   ![img.png](LinearLayer.png)
   
2) SoftMax слой.
   ![img.png](SoftMax1.png)
   ![img.png](SoftMax2.png)


Пояснение по функциям активации.
1) ReLU.
   ![img.png](ReLU.png)
   
2) Sigmoid.
   ![img.png](Sigmoid.png)
   
Пояснение по лоссам.
1) MSE loss.
   ![img.png](MSE.png)
   
2) Cross-entropy loss (NLL loss).
   ![img.png](CrossEntropy.png)
   
3) Kullback-Leibler divergence loss.
   Отличается от Cross-entropy loss только лишь на энтропию истинного распределения.
   Градиент остаётся таким же.
   
Замечания: 1) убрал apply_grad, потому что лучше передавать параметры
2) убрал step в Model, потому что он не нужен
3) Выделил отдельно forward, так понятнее и по pytorch
4) optimizer внутри слоя точно не нужен
