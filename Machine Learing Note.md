### Entropy

1. Information entropy
   $$
   H(X) = -\sum_{x} p(x) \log p(x)
   $$

2. Joint entropy
   $$
   H(X, Y) = - \sum_{x, y} p(x, y) \log p(x, y)
   $$

3. Conditional entropy
   $$
   H(Y|X) =\sum_x p(x) H(Y|X=x) \\
    = -\sum_x p(x) \sum_y p(y|x) \log p(y|x) \\
    = - \sum_{x,y} p(x, y) \log p(y|x)
   $$
   $H(Y|X) = H(X,Y) - H(X)$

4. Relative entropy, **Kullback-Leibler divergence**:
   $$
   D_{KL} (p||q) =\sum_x p(x) \log \frac {p(x)} {q(x)}
   $$

5. Cross entropy
   $$
   H(p,q) = -\sum_x p(x) \log q(x)
   $$

   $$
   D_{KL} (p||q) = H(p, q) - H(p)
   $$

   当$H(p)$为常量时（在机器学习中，训练数据的分布是固定的），最小化相对熵$D_{KL}(p||q)$等价于最小化交叉熵$H(p,q)$，也等价于最大化似然估计（可以参考Deep Learning 5.5）。

### Linear Algebra

1. Eigenvector and eigenvalue
   $$
   A \overrightarrow x = \lambda \overrightarrow x
   $$

2. Positive definite

### Probability

1. Conditional probability

   $p(x|y)=\frac{p(x, y)}{p(y)}$

2. Independent

   $x \perp y$: $x$ and $y$ are independent, $p(X=x, Y=y) = p(X=x)p(Y=y)$, and $p(x|y)=p(x)$.

   $x \perp y | z$: $x$ and $y$ are **conditionally independent** given $z$, i.e., $p(x, y|z)=p(x|z)p(y|z)$. Then, $p(x|y,z)=\frac{p(x,y|z)}{p(y|z)}=p(x|z)$.

3. Chain Rule
   $$
   P(x^{(1)},\dotsc,x^{(n)}) = P(x^{(1)}) \prod_{i=2}^{n} P(x^{(i)}|x^{(1)},\dotsc,x^{(i-1)})
   $$

4. Expectation
   $$
   \mathbb{E}_{x \sim P}[f(x)] = \sum_{x} p(x) f(x)
   $$
   If it is clear which random variable the expectation is over, we may omit the subscript entirely, as in $\mathbb{E}[f(x)]$.

   1. Conditional Expectation
      $$
      \mathbb{E}_{x \in P(X|Y=y)}[f(x)|Y=y] = \sum_{x} p(x|y) f(x)
      $$
   
   2. Iterated Expectation
      $$
      E[E[X|Y]] = E[X]
      $$
      
   
5. Variance
   $$
   \mathrm{Var}(f(x)) = \mathbb{E}[(f(x) - \mathbb{E}[f(x)])^2]
   $$

   1. Variance Decomposition
      $$
      Var(X) = Var(E[X|Y]) + E[Var(X|Y)]
      $$
      
   
6. Covariance
   $$
   \mathrm{Cov} (f(x), g(y)) = \mathbb{E}[(f(x) - \mathbb{E} [f(x)]) (g(x) - \mathbb{E} [g(y)])]
   $$
   If there are $n$ possible realizations of $(f(x), g(y))$, namely $(f(x_i), g(y_i))$ with probabilities $p_i$ for $i=1,\dotsc,n$, then the covariance is
   $$
   \mathrm{Cov}(f(x), g(y)) = \sum_{i=1}^{n} p_i (f(x_i) - \mathbb{E}[f(x)]) (g(y_i) - \mathbb{E}[g(y)]) )
   $$

7. Common Probability Distributions

   1. Bernoulli Distribution

   2. Binomial Distribution

      $B(n, p)$, where $n \in \{0, 1, 2, \dotsc\}$  is the number of trials, $p \in [0, 1]$ is the success probability for each trial.

   3. Multinomial Distribution

   4. Multinoulli Distribution

   5. Gaussian Distribution

   6. Exponential and Laplace Distributions

   7. Dirac Distribution and empirical Distribution

   8. Mixtures of Distributions

      1. Gaussian mixture model

8. Bayes' Rule
   $$
   P(x|y) = \frac {P(x)P(y|x)} {P(y)}
   $$
   $P(y)=\sum_x P(y|x)P(x)$

   