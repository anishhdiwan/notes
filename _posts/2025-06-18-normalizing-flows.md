---
layout: post
category: unpublished
title: Normalizing Flows & Flow Matching
description: A tutorial on normalizing flows.
# image: /assets/reinforce_algo_sutton.png # URL to the post's featured image
# featured: true
---

Normalizing Flows are yet another family of likelihood based generative models. Like in most generative modelling problems, we wish to approximate some unknown data distribution given samples drawn from it. The neat things about flow-based models is that they (i) provide a tractable way of sampling from the approximated distribution and (ii) approximate actual normalized distributions (i.e. the probabilities sum to 1 and it is tractable to compute the likelihood of any given sample). [^1]

Flow based models essentially learn mappings from simple (tractable) distributions to complex ones (distributions are difficult to express analytically). The normalizing flow is defined as the series of functions that maps the simple distribution to the complex one. Before diving into generative modelling, it is first important to understand the basic rule that enables this transformation.

### Change of Variables
Say you are given a random variable $$z \in \mathbb{R}^n$$ sampled from a simple, tractable distribution $$p_z(z)$$ (for example, the unit Gaussian $$\mathcal{N}(0, 1)$$). Say there exists a function $$f: \mathbb{R}^n \rightarrow \mathbb{R}^n$$ that maps $$z$$ to a different random variable $$x$$ distributed as per $$p_x(x)$$. Given some $$z$$, what then is the probability of the $$x$$ obtained by the mapping $$f(z)$$? In other words, can we obtain the density $$p_x$$ given $$p_z$$ and a function mapping the samples of the $$p_z$$ to the samples of $$p_x$$?

Assuming $$f$$ is invertible and bijective, then the change of variables rule states that ([proof](#proofs)):

$$
\begin{align*}
p_x(x) &= p_z(z) \left \lvert \text{det} \left( \frac{\partial f(z)}{\partial z} \right) \right \rvert^{-1} \\
&= p_z(f^{-1}(x)) \left \lvert \text{det} \left( \frac{\partial f^{-1}(x)}{\partial x} \right) \right \rvert
\end{align*}
$$

Here, $$\left \lvert \text{det} \left( \frac{\partial f(z)}{\partial z} \right) \right \rvert^{-1}$$ scales the probability of sample $$x$$ to neutralise any volume changes to $$p_x$$ caused by the appication of $$f$$. 


### Proofs

---
{: data-content="footnotes"}

[^1]: Throughout this page, I use the term distribution but in reality these are "densities" in continuous spaces.