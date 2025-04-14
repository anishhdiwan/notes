---
layout: post
category: tutorial
title: Lagrange Multipliers & Lagrangian Duality
description: 
# image: /assets/reinforce_algo_sutton.png # URL to the post's featured image
---

### Lagrange Multipliers

We start by stating the goal of the optimisation problem we are interested in -- constrained (not necessarily convex) min or max problem. We wish to minimise or maximise a function $$f(x, y, z, ...)$$ of multiple variables such that the variables $$x, y, z, ...$$ are **_not_** independent. We capture their inter-dependence by a function $$g(x, y, z, ...)$$. 

$$
\begin{align*}
\min              &\; f(x, y, z, ...) \\
\text{s.t.} &\;
g(x, y, z, ...) = c \\
\end{align*}
$$

Given such a problem, the following statement holds true in the general case:

> At the optimal value, the level curve of the objective function is tangent to the level curve of the constraint function.

Given two functions that are tangent to each other, it follows that their normal vectors are parallel ([why?](#why-are-the-normals-parallel)). Meaning that the gradient of the objective function is parallel to the gradient of the constraint function (at the optimal value, in their respective level sets).

$$
\begin{align*}
\nabla f &\parallel \nabla g \\
\nabla f &= \lambda \nabla g
\end{align*}
$$

Where lambda (Lagrange multiplier) is some scalar that represents the relative scale of the two normal vectors. This means that our optimisation problem can now be rewritten as a system of equations following this condition that the gradients are parallel. Here is a [geometric illustration of this](#geometric-example-lagrange-multipliers) with a simple 2D problem. 

$$
\begin{align*}
&\nabla f = \lambda \nabla g \\
&g(x, y, z, ...) = c
\end{align*}
$$

Is there a solution for this in general? Not necessarily. But this provides us the intuition needed to then think of constraints in terms of Lagrange multipliers. 

### Lagrangian Duality

#### Problem Definition & Getting Rid Of The Constraints

Having seen the relationship between the optimisation and objective functions, we now look at the general setup of an optimisation problem in terms of its Lagrangian. Similarly to the previous case, we are interested in the following optimisation problem. i.e. we are trying to minimise some functions $$f$$ subject to a set of $$p$$ inequality constraints and a set constraint that the solutions lie in some set $$\mathcal{X} \subseteq \mathbb{R^d}$$. Let us call this problem $$(P)$$.

$$  
\begin{align*}
& \;\;\;(P) \\
\min              &\; f(x) \\
\text{s.t.} &\;
g_i(x) \leq 0 \text{ where } i \in \{ 1, 2, ..., p \}\\
&\; x \in \mathcal{X}
\end{align*}
$$

We define the Lagrangian of this function $$\mathcal{L}(x, \mu)$$ as a linear combination of $$f$$ and its constraints $$g_i$$ such that every constraint is accounted for using one Lagrange multiplier ($$\mu_i$$). 

$$\mathcal{L}_{x \in \mathcal{X} ; \mu \geq 0}(x, \mu) = f(x) + \sum_{i=1}^p \mu_i g_i(x)$$

In the general case, the following then holds true:

> Minimising $$(P)$$ is equivalent to minimising the supremum (bounded maximum) of the Lagrangian 

$$
\begin{equation}
\min_{x \in \mathcal{X}} \; \text{sup}_{\mu \geq 0} \; \{ f(x) + \sum_{i=1}^p \mu_i g_i(x) \}
\end{equation}
$$

To show this, let us consider a point $$x$$ in the feasible set (set where all constraints are satisfied) of this problem. Since $$x$$ is in the feasible set, $$g_i(x') \leq 0$$ for all constraints $$g_i$$. Given that all $$\mu_i$$ are non-negative, the maximum value $$\text{sup}_{\mu \geq 0} \{ \sum_{i=1}^p \mu_i g_i(x) \}$$ can take in the feasible set is $$0$$ and that is achieved when $$\mu_i = 0$$. For any other point not in the feasible set, at least one $$g_i$$ could be positive, meaning that $$\text{sup}_{\mu \geq 0} \{ \sum_{i=1}^p \mu_i g_i(x) \}$$ can be maximised to reach a value of $$\infty$$ my arbitrarily increasing the respective $$\mu_i$$. Hence, under the feasible set:

$$\text{sup}_{\mu \geq 0} \; \mathcal{L} (x, \mu) = \begin{cases}
  f(x) & \text{if } x \in \text{ feasible set} \\
  \infty & \text{if } x \notin \text{ feasible set}
\end{cases}
$$

From this it follows that minimising the original objective under the constraints (problem $$(P)$$) is equivalent to minimising the supremum of the Lagrangian of the objective.

$$\min_{x \in \mathcal{X}} \; \text{sup}_{\mu \geq 0} \; \mathcal{L} (x, \mu) = \min_{x \in \mathcal{X}} \; f(x) \text{ where } x \in \text{ feasible set}$$


#### Lagrangian Dual

Thus far we have just defined the problem and simplified it by removing the constraints. Now, let us take a moment to think of the "opposite" problem. Instead of minimising the supremum (bounded maximum) of our Lagrangian, what if we maximise the infimum (bounded minimum)? Are these two problems equivalent? Not necessarily, but thinking about the max of the min will give us some intuition on the solutions to our original problem. So, we define the Lagrangian dual function $$d(\mu)$$ as follows:

$$d(\mu) = \text{inf}_{x \in \mathcal{X}} \; \{ f(x) + \sum_{i=1}^p \mu_i g_i(x) \}$$

Notice that $$f(x) + \sum_{i=1}^p \mu_i g_i(x)$$ is just a set of affine functions in $$\mu_i$$. Meaning that for every $$x$$, $$f(x) + \sum_{i=1}^p \mu_i g_i(x)$$ gives us a set of lines. The infimum of these lines is just the problem of finding the right $$x$$ such that we get the lowest line. Another nice property (and the main reason for computing the dual function) is that being the infimum of a set of affine functions, $$d(\mu)$$ is **_always concave_**. This is a great finding since it allows us to take any potentially non-convex objective and convert it to a concave function (these functions are quite easy to optimise). Whether optimising this dual function is the same as optimising the original function is a separate question -- one that we will soon answer in the following paragraphs. 

Having defined the Lagrangian dual function, let us as discussed, try to define a problem that finds its maximum (previously in $$(P)$$ we tried to find the minimum of a supremum while now in the dual problem $$(D)$$ we are trying to find the maximum of the infimum). The Lagrangian dual problem $$(D)$$ (while the problem $$(P)$$ is called the primal) is defined as:

$$  
\begin{align*}
& \;\;\;(D) \\
&\max_{\mu \geq 0} \; \text{inf}_{x \in \mathcal{X}} \; L(x, \mu)
\end{align*}
$$

The dual problem is a concave maximisation problem (equivalent to convex minimisation) and is often easy to solve!

#### Mix-Max Inequality & Weak Duality Theorem

Having defined the dual and the primal, let's now look at the relationship between them. In general, for any function $$f(x,y): \mathcal{X} \times \mathcal{Y} \rightarrow \mathbb{R}$$ the following is always true:

$$\max_{x \in \mathcal{X}} \; \min_{y \in \mathcal{Y}} \; f(x,y) \leq \min_{y \in \mathcal{Y}} \; \max_{x \in \mathcal{X}} \; f(x,y)$$

The proof is rather simple (I will skip the formalism for this). For any function it is true that its min under one variable is less than its value under any other values of that variable. Now it must also be true that the max of this min is less than the max of the function under any values of that function. Which means that the max under the second variable of the original min must also be less than the min under the second variable of the max of the function. 

This property is also true for the Lagrangian (we swap in the supremum and infimum for correctness):

> $$d(\mu*) \triangleq \max_{\mu \geq 0} \; \text{inf}_{x \in \mathcal{X}} \; \mathcal{L}(x, \mu) \leq \min_{x \in \mathcal{X}} \; \text{sup}_{\mu \geq 0} \; \mathcal{L}(x, \mu) \triangleq f(x*)$$

Finally, this brings us to the weak duality theorem which states that:

> If $$x*$$ is the globally optimal solution of the minimisation problem $$(P)$$ and $$\mu*$$ is the globally optimal solution of the maximisation problem $$(D)$$ then $$d(\mu) \leq d(\mu*) \leq f(x*) \leq f(x)$$. 

Which means that the optimal solution to the dual is always less than or equal to the optimal solution to the primal! Hence, the dual's solution is a lower bound for the primal's solution. This is great because the dual is convex and typically is easier to solve! 

$$f(x*) - d(\mu*) = \text{ duality gap}$$ 

When $$f(x*) - d(\mu*) = 0$$, the problem is said to have strong duality and the primal can be directly solved by solving the dual! The duality gap is zero when certain conditions (Slater's conditions) are met. 

---

### Additional Clarifications

#### Geometric Example (Lagrange multipliers)

Consider the problem of finding the point closest to the origin on the hyperbola $$xy = 3$$. The distance to any point from the origin is $$x^2 + y^2$$ (we drop the sq-root for convenience). The optimisation problem is then to:

$$
\begin{align*}
& \min x^2 + y^2 \text{ (objective: f)} \\
& \text{s.t. } xy = 3 \text{ (constraint: g)}
\end{align*}
$$

<div align="center">
<iframe src="https://www.desmos.com/calculator/53c5tdyyjt?embed" width="500" height="500" style="border: 1px solid #ccc" frameborder=0></iframe>
</div>

The level curves of both the constraint and the objective function are plotted in the figure. Notice that the normals of both functions are parallel and their scaling difference is captured by the Lagrange multiplier. This means that $$\nabla f \parallel \nabla g$$. The problem then turns into a set of equations like so:

$$
\begin{align*}
\nabla f(x,y) &= \langle 2x , 2y \rangle = \langle y , x \rangle = \nabla g(x,y) \\
xy &= 3
\end{align*}
$$



#### Why are the normals parallel?   

Notice that at the minimum or maximum, at any direction along the level curve of the constraint ($$g = c$$), the rate of change of $$f$$ must be zero (that's what happens at min or max points). Formally, for a unit vector $$\hat{u}$$ tangent to the level set of the constraint function ($$g = c$$), it must be true that $$\frac{df}{ds \vert \hat{u}} = 0$$ (rate of change of $$f$$ evaluated at $$\hat{u}$$ is zero). 

$$\frac{df}{ds \vert \hat{u}} = \nabla f \cdot \hat{u} = 0$$ 

Which means that $$\hat{u} \perp \nabla f$$. But since $$\hat{u}$$ is tangent to $$g = c$$, $$\hat{u} \perp \nabla g$$. It follows then that $$\nabla f \parallel \nabla g$$.




#### Why is $$\nabla f \perp \{f = c\}$$ (level surface)[^2]?

Given a function $$f(x, y, z, ..)$$, we are interested in the level surface of the function. The level surface is simply where $$f = c$$ where c is some constant. We wish to prove that $$\nabla f$$ at any point on the level surface is perpendicular to the level surface. To show this, first consider some arbitrary parametric curve $$r(t) = \langle x(t), y(t), z(t) \rangle$$ which lies on the level surface (meaning that $$r(t)$$ is tangential to $$f = c$$). Following the chain rule, 

$$
\begin{align*}
\frac{df}{dt} &= \frac{\partial f}{\partial x} \frac{dx}{dt} + \frac{\partial f}{\partial y} \frac{dy}{dt} + \frac{\partial f}{\partial z} \frac{dz}{dt} \\
\frac{df}{dt} &= \nabla f \cdot \frac{dr}{dt} \\ \\

\text{where } & \frac{dr}{dt} = \langle \frac{dx}{dt}, \frac{dy}{dt}, \frac{dz}{dt}, \rangle
\end{align*}
$$

Observe that $$f(t) = c = \text{constant}$$, meaning that our function does not change with $$t$$ (because $$t$$ is a parameter on the level surface), and its gradient w.r.t $$t$$ is zero.

$$
\begin{align*}
\frac{df}{dt} &= 0 \\
\nabla f \cdot \frac{dr}{dt} &= 0
\end{align*}
$$

Hence $$\nabla f \perp \frac{dr}{dt}$$

So the gradient of f is perpendicular to the rate of change of some parametric function on the level surface. But since $$r(t)$$ is completely arbitrary, $$\nabla f$$ can be perpendicular to the rate of change of any parametric curve on the level surface. And since we assumed that $$r(t)$$ is tangential to the level surface, it implies that $$\nabla f$$ is perpendicular to any tangent to the level surface. Which means that its perpendicular to the level surface. 


#### References
This tutorial borrows several explanations from the wonderful MIT-OCW lecuture "[Lagrange multipliers: MIT 18.02 Multivariable Calculus, Fall 2007](https://www.youtube.com/watch?v=15HVevXRsBA&t=1195s)"

---
{: data-content="footnotes"}

[^2]: Lec 12: Gradient; directional derivative; tangent plane : MIT 18.02 Multivariable Calculus, Fall 07
