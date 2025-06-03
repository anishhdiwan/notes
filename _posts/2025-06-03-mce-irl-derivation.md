---
layout: post
category: tutorial
title: MCE-IRL Derivation
description: A derivation of the maximum causal entropy IRL results. 
# image: /assets/reinforce_algo_sutton.png # URL to the post's featured image
# featured: true
---

### MCE-IRL Optimization Problem

We seek a stochastic policy $$\{\pi(a_t\mid s_t)\}_{t=1}^{T-1}$$ given state‐marginal distributions $$\{\rho(s_t)\}_{t=1}^T$$ such that $$\pi$$ has the maximum **causal entropy** subject to matching empirical feature counts and the usual normalization/dynamics constraints. In expectation form, this reads:

$$
\begin{aligned}
\max_{\pi}\quad 
&\sum_{t=1}^{T-1} 
\mathbb{E}_{\substack{s_t \sim \rho(s_t)\\a_t \sim \pi(a_t\mid s_t)}} 
\bigl[-\,\ln \pi(a_t\mid s_t)\bigr] 
\quad\bigl(\text{causal entropy of }\pi\bigr)\\[6pt]
\text{s.t.}\quad
&\underbrace{
\sum_{t=1}^{T-1} 
\mathbb{E}_{s_t,a_t}\bigl[\phi(s_t,a_t)\bigr]
\;+\; 
\mathbb{E}_{s_T \sim \rho(s_T)}\bigl[\phi(s_T, \cdot)\bigr]
}_{\;\text{expected feature sum}}
\;=\;\widehat\phi,
\\[6pt]
&\rho(s_1) \;=\; \mu(s_1), 
\\[6pt]
&\rho(s_t) 
\;=\; 
\mathbb{E}_{\substack{s_{t-1}\sim\rho(s_{t-1})\\a_{t-1}\sim\pi(\,\cdot\mid s_{t-1})}}
\bigl[p\bigl(s_t\mid s_{t-1},\,a_{t-1}\bigr)\bigr]
\quad\text{for }t=2,\dots,T,
\\[6pt]
&\sum_{a_t} \pi(a_t\mid s_t) \;=\; 1 
\quad\text{for every }s_t,\;t=1,\dots,T-1.
\end{aligned}
$$

Here:

- $$\displaystyle \mathbb{E}_{s_t,a_t}[\cdot]$$ is shorthand for 
  $$\displaystyle \sum_{s_t}\rho(s_t)\sum_{a_t}\pi(a_t\mid s_t)\,[\,\cdot\,]$$
- $$\widehat\phi\in\mathbb{R}^d$$ is the vector of empirical (expert) feature counts:
  $$
  \widehat\phi \;=\; \frac{1}{N_D}\sum_{i=1}^{N_D} \sum_{t=1}^T \phi\bigl(s_t^{(i)},\,a_t^{(i)}\bigr)
  $$
- $$\mu(s_1)$$ is the known initial state distribution
- $$p(s_t\mid s_{t-1},a_{t-1})$$ are known transition probabilities
- Episodes are assumed to end at $$s_T$$ with the last action being $$a_{T-1}$$. At the final time $$T$$, the “action” is always $$a_T= \cdot$$, so $$\phi(s_T, \cdot)$$ depends only on $$s_T$$

Our approach to solving this optimization problem is using Lagrangian methods. The general procedure ([outlined here](https://anishhdiwan.github.io/notes/max-ent-irl.html)) is to first write out the Lagrangian of the problem, compute $$\pi$$ by setting its partial derivative to zero, then compute the lagrangian multiplies by gradient based optimization.

---

### Lagrangian

Introduce multipliers:

- $$\theta\in\mathbb{R}^d$$ to enforce feature matching
- $$\bigl\{V(s_t)\bigr\}_{t=1}^T$$ for the flow/dynamics constraints 

  $$\rho(s_t)=\mathbb{E}_{s_{t-1},a_{t-1}}[p(s_t\mid s_{t-1},a_{t-1})]$$
  
  (In particular, $$V(s_1)$$ enforces $$\rho(s_1)=\mu(s_1)$$)

- $$\bigl\{\lambda(s_t)\bigr\}_{t=1}^{T-1}$$ for the normalization constraints $$\sum_{a_t}\pi(a_t\mid s_t)=1$$

We then form the Lagrangian $$L\bigl(\{\rho,\pi\},\,\theta,\{V\},\{\lambda\}\bigr)$$.

$$
\begin{aligned}
L 
&=\underbrace{
\sum_{t=1}^{T-1} 
\mathbb{E}_{s_t,a_t}\bigl[-\,\ln \pi(a_t\mid s_t)\bigr]
}_{\text{(A) causal entropy}} 
\;+\;
\underbrace{
\theta^\top\Bigl[\,
\widehat\phi 
\;-\;\sum_{t=1}^{T-1}\mathbb{E}_{s_t,a_t}[\phi(s_t,a_t)]
\;-\;\mathbb{E}_{s_T}[\phi(s_T,0)]\Bigr]
}_{\text{(B) feature‐matching}} 
\\[6pt]
&\quad
+\;\underbrace{
\sum_{t=1}^{T-1}\sum_{s_t} 
\lambda(s_t)\Bigl[\,1 - \sum_{a_t} \pi(a_t\mid s_t)\Bigr]
}_{\text{(C) normalization of }\pi}
\\[6pt]
&\quad
+\;\underbrace{
\sum_{t=2}^T \sum_{s_t} 
V(s_t)\Bigl[
\underbrace{
\mathbb{E}_{s_{t-1},a_{t-1}}\bigl[p(s_t\mid s_{t-1},a_{t-1})\bigr]
}_{\substack{\text{(i) incoming}\\\text{flow to }s_t}}
\;-\;\rho(s_t)\Bigr]
}_{\text{(D) flow/dynamics for }t\ge2}
\\[6pt]
&\quad
+\;\underbrace{
\sum_{s_1} 
V(s_1)\,\bigl[\mu(s_1)\;-\;\rho(s_1)\bigr]
}_{\text{(E) initial‐state constraint}}
\end{aligned}
$$

#### To spell out each term:

1. **(A) Causal entropy term**  
   $$
   \sum_{t=1}^{T-1} \mathbb{E}_{s_t,a_t}\bigl[-\ln \pi(a_t\mid s_t)\bigr]
   \;=\;
   \sum_{t=1}^{T-1} \sum_{s_t}\rho(s_t)\sum_{a_t} \pi(a_t\mid s_t)\,\bigl[-\ln\pi(a_t\mid s_t)\bigr]
   $$

2. **(B) Feature‐matching term**  
   $$
   \begin{aligned}
   &\theta^\top\Bigl[\,
   \widehat\phi 
   \;-\;\sum_{t=1}^{T-1}\mathbb{E}_{s_t,a_t}[\phi(s_t,a_t)] 
   \;-\;\mathbb{E}_{s_T}[\phi(s_T,0)]\Bigr]
   \\[4pt]
   &=\;\theta^\top \widehat\phi 
   \;-\;\theta^\top 
   \sum_{t=1}^{T-1}\sum_{s_t}\rho(s_t)\sum_{a_t}\pi(a_t\mid s_t)\,\phi(s_t,a_t)
   \;-\;\theta^\top\sum_{s_T}\rho(s_T)\,\phi(s_T,0)
   \end{aligned}
   $$

3. **(C) Normalization of $$\pi$$**  
   $$
   \sum_{t=1}^{T-1}\sum_{s_t} \lambda(s_t)\Bigl[\,1 - \sum_{a_t}\pi(a_t\mid s_t)\Bigr]
   $$

   For each fixed $$t$$ and $$s_t$$, $$\sum_{a_t}\pi(a_t\mid s_t)=1$$ is required.  The term $$\lambda(s_t)\bigl(1 - \sum_{a_t}\pi(a_t\mid s_t)\bigr)$$ enforces that

4. **(D) Flow/dynamics for $$t\ge2$$**  
   $$
   \sum_{t=2}^T \sum_{s_t} V(s_t)\Bigl[\,
   \underbrace{\sum_{s_{t-1}}\rho(s_{t-1})\sum_{a_{t-1}}\pi(a_{t-1}\mid s_{t-1})\,p(s_t\mid s_{t-1},a_{t-1})}_{\mathbb{E}_{s_{t-1},a_{t-1}}[\,p(s_t\mid s_{t-1},a_{t-1})\,]}
   \;-\;\rho(s_t)\Bigr]
   $$
   Here $$V(s_t)$$ enforces
   
   $$\rho(s_t)\;=\;\sum_{s_{t-1},a_{t-1}}\rho(s_{t-1})\,\pi(a_{t-1}\mid s_{t-1})\,p(s_t\mid s_{t-1},\,a_{t-1})$$

5. **(E) Initial‐state constraint for $$t=1$$**  
   $$\sum_{s_1}V(s_1)\bigl[\mu(s_1)-\rho(s_1)\bigr]$$ enforces $$\rho(s_1)=\mu(s_1)$$

---

### Partial Derivative $$\displaystyle \frac{\partial L}{\partial \pi(a_t\mid s_t)}$$

We now compute, for a fixed $$t\in\{1,\dots,T-1\}$$ and fixed $$(s_t,a_t)$$, the partial derivative $$\partial L/\partial\pi(a_t\mid s_t)$$.  Since $$\rho(s_t)$$ is treated as constant when differentiating w.r.t. $$\pi$$, the only terms in $$L$$ that involve $$\pi(a_t\mid s_t)$$ are:

1. **Entropy term** (A)  
2. **Feature‐matching term** (part of (B) where $$t'=t$$)  
3. **Normalization term** (C) at time $$t$$ and state $$s_t$$  
4. **Flow term** (D), but only the piece at time $$t+1$$ (since $$\pi(a_t\mid s_t)$$ appears in $$\mathbb{E}_{s_t,a_t}[p(s_{t+1}\mid s_t,a_t)]$$).

We handle each contribution in turn


#### Contribution from Entropy $$\bigl[\mathbb{E}_{s_t,a_t}[-\ln \pi(a_t\mid s_t)]\bigr]$$


At time $$t$$: 

$$
E_{\mathrm{ent},t}
\;=\;\sum_{s_t}\rho(s_t)\sum_{a_t}\;\pi(a_t\mid s_t)\,\bigl[-\,\ln\pi(a_t\mid s_t)\bigr]
$$

Fix a particular pair $$(s_t,a_t)$$, the derivative of $$\mathbb{E}_{\mathrm{ent},t}$$ with respect to $$\pi(a_t\mid s_t)$$ is

$$
\frac{\partial E_{\mathrm{ent},t}}{\partial \pi(a_t\mid s_t)} = \frac{\partial}{\partial \pi(a_t\mid s_t)} \; \bigl( \rho(s_t)\;\pi(a_t\mid s_t)\,\bigl[-\,\ln\pi(a_t\mid s_t)\bigr] \bigr)
$$

$$
\boxed{
\frac{\partial E_{\mathrm{ent},t}}{\partial \pi(a_t\mid s_t)}
\;=\;
\rho(s_t)\,\bigl[-\,\ln\pi(a_t\mid s_t)\;-\;1\bigr]
}
$$


#### Contribution from Feature‐Matching $$\bigl[-\theta^\top\,\mathbb{E}_{s_t,a_t}[\phi(s_t,a_t)]\bigr]$$

Within (B), only the term at time $$t+1$$ depends on $$\pi(a_t\mid s_t)$$. Concretely,

$$
E_{\mathrm{feat},t}
\;=\;
-\,\theta^\top 
\sum_{s_t}\rho(s_t)\sum_{a_t}\pi(a_t\mid s_t)\,\phi(s_t,a_t).
$$

For a fixed $$(s_t,a_t)$$, this is a linear function in $$\pi(a_t\mid s_t)$$.  Hence

$$
\frac{\partial E_{\mathrm{feat},t}}{\partial \pi(a_t\mid s_t)}
\;=\;
-\,\theta^\top \bigl[\rho(s_t)\,\phi(s_t,a_t)\bigr]
$$

$$
\boxed{
\frac{\partial E_{\mathrm{feat},t}}{\partial \pi(a_t\mid s_t)}
\;=\;
-\,\rho(s_t)\,\bigl[\theta^\top \phi(s_t,a_t)\bigr].
}
$$


#### Contribution from Normalization $$\bigl[\sum_{s_t} \lambda(s_t)(\,1 - \sum_{a_t}\pi(a_t\mid s_t))\bigr]$$

At time $$t$$, the relevant piece is

$$
E_{\mathrm{norm},t}
\;=\;
\sum_{s_t} \lambda(s_t)\,\Bigl[\,1 \;-\;\sum_{a_t}\pi(a_t\mid s_t)\Bigr].
$$

Fixing $$s_t$$, the derivative of $$\bigl[\,1 - \sum_{a_t}\pi(a_t\mid s_t)\bigr]$$ w.r.t. $$\pi(a_t\mid s_t)$$ is $$-1$$.  Therefore, for that $$(s_t,a_t)$$,

$$
\boxed{
\frac{\partial E_{\mathrm{norm},t}}{\partial \pi(a_t\mid s_t)}
\;=\;
-\,\lambda(s_t).
}
$$


#### Contribution from Flow (Dynamics) at Time $$t+1$$

Within (D), the only term that depends on $$\pi(a_t\mid s_t)$$ is the “incoming flow” for $$s_{t+1}$$.

$$
E_{\mathrm{flow},\,t+1}
\;=\;
\sum_{s_{t+1}} 
V(s_{t+1})\,
\underbrace{\sum_{s_t}\rho(s_t)\sum_{a_t}\pi(a_t\mid s_t)\,p\bigl(s_{t+1}\mid s_t,a_t\bigr)
}_{\displaystyle 
\mathbb{E}_{s_t,a_t}[\,p(s_{t+1}\mid s_t,a_t)\,]
}
$$

We treat $$V(s_{t+1})$$ and $$\rho(s_t)$$, $$p(s_{t+1}\mid s_t,a_t)$$ as constants when differentiating w.r.t. $$\pi(a_t\mid s_t)$$.  Hence, at that fixed $$(s_t, a_t)$$


$$E_{\mathrm{flow},\,t+1}
\;=\;
\sum_{s_{t+1}} V(s_{t+1})\,\rho(s_t)\,\pi(a_t\mid s_t)\,p(s_{t+1}\mid s_t,a_t)
$$


$$
\frac{\partial E_{\mathrm{flow},\,t+1}}{\partial \pi(a_t\mid s_t)}
\;=\;
\sum_{s_{t+1}} 
V(s_{t+1}) \,\rho(s_t)\,p(s_{t+1}\mid s_t,a_t)
\;=\;\rho(s_t)\,
\underbrace{\sum_{s_{t+1}} p(s_{t+1}\mid s_t,a_t)\,V(s_{t+1})}_{\displaystyle \mathbb{E}_{s_{t+1}\sim p}[\,V(s_{t+1})\,]}.
$$


$$
\boxed{
\frac{\partial E_{\mathrm{flow},\,t+1}}{\partial \pi(a_t\mid s_t)}
\;=\;
\rho(s_t)\;\mathbb{E}_{s_{t+1}\sim p(\cdot\mid s_t,a_t)}\bigl[\,V(s_{t+1})\bigr].
}
$$


#### Summing All Derivatives

- Adding the partial derivatives, we get, for each $$t=1,\dots,T-1$$ and each $$(s_t,a_t)$$:

$$
\begin{aligned}
\frac{\partial L}{\partial \pi(a_t\mid s_t)}
&=
\underbrace{\rho(s_t)\bigl[-\ln\pi(a_t\mid s_t) - 1\bigr]}_{\text{(3.1)}}
\;+\;
\underbrace{\bigl[-\,\rho(s_t)\,\theta^\top\phi(s_t,a_t)\bigr]}_{\text{(3.2)}}
\;+\;
\underbrace{\bigl[-\,\lambda(s_t)\bigr]}_{\text{(3.3)}}
\;+\;
\underbrace{\rho(s_t)\,\mathbb{E}_{s_{t+1}}[\,V(s_{t+1})\,]}_{\text{(3.4)}} \\
&=
\rho(s_t)\Bigl[\,
-\ln\pi(a_t\mid s_t)\;-\;1\;-\;\theta^\top\phi(s_t,a_t)\;+\;\mathbb{E}_{s_{t+1}}[V(s_{t+1})]
\Bigr]
\;-\;\lambda(s_t)
\end{aligned}
$$

- Setting $$\frac{\partial L}{\partial \pi(a_t\mid s_t)} = 0$$ to compute the optimal policy,

$$
\rho(s_t)\Bigl[\,
-\ln\pi(a_t\mid s_t)\;-\;1\;-\;\theta^\top\phi(s_t,a_t)\;+\;\mathbb{E}_{s_{t+1}}[V(s_{t+1})]
\Bigr]
\;-\;\lambda(s_t)
\;=\;0
$$

- Assuming $$\rho(s_t)>0$$ (for states $$s_t$$ that are reachable):

$$
\begin{align}
&-\,\ln\pi(a_t\mid s_t)\;-\;1\;-\;\theta^\top\phi(s_t,a_t)\;+\;\mathbb{E}_{s_{t+1}}[V(s_{t+1})]
\;=\;\frac{\lambda(s_t)}{\rho(s_t)} \\

&-\,\ln\pi(a_t\mid s_t)\;-\;\theta^\top\phi(s_t,a_t)\;+\;\mathbb{E}_{s_{t+1}}[V(s_{t+1})]
\;=\;C_t(s_t)
\end{align}
$$

- or equivalently

$$
\ln\pi(a_t\mid s_t)
\;=\;
-\,\theta^\top\phi(s_t,a_t)
\;+\;\mathbb{E}_{s_{t+1}}[V(s_{t+1})]
\;-\;C_t(s_t)
$$

- Define the normalizer so that $$\pi(\cdot\mid s_t)$$ sums to 1.  In other words,

$$
\pi(a_t\mid s_t)
\;\propto\;
\exp\Bigl(\,-\,\theta^\top\phi(s_t,a_t)\;+\;\mathbb{E}_{s_{t+1}}[V(s_{t+1})]\Bigr).
$$

- Concretely, let

$$
Z_t(s_t)
\;=\;
\sum_{a'_t}\exp\Bigl(\,-\,\theta^\top\phi(s_t,a'_t)\;+\;\mathbb{E}_{s_{t+1}}[\,V(s_{t+1})\,]\Bigr).
$$

This results in the familiar **soft‐value** or “soft‐Bellman” policy:

$$
\boxed{
\begin{align}
\pi(a_t\mid s_t) &\;=\; 
\frac{
\exp\Bigl(\,-\,\theta^\top\phi(s_t,a_t)\;+\;\mathbb{E}_{s_{t+1}}[\,V(s_{t+1})\,]\Bigr)
}
{
\displaystyle 
Z_t(s_t)
} \\\\
&\;=\; 
\frac{
\exp\bigl(-\,\theta^\top\phi(s_t,a_t)\;+\;\mathbb{E}_{s_{t+1}}[\,V(s_{t+1})\,]\bigr)
}
{
\sum_{a'_t}
\exp\bigl(-\,\theta^\top\phi(s_t,a'_t)\;+\;\mathbb{E}_{s_{t+1}}[\,V(s_{t+1})\,]\bigr)
}
\end{align}
}
$$

---

### 4. Final Result

$$
\pi(a_t\mid s_t)
\;\propto\;
\exp\Bigl(\,-\,\theta^\top\phi(s_t,a_t)\;+\;\mathbb{E}_{s_{t+1}}[\,V(s_{t+1})\,]\Bigr)
$$

Further, the dynamics Lagrangian multiplier can be viewed as the soft value function of a state

$$
V(s_t)
\;=\;
\ln \sum_{a_t}
\exp\Bigl(\,-\,\theta^\top\phi(s_t,a_t)\;+\;\mathbb{E}_{s_{t+1}}[\,V(s_{t+1})\,]\Bigr)
$$

With 

$$\pi(a_t\mid s_t) = \exp\bigl(Q_t(s_t,a_t) - V(s_t)\bigr)$$

$$\;Q_t(s_t,a_t) = -\,\theta^\top\phi(s_t,a_t) + \mathbb{E}_{s_{t+1}}[\,V(s_{t+1})\,]$$

$$r(s_t, a_t) = -\,\theta^\top\phi(s_t,a_t)$$
