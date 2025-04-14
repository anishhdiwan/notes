---
layout: post
category: tutorial
title: Maximum [Causal] Entropy Inverse Reinforcement Learning
description: For several of the tasks to which we hope to apply reinforcement learning (complex and high speed robotic locomotion/manipulation, autonomous driving, multi-agent strategy games, long-horizon planning), the reward function that one must optimise is far too convoluted for humans to even conjure. How then can we teach machines these tasks? The key here is that humans CAN in-fact solve these complex tasks that we wish machines to be able to solve. Can we then just demonstrate the tasks to the machines? Inverse reinforcement learning (IRL) is a paradigm where the goal is to infer the reward function being implicitly maximised by using demonstrations provided by an expert (typically a human). This post dives into IRL frameworks called Maximum Entropy IRL (MaxEntIRL) and Maximum Causal Entropy IRL (MaxCausalEntIRL).
# image: /assets/reinforce_algo_sutton.png # URL to the post's featured image
---

Reinforcement learning (RL) is a machine learning problem where an agent tries learn a policy by maximising the rewards (scalar) it receives on interacting with its environment. While reinforcement learning has shown great success in a wide array of decision-making problems, it still has one key limitation. When applying RL, the problem designer is typically assumed to know the reward function that the agent must maximise. While this seems like a trivial, preparatory step in applying RL, designing an expressive and informative reward function is expressly non-trivial. For several of the tasks to which we hope to apply reinforcement learning (complex and high speed robotic locomotion/manipulation, autonomous driving, multi-agent strategy games, long-horizon planning), the reward function that one must optimise is far too convoluted for humans to even conjure. How then can we teach machines these tasks? The key here is that humans CAN in-fact solve these complex tasks that we wish machines to be able to solve. Can we then just demonstrate the tasks to the machines? 

Inverse reinforcement learning (IRL) is a paradigm where the goal is to infer the reward function being implicitly maximised by using demonstrations provided by an expert (typically a human). This post will skip a lot of the necessary [background](https://anishhdiwan.github.io//assets/pdf/msc_lit_review.pdf) and dive straight into a framework Maximum Causal Entropy IRL (MaxCausalEntIRL). We are interested in the IRL problem where the expert is assumed to have operated in an MDP with some unknown policy $$\pi_E$$. $$S$$, $$A$$, $$T(s_{t+}, s_t, a_t) = P(s_{t+1} \vert s_t, a_t)$$, $$R(s_t, a_t, s_{t+1})$$, and $$\gamma \in [0,1]$$ are the set of states, set of actions, transition dynamics model, reward function (unknown to the imitator), and discount factor. The states $$s \in S$$ are described by features $$\phi(s)$$ ($$\phi: S \rightarrow \mathbb{R}^d$$) and we say that the features of a trajectory are just the sum of the features of the individual states encountered in the trajectory ($$\phi(\tau) = \sum_{t=1}^{T}\phi(s_t)$$).

While $$\pi_E$$ is unknown to the imitator, it has access to a dataset of the expert's trajectories $$\mathcal{D} = \{ \tau_i \}_{1}^{N}$$ where $$N$$ is the number of trajectories provided and $$\tau_i$$ is a trajectory like so: $$\{ (s_0, a_0), (s_1, a_1), ..., (s_T, a_T) \}$$. Given this, the imitator aims to first recover the reward function $$R$$ and then learn a policy $$\pi_L$$ ($$L$$  for learner) that closely resembles $$\pi_E$$.

### Maximum Causal Entropy IRL: Intuition

What does it mean for $$\pi_L$$ to closely resemble $$\pi_E$$? Here, we interpret imitation as having similar occupancy over the MDP's states. Meaning that our recovered reward function must be such that the expected (fancy term for probabilistic average) visitation-frequency of states under the learnt policy $$\pi_L$$ must match the expected visitation-frequency of the states under the expert's policy. In simple words, the reward function is deemed sufficient if upon maximising the learnt reward, the learner visits the same states/actions that the expert did (on average). Mathematically, we can write this as:

$$\mathbb{E}_{\pi_L} [\phi(\tau)] = \mathbb{E}_{\pi_E} [\phi(\tau)]$$

Unfortunately, the IRL problem is ill-posed. A policy (say the expert's policy $$\pi_E$$) can be optimal under multiple reward functions. Conversely, multiple policies can be optimal under the same reward function [Ng, et. al. (1999), Ziebart, et al. (2008) ,Ziebart, et al. (2010)]. Given the current problem formulation, we could end up with multiple solutions for our reward function. Some of these might be trivial (such as all zero rewards). However, there might also be several equally good-looking reward functions that might otherwise be very difficult to contradistinguish. The principle of maximum entropy [Jaynes, (1957)] suggests that among several distributions (over trajectories) that satisfy the feature expectation matching criterion, it is wise to choose one with the maximum entropy <sup>[(what's entropy)](#entropy)</sup>. A distribution with the highest relative entropy is also one with the least information about the system. In other words, such a learnt policy would be the _least commital_ in its approach to decision-making. Meaning that such a policy would be more open to equally-viable alternative decisions (instilling a certain amount of multi-modality and flexibility to its decisions). Said formally, if state $$s$$ never appears in the expert's demonstrations, then the learnt policy $$\pi_L(a \vert s)$$ must be equiprobable (a low-entropy policy would instead inexplicably prefer a few actions over others). 

But is it sensible to just maximise the (Shannon) entropy $$H(s_{0:T-1}, a_{0:T-1})$$ even in stochastic settings? $$H(s_{0:T-1}, a_{0:T-1})$$ is the entropy over the complete trajectory distribution (in the joint distribution of states and actions). Which means the stochasticity of state transitions in the MDP contributes to a portion of $$H(s_{0:T-1}, a_{0:T-1})$$. Only maximising the entropy would mean that the policy would inherently prefer actions that take the agent to states with high uncertainty. In contrast, our objective with entropy maximisation was to simply prefer actions with higher uncertainty while having no effect over the states that the agent visits. The difference is subtle but significant[^1]. 

$$
\begin{align*}
H(s_{0:T-1}, a_{0:T-1}) &= \sum_{t=1}^{T-1} H(s_t, a_t \vert s_{0:t-1}, a_{0:t-1}) \\
&= \sum_{t=1}^{T-1} H(s_t \vert s_{0:t-1}, a_{0:t-1}) + \sum_{t=1}^{T-1} H(a_t \vert s_{0:t}, a_{0:t-1}) \\
&= \underbrace{\sum_{t=1}^{T-1} H(s_t \vert s_{t-1}, a_{t-1})}_{\text{transition dynamics contribution}} +  \underbrace{\sum_{t=1}^{T-1} H(a_t \vert s_t)}_{\text{policy contribution}}
\end{align*}
$$

Here, the first term is the entropy contribution of the transition dynamics and the second term is the entropy contribution of the policy. We only want to maximise the second term. This is called the causal entropy and is defined as:

$$H(a_{0:T-1} \lvert \rvert s_{0:T-1}) = \sum_{t=1}^{T-1} H(a_t \vert s_t)$$

Adding in the maximum causal entropy condition, we obtain a constrained optimisation problem:

$$
\begin{align*}
\arg \max_{\pi_L}              &\; H(a_{0:T-1} \lvert \rvert s_{0:T-1}) \\
\text{s.t.} &\;
\mathbb{E}_{\pi_L} [\phi(\tau)] = \mathbb{E}_{\pi_E} [\phi(\tau)] \\
& \sum_{a} \pi_L(a \vert s_t)                               = 1 \\
& \forall a_t, s_t, \; \pi_L(a_t \vert s_t)                            \geq 0
\end{align*}
$$

### MaxCausalEnt IRL: Optimisation Walkthrough

In this section, we will walk through the procedure to solve the optimisation problem outlined above. This is not the complete proof (I skip parts where the expressions are expanded, rearranged, and reduced) but a functional explanation of the procedure <sup>[(full proof)](https://arxiv.org/pdf/2203.11409)</sup> . The general procedure ([optimisation w. Lagrangian duality](https://anishhdiwan.github.io/notes/lagrange-multipliers.html)) is to:

- Formulate the Lagrangian for the problem (such that that maximising the constrained optimisation problem is equivalent to maximising the min of the Lagrangian).
- Compute the dual optimisation problem (such that $$\min \max \text{dual problem}$$ is at least an upper bound for $$\max \min \text{Lagrangian}$$ -- in this case there is strong duality so instead $$\min \max \text{dual problem} \Leftrightarrow \max \min \text{Lagrangian}$$).
- Solve the dual problem by setting $$\nabla \text{Lagrangian}$$ and $$\nabla \text{dual}$$ to zero. We will also see how reinforcement learning patterns emerge from this procedure and how dual ascent can be seen as RL + Reward Learning.

The Lagrangian is defined by including the constraints into the objective and weighting them by a Lagrange multiplier.

$$
\begin{align*}
\mathcal{L}(\pi_L, \theta, \mu) = \;\; &H(a_{0:T-1} \lvert \rvert s_{0:T-1}) \\
&+ \theta^\intercal \left[\mathbb{E}_{\pi_L} \left[\sum_{t=1}^{T} \gamma^t \phi(s_t, a_t) \right] - \mathbb{E}_{\pi_E} \left[ \sum_{t=1}^{T} \gamma^t \phi(s_t, a_t) \right] \right] \\
&+ \mu  \sum_{s_t \in S, 0 \leq t < T} \sum_{a} \pi_L(a \vert s_t) - 1
\end{align*}
$$ 

A few things to note here are that **(1)** the policy being a probability distribution is constrained to the set $$\Pi$$ of all valid probability distributions. i.e. normalisation: $$\sum_{a} \pi_L(a \vert s_t) = 1$$ and non-negativity: $$\pi_L(a_t \vert s_t) \geq 0$$. **(2)** the Lagrange multiplier $$\theta$$ can be interpreted as a set of weights on the reward function that impose the feature expectation matching constraint: since $$r(s_t, a_t) = \text{some function of } \phi(s_t, a_t)$$. This also implies that the reward function is linear in trajectory features.

Having defined the Lagrangian, we now know that the original problem is equivalent to $$\max_{\pi_L \in \Pi} \; \min_{\theta \in \mathbb{R}^d} \; \mathcal{L}(\pi_L, \theta, \mu)$$

Because of strong duality (proof in [Ziebart, et al. (2010)]), the primal problem is equivalent to its dual:

$$(P) \triangleq \max_{\pi_L \in \Pi} \; \min_{\theta \in \mathbb{R}^d} \; \mathcal{L}(\pi_L, \theta, \mu) \Leftrightarrow \min_{\theta \in \mathbb{R}^d} \; \max_{\pi_L \in \Pi} \; \mathcal{L}(\pi_L, \theta, \mu) \triangleq (D)$$ 

Where the dual function is defined as $$\max_{\pi_L \in \Pi} \; \mathcal{L}(\pi_L, \theta, \mu)$$. This problem can then be solved by dual ascent by leveraging the fact that at the optimal value, $$\nabla_{\pi_L} \mathcal{L}(\pi_L, \theta, \mu) = 0$$ and $$\nabla_{\theta} \mathcal{L}(\pi_L*, \theta, \mu) = \text{direction of steepest descent}$$

I have skipped the computation of the derivatives here. Under a reward function parametrised by some (potentially non-optimal) $$\theta$$, $$\nabla_{\pi_L} \mathcal{L}(\pi_L, \theta, \mu) = 0$$ when:

$$\pi_L(a_T \vert s_t) = \exp (Q_{\theta, t}^{\text{soft}}(s_t, a_t) - V_{\theta, t}^{\text{soft}}(s_t))$$

Where the value functions satisfy a recursive relationship. As it turns out, in equating the derivatives to zero, a soft version of value iteration emerges as a solution to the optimal policy. In this soft value iteration:

$$
\begin{align*}
V_{\theta, t}^{\text{soft}}(s_t) &= \log \sum_{a_t \in A} \exp (Q_{\theta, t}^{\text{soft}}(s_t, a_t)) \\
Q_{\theta, t}^{\text{soft}}(s_t, a_t) &= \underbrace{\theta^\intercal \phi(s_t, a_t)}_{\text{reward fn.}} + \gamma \; \mathbb{E}_{\tau} \left[ V_{\theta, t+1}^{\text{soft}}(s_{t+1} \vert s_t, a_t)  \right] \\
Q_{\theta, T-1}^{\text{soft}}(s_{T-1}, a_{T-1}) &= \theta^\intercal \phi(s_{T-1}, a_{T-1}) ^\text{ there is admittedly some hand-waving at t = T}
\end{align*}
$$

This version of value iteration is termed soft because instead of the value functions satisfy a "softened" version of Bellman equations since the max over actions is replaced by a log-sum-exp which acts as a soft max. This intuitively leads us to the solution of the dual function, which is to simply run reinforcement learning until a policy $$\pi_L$$ which maximises the reward function parametrised by the current $$\theta$$ is obtained. 

The dual problem (solution to the optimum $$\theta$$ which minimises the max over the Lagrangian) is obtained by computing the gradient $$\nabla_{\theta} \mathcal{L}(\pi_L*, \theta, \mu)$$ and stepping in that direction (gradient descent). This is a much simpler computation as only the feature expectation matching term depends on $$\theta$$.

$$\nabla_{\theta} \mathcal{L}(\pi_L*, \theta, \mu) = \mathbb{E}_{\pi_L} \left[\sum_{t=1}^{T} \gamma^t \phi(s_t, a_t) \right] - \mathbb{E}_{\pi_E} \left[ \sum_{t=1}^{T} \gamma^t \phi(s_t, a_t) \right]$$

>This finally brings us to the **MaxCausalEnt IRL procedure:**
>- Initialise some $$\theta_0$$ and set iterations $$k \gets 0$$
>- Obtain a policy $$\pi^{k}_L$$ via soft value iteration
>- Update reward function 
> $$\theta_{k+1} \gets \theta_k + $$ 
> $$\alpha \left[\mathbb{E}_{\pi_L} \left[\sum_{t=1}^{T} \gamma^t \phi(s_t, a_t) \right] - \mathbb{E}_{\pi_E} \left[ \sum_{t=1}^{T} \gamma^t \phi(s_t, a_t) \right] \right] $$

### Fundamentals: Entropy
[Entropy](https://en.wikipedia.org/wiki/Entropy_(information_theory)) is a quantity arising in several scientific areas. In our discussion, we refer to the Shannon entropy (information theory). In simple terms, this is the amount of randomness in a probability distribution, with higher values indicating a higher expected uncertainty. It relates closely to another quantity called "surprise" which is defined as $$\log \frac{1}{p(x)}$$. An event that is improbable is very surprising (and surprise exactly measures this). Entropy is the expected surprise. This means that it is a weighted average of the probability of an event with its surprise.

$$
\begin{align*}
H(p(x)) &\triangleq \mathbb{E}_{p} \left[ \log \frac{1}{p(x)} \right] \\
&\triangleq \sum_{x \in X} p(x) \log \frac{1}{p(x)} \\
&\triangleq - \sum_{x \in X} p(x) \log p(x) \\
&\triangleq - \mathbb{E}_p \log p(x)
\end{align*}
$$


#### References

Ziebart, Brian D and Maas, Andrew L and Bagnell, J Andrew and Dey, Anind K and others (2008). Maximum entropy inverse reinforcement learning.. 

Ziebart, Brian D and Bagnell, J Andrew and Dey, Anind K (2010). Modeling interaction via the principle of maximum causal entropy. 

Ng, Andrew Y and Harada, Daishi and Russell, Stuart (1999). Policy invariance under reward transformations: Theory and application to reward shaping. 

Jaynes, Edwin T (1957). Information theory and statistical mechanics. Physical review


---
{: data-content="footnotes"}

[^1]: At any given state, you'd like to be open to all options. But there's not necessarily a benefit to being in states that are unpredictable.
