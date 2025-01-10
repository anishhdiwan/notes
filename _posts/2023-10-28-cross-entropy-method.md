---
layout: post
category: paper-notes
title: Cross Entropy Method
---

The cross-entropy method (CEM) [Kobilarov, (2012), Botev, et al. (2013)] is one of the standard tools seen in model-based RL literature to solve estimation and optimization problems. It is fundamentally based on iteratively minimising the Kullback-Leibler divergence (or cross-entropy loss) between time-evolving versions of a distribution to obtain a final distribution such that there is a high probability of sampling high utility samples (utility is another word for cost or whatever objective is being maximised). 


An example use case could be as follows. Given the dynamics and reward model in a reinforcement learning task, CEM (or more precisely, "CEM for optimization" as CEM can also be used for estimation — more on that later) can be used in the planning step to obtain a "good" action distribution for the current state. Then, something like MPC could be used to enact the first action from this sampling distribution (and then iteratively move ahead). The process would involve:

- In the simulation step, sampling a sequence of action vectors of size equal to the horizon
- Evaluating a combined reward signal for this action sequence from the current state
- Updating the sampling distribution such that the updated distribution offers a better expected value of the objective
- Repeating, for some number of simulation steps
- In the real world, sampling a sequence of actions from the final CEM distribution for this state and using some variant of MPC as usual. Then repeating this for as long as necessary

### Formalisation

Although CEM is generally used for optimization, the original problem definition is for rare-event estimation. Its applications are to:

**Estimate** the probability of a rare event $$l = \mathbb{E}(S(x))$$ where $$x$$ is a random variable taking values in some set $$\mathcal {X}$$ and $$S()$$ is some (objective) function on $$\mathcal{X}$$. Estimation is especially useful when $$l = \mathbb{P}(S(x) >= \lambda )$$ where $$\lambda$$ is a threshold indicating the rare event.

**Optimize** some given objective function $$S(x)$$ where $$x \in X$$. If $$S()$$ is unknown then it can be estimated via simulation.

#### References

Kobilarov, Marin (2012). Cross-entropy motion planning. The International Journal of Robotics Research

Botev, Zdravko I and Kroese, Dirk P and Rubinstein, Reuven Y and L’Ecuyer, Pierre (2013). The cross-entropy method for optimization. 