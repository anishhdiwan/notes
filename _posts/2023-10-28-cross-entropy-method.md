---
layout: post
category: paper-notes
title: Cross Entropy Method
description: The cross-entropy method (CEM) [Kobilarov, (2012), Botev, et al. (2013)] is one of the standard tools seen in model-based RL literature to solve estimation and optimization problems. It is fundamentally based on iteratively minimising the Kullback-Leibler divergence (or cross-entropy loss) between time-evolving versions of a distribution ...
# image: /assets/images/post-thumbnail.jpg # URL to the post's featured image
# image_alt: "A description of the image for accessibility purposes." # Alt text for the image
---

The cross-entropy method (CEM) [Kobilarov, (2012), Botev, et al. (2013)] is one of the standard tools seen in model-based RL literature to solve estimation and optimization problems. It is fundamentally based on iteratively minimising the Kullback-Leibler divergence (or cross-entropy loss) between time-evolving versions of a distribution to obtain a final distribution such that there is a high probability of sampling high utility samples (utility is another word for cost or whatever objective is being maximised). 


An example use case could be as follows. Given the dynamics and reward model in a reinforcement learning task, CEM (or more precisely, "CEM for optimization" as CEM can also be used for estimation) can be used in the planning step to obtain a "good" action distribution for the current state. Here's an algorithm for this.

- In the simulation step, sample a sequence of action vectors (say of length = horizon)
- Evaluate a combined reward signal for this action sequence from the current state
- Update the sampling distribution such that the updated distribution offers a better expected value of the objective
- Repeat for some number of simulation steps
- In the real world, sample a sequence of actions from the final CEM distribution for this state

### Formalisation

Although CEM is generally used for optimization, the original problem definition is for rare-event estimation. Its applications are to:

**Estimate** the probability of a rare event $$l = \mathbb{E}(S(x))$$ where $$x$$ is a random variable taking values in some set $$\mathcal {X}$$ and $$S()$$ is some (objective) function on $$\mathcal{X}$$. Estimation is especially useful when $$l = \mathbb{P}(S(x) >= \lambda )$$ where $$\lambda$$ is a threshold indicating the rare event.

**Optimize** some given objective function $$S(x)$$ where $$x \in X$$. If $$S()$$ is unknown then it can be estimated via simulation.

#### References

Kobilarov, Marin (2012). Cross-entropy motion planning. The International Journal of Robotics Research

Botev, Zdravko I and Kroese, Dirk P and Rubinstein, Reuven Y and L’Ecuyer, Pierre (2013). The cross-entropy method for optimization. 