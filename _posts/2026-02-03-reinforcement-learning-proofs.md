---
layout: post
category: tutorial
title: A blueprint for reinforcement learning proofs
description: A tutorial on common proof themes and popular results in RL.
featured: true
---

This tutorial walks through popular proofs in reinforcement learning. I present common ways of proving things (mostly properties of algorithms/operators) and illustrate these blueprints with examples and influential basic results in RL. Naturally, I assume that the reader is familiar with RL, definitions of basic quantities, and some popular algorithms. 

### Common Proof "Blueprints"

#### Contractions & Fixed-Point Theorem
***Tl:Dr:*** A function is called a contraction if it "squeezes" some metric space. As per the Banach fixed-point theorem, when contractions are recursively applied on the space, they converge to a fixed-point. We can use this property to show that recursive application of some RL operator always converges to a solution! 

<div class="definition" id="def-contraction">
  <span class="definition-title">Definition (Contraction)</span>
  Let $(X, d)$ be a metric space. A mapping $T : X \rightarrow X$ is a contraction mapping (or contraction) if there exists a constant $c \in [0,1)$ such that:
  $$d(T(x), T(y)) \leq c\, d(x,y).$$
  for all $x, y \in X$
</div>

#### References


---
{: data-content="footnotes"}

[^1]: [The Contraction Mapping Theorem](https://www.math.ucdavis.edu/~hunter/book/ch3.pdf)





<!-- #### How to write definitions here?

<div class="definition" id="def-policy">
  <span class="definition-title">Definition (Policy)</span>
  A policy is a conditional distribution \( \pi(a \mid s) \).
</div>

<div class="theorem" id="theorem-contraction">
  <span class="theorem-title">Theorem (Contraction)</span>
  If \(T\) is a \(\gamma\)-contraction, then there exists a unique fixed point \(x^\star\).
</div>

<div class="proof">
  <span class="proof-title">Proof</span>
  Apply the Banach fixed-point theorem. <span class="qed">□</span>
</div>

See [Definition (Policy)](#def-policy) for the formal statement.
See [Theorem (Contraction)](#theorem-contraction) for the formal statement.
 -->

