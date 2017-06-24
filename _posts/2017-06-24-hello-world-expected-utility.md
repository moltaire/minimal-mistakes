---
title: The decision model "Hello, world!"
excerpt: How to implement and fit a basic Expected Utility model in python
tags:
  - python
  - modeling
  - under_construction
---

# Background, idea

The *hello world* example in cognitive modeling in decision making is probably something like a simple Expected Utility (EU) model. This is what we will build here, to highlight modeling conventions, best practices and implementation details along the way. 


```python
import numpy as np
import pandas as pd
```

# Experiment

Let's say we have a single subject make repeated choices (`n_trials = 100`) between two (`n_gambles = 2`) risky gambles of the form {$p$; $m$}, where $p$ is the probability to win an amount $m$. With probability $1 - p$ nothing is won.  

First, we generate pairs of gambles that we would show to our subject. We do this in a vectorized fashion (we create an array of size `n_trials` $\times$ `n_gambles`, as it makes writing our model functions much easier later on.


```python
np.random.seed(444)

n_trials = 100
n_gambles = 2

m = np.random.randint(1, 9, size=(n_trials, n_gambles))
p = np.random.uniform(0, 1, size=(n_trials, n_gambles))

df = pd.DataFrame(dict(trial=np.arange(n_trials),
                       m1=m[:, 0],
                       m2=m[:, 1],
                       p1=p[:, 0],
                       p2=p[:, 1]))
df[['trial', 'p1', 'm1', 'p2', 'm2']].head()
```

<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>trial</th>
      <th>p1</th>
      <th>m1</th>
      <th>p2</th>
      <th>m2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.682016</td>
      <td>4</td>
      <td>0.803216</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0.999261</td>
      <td>8</td>
      <td>0.456960</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0.320048</td>
      <td>4</td>
      <td>0.947906</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0.172969</td>
      <td>8</td>
      <td>0.302126</td>
      <td>5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>0.069834</td>
      <td>7</td>
      <td>0.562184</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>

This is a minimal representation of our task, and indeed we could imagine that this would be the data we obtain from our subject. What's missing, of course, is the subject's actual behavior: The button presses, or choices the subject made in each trial. For our purposes here, we will *simulate* these from our model: We will assume that our fake subject uses this model to make their choices.

### Simulate choice data

In order to simulate from the model, we need to write the model first. The intuition behind the EU model is that subjects assign a utility $U$ to each outcome of each available alternative, and then (probabilistically) decide according to expected utilities (i.e. calculating expectations from outcome probabilities and outcome utilities). Utilities $U$ are often assumed to follow a nonlinear function of the objective outcomes like $u(x) = x^\alpha$, where $\alpha$ is a free parameter that determines the shape of the *utility curve* for a given subject.

```python
def U(x, α):
    """
    Compute utilities of x
    using a power function with parameter α.
    
    Parameters:
    x: outcomes, array like
    α: power parameter, scalar
    """
    u = x**α
    return u
```

```python
def EU(p, x, α):
    """
    Compute expected utilities of outcomes x
    with probabilities p and power parameter α.
    
    Parameters:
    p: outcome probabilities, array like
    x: outcomes, array like
    α: power parameter, scalar
    """
    eu = p*U(x, α)
    return eu
```

##### Choice rule

One additional part of the model that is traditionally out of focus is the *choice rule*. It translates option utilities into choice probabilities. There are many ways to do this, so the choice of the choice rule matters and is considered an *auxilliary assumption to the model*. 

* --- this post is still under construction --- *