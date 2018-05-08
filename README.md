# MCA - Markov Cluster Algorithm

## Introduction

The present repository contains numpy and spicy sparse implementations of the [Markov cluster algorithm](https://micans.org/mcl/). It tries to find agglomerates of vertices that are connected densely.

## Istallation

### Platform

Tested on Windows 10.

### Prerequisites

It requires and has been tested for the following modules:
1. `python` (3.5)
1. `numba` (0.30.1)
1. `numpy` (1.14.0)
1. `scipy` (0.18.1)
1. `sklearn` (0.18.1)

### Installation

One needs the full contents of this repository to run the MCA classifier. 

## How to use?

The attribute and method names follow the `scipy` convention:

```
    cls = MCsparse(diag_scale = 1.0, expand_power = 2, inflate_power = 2, 
                   max_iter = 10, save_steps = 0, threshold = 0.00001, tol = 0.001)
                 
    cls.fit(X)
    
    labels = cls.labels_
    
    labels = cls.fit_predict(X)
```

## Issues and TODO

* The time being, the programs can only be called from python code. A command line parser will be added later on.

* The `numpy` implementation has not been fully tested. It will perhaps be later removed from this repo.

* The `_colour_array_numba` function needs to be run in the `__init__` step to save compilation time on large graphs.

