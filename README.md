# GD v.s. CGD for Linear Regression
Gradient Descent (GD) v.s. Conjugate Gradient Descent (CGD) for 2-D Linear Regression

## Introduction
This is one part of the course assignment for [HKUST-GZ MICS 6000I Physical Design Automation of Digital Systems](https://yuzhe630.github.io/teaching/2022-fall.html). This project is alive, maintained by <linfeng.du@connect.ust.hk>. Any discussion or suggestion would be greatly appreciated!

## Requirements
* Python 3.9
    * ply 3.11
    * matplotlib 3.5.1
    * logging 0.5.1.2
    * numpy 1.21.5

## How to Run
* `python main.py`
    * dataset: [data.txt](./data.txt)
    * source code: [main.py](./main.py)

## Results
* Logs:
    * in [GD-CGD.log](./GD-CGD.log)
* Convergence trajectories:
    * Gradient Descent (GD): in [GD-cost-convergence.png](./GD-cost-convergence.png)
    * Conjugate Gradient Descent (CGD): in [CGD-cost-convergence.png](./CGD-cost-convergence.png)
* 3-D Convergence Animation:
    * Gradient Descent (GD): in [GD-result-3d.gif](./GD-result-3d.gif)
    * Conjugate Gradient Descent (CGD): in [CGD-result-3d.gif](./CGD-result-3d.gif)
* Intermediate results (3-D figure) during convergence
    * in [result folder](./result/)