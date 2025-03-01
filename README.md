This is the code for the paper Objective Bound Conditional Gaussian Process for Bayesian Optimization. Their original code is available at https://github.com/twj-KAIST/OBCGP-BO. However, their code does not have any version information and their code seems to use tensorflow 0.x and gpflow 0.x, which is no longer available in pip. Therefore, we make some ajustments in their code to make it run in tensorflow 1.x and gpflow 1.x. In addition, we fixed some bugs in the code.

Here is how to use this package:
```bash
git clone https://github.com/HanyangHenry-Wang/OBCGP.git && cd OBCGP
conda create --name OBCGP-env python=3.6 -y
conda activate OBCGP-env
pip install -r requirements.txt 
```
