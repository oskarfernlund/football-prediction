# Football Prediction :soccer:
---

Prediction of football scores using Gaussian Processes. We create independent models of home and away team scores using a sparse variational Gaussian Process (SVGP) with a fully-factorised Poisson likelihood and a Matérn 5/2 kernel.

$p(\mathbf{y}|\mathbf{f}) = \prod_{i=1}^{N} \frac{\lambda_{i}^{y_i} \exp(-\lambda_i)} {y_i!}$ where $\lambda_i = \exp (f_i)$ and $f \sim \mathcal{GP}(m, k)$.

The bulk of the code is contained in the notebook `model.ipynb` with some helper functions in `/src`.


## Building the Environment :hammer:

To build the environment, run the following commands in a shell:

```
$ python -m venv env
$ source env/bin/activate
$ (env) pip install -r requirements.txt
```

To create a Jupyter kernel, run:

```
$ (env) ipython kernel install --name "env" --user
```

**Note**: The setup instructions above have only been tested for python 3.10 on M1 Mac.

## Running the Jupyter Notebook :running:

To run the jupyter notebook (`model.ipynb`), run:

```
$ (env) jupyter notebook
```



