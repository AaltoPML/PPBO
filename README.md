Projective Preferential Bayesian Optimization
==============================================

The repository contains a Python implementing of the method described in

Petrus Mikkola, Milica Todorović, Jari Järvi, Patrick Rinke, Samuel Kaski. (2020)  
**Projective Preferential Bayesian Optimization**,  
*In Proceedings of the 37th International Conference on Machine Learning*.  
([ICML'20 proceedings](https://proceedings.icml.cc/static/paper_files/icml/2020/2377-Paper.pdf), [BibTeX](docs/bibtex.bib), [arXiv preprint](https://arxiv.org/abs/2002.03113))

**Abstract**:  
Bayesian optimization is an effective method for finding extrema of a black-box function. We propose a new type of Bayesian optimization for learning user preferences in high-dimensional spaces. The central assumption is that the underlying objective function cannot be evaluated directly, but instead a minimizer along a projection can be queried, which we call a projective preferential query. The form of the query allows for feedback that is natural for a human to give, and which enables interaction. This is demonstrated in a user experiment in which the user feedback comes in the form of optimal position and orientation of a molecule adsorbing to a surface. We demonstrate that our framework is able to find a global minimum of a high-dimensional black-box function, which is an infeasible task for existing preferential Bayesian optimization frameworks that are based on pairwise comparisons.


## Overview


### Requirements
--- (Core library) ---
 * Python 3.7
 * numpy
 * pandas
 * scipy
 * sklearn
 * arspy
 * GPyOpt
 * matplotlib 
 
--- (User experiment) ---
 * ase
 * notebook
 * nglview
 * ipywidgets
 * ipympl
 * jupyter_ui_poll
 

### Instructions to experiment with Camphor/Cu(111)
The interface for eliciting the user preferneces over Camphor/Cu(111) test case described in the "User experiment" section of the paper. 

*Procedure - Linux*

Clone the repository: <br />
#> git clone https://github.com/AaltoPML/PPBO <br />
or download from the link above as a zip file, and unpack it

Install python packages: <br />
#> pip install -r requirements.txt

Run the jupyter notebook system <br />
#> jupyter notebook

Find the Jupyter-notebook *camphor-copper.ipynb*, and click it.<br />
In the notebook, click ![Screenshot_2020-05-22 Camphor-Copper - Jupyter Notebook](https://user-images.githubusercontent.com/57790862/82723533-47d17600-9cd8-11ea-9978-46f4551af440.png)!

### Instructions to run numerical experiments
Run the numerical described in the "Numerical experiment" section of the paper. Please note that the numerical experiments may take a long time even in a computer with tens of CPUs.<br />

Open ppbo_numerical_main.py<br />
Set a correct working directory (wd = ...)<br />
Uncomment the objective function you would like to run (e.g. env.run(six_hump_camel))<br />
Run the script!<br />

## Contact

 * Petrus Mikkola, petrus.mikkola@aalto.fi
 * Samuel Kaski, samuel.kaski@aalto.fi


Work done in the [Probabilistic Machine Learning research group](https://research.cs.aalto.fi/pml/) at [Aalto University](https://www.aalto.fi/fi).


## Reference

 * Mikkola, P., Todorović, M., Järvi, J., Rinke, P., and Kaski, S. (2020). Projective Preferential Bayesian Optimization.


## License

GPL v3, see `LICENSE`
