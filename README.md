# Understanding Heating in Active Region Cores through Machine Learning II. Classifying Observations

- [arXiv](https://arxiv.org/abs/2107.07612)
- [ApJ](https://iopscience.iop.org/article/10.3847/1538-4357/ac1514/pdf)
- [ADS](https://ui.adsabs.harvard.edu/abs/2021ApJ...919..132B/abstract)

## Authors

* W. T. Barnes, *NRC Postdoctoral Research Associate, Naval Research Laboratory*
* S. J. Bradshaw, *Department of Physics and Astronomy, Rice University*
* N. M. Viall, *NASA Goddard Space Flight Center*

## Abstract

Constraining the frequency of energy deposition in magnetically-closed active region cores requires sophisticated hydrodynamic simulations of the coronal plasma and detailed forward modeling of the optically-thin line-of-sight integrated emission.
However, understanding which set of model inputs best matches a set of observations is complicated by the need for any proposed heating model to simultaneously satisfy multiple observable constraints.
In this paper, we train a random forest classification model on a set of forward-modeled observable quantities, namely the emission measure slope, the peak temperature of the emission measure distribution, and the time lag and maximum cross-correlation between multiple pairs of AIA channels.
We then use our trained model to classify the heating frequency in every pixel of active region NOAA 1158 using the observed emission measure slopes, peak temperatures, time lags, and maximum cross-correlations and are able to map the heating frequency across the entire active region.
We find that high-frequency heating dominates in the inner core of the active region while intermediate frequency dominates closer to the periphery of the active region.
Additionally, we assess the importance of each observed quantity in our trained classification model and find that the emission measure slope is the dominant feature in deciding with which heating frequency a given pixel is most consistent.
The technique presented here offers a very promising and widely applicable method for assessing observations in terms of detailed forward models given an arbitrary number of observable constraints.

This is a follow-up to [Barnes, Bradshaw, and Viall (2019)](https://ui.adsabs.harvard.edu/abs/2019ApJ...880...56B/abstract). You can find the code and complete text and source code of that paper [here](https://github.com/rice-solar-physics/synthetic-observables-paper-models).

## Building the Paper

This repository contains all of the code and data needed to reproduce all of the figures in this paper, including training the random forest model that is used for making the heating frequency predictions.

First, build the conda environment and install all of the needed packages. This assumes you're using the [Anaconda distribution of Python](https://www.anaconda.com/products/individual), but this can be done using any distribution or environment manager.

```shell
$ conda create --name barnes-bradshaw-viall-20 --file conda-env.txt
$ conda activate barnes-bradshaw-viall-20
```

Next, build the paper using LaTeX and PythonTeX. You will need to install the PythonTeX package. This will run all of the needed Python code (including inline in the LaTeX source) to build all of the figures and train all of the models.

```shell
$ cd paper
$ pdflatex -synctex=1 -interaction=nonstopmode -file-line-error paper.tex
$ pythontex --interpreter python:python paper.tex
$ bibtex paper
$ pdflatex -synctex=1 -interaction=nonstopmode -file-line-error paper.tex
$ pdflatex -synctex=1 -interaction=nonstopmode -file-line-error paper.tex
```

This may take several minutes. You can find the built PDF in `paper/paper.pdf`.
