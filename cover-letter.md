---
# spell-checker: disable
documentclass: scrartcl
title: Cover Letter
subtitle: "*Understanding Heating in Active Region Cores through Machine Learning II. Classifying Observations*"
author:
- W.T. Barnes
- S.J. Bradshaw
- N.M. Viall
bibliography: [paper/references.bib]
# spell-checker: enable
---
We are grateful to the referee for providing a thorough review of the manuscript and for making several very useful suggestions, all of which have greatly improved this manuscript. Below, we respond to each of the referee's specific concerns. Additionally, we note these changes in the text as appropriate using the "track changes" functionality provided by the AAS\TeX\ document class. A summary of these changes is included on the last page of the corrected manuscript as well.

# Specific Concerns

The comments of the referee are noted in *italics*. Our response to each concern is directly below in the indented block of text.

* *Please define hot and cool in the introduction, rather than using the terms in quotes. In general, using a term in quotes conveys no meaning and a simple definition provides more clarity.*

    > In our description of the emission measure slope, we have removed the word "cool" as the temperature range over which the slope is calculated is already defined a few sentences later. We have clarified the descriptions of the hot and cool channels in the context of the time lag channel pairs.

* *I would appreciate a discussion about the errors in both the synthetic features and real features. Do the authors compute errors for the real features (e.g. a time lag of 1000 seconds plus or minus 10 seconds)? Do the authors calculate errors in the synthetic features? If so, how do they compare? If not, what is a reasonable way to calculate them or account for them? Are the errors considered in the Random Forest model (this does not appear to be the case)? If not, why not?*

    > The Random Forest classification procedure used provides no way to specify uncertainties on the input data themselves. However, the Random Forest model does provide a measure of the uncertainty of the classifier through the probability maps in Figure 7 and we have added a sentence at the end of the second paragraph in Section 3.2 explicitly stating this. Additionally, when computing both the synthetic and observed emission measure distributions, we incorporate the uncertainties on the AIA intensities as computed by the `aia_bp_estimate_error.pro` routine in SSW. This is explicitly stated in Section 2.1 and in Section 3.2 of @barnes_understanding_2019. These uncertainties in the intensities then propagate to uncertainties on the differential emission measure distribution using the method of @hannah_differential_2012. While we do not consider these uncertainties when computing the emission measure slope, the effect on the resulting slope is likely small as the emission measure distribution is reasonably well-constrained by the AIA intensities over this temperature range. Regarding the time lag measurements, while we have not computed confidence intervals on the time lag itself, we note that the nominal time lag represents a weighted average of all the time lags along the line of sight over the data window in question. The cross correlation value of the time lag is due to both the error from noise in the data as well as an actual, physical spread of time lags (rather than inaccuracy or imprecision of the method), depending on what is along a given line of sight. @viall_signatures_2016 computed the distribution of time lags and cross-correlations for only fluctuations due to noise and showed that this physical spread dominates the distribution of observed time lags in active region observations.

* *I am not sure if the authors created 31 or 32 features in Paper I. Paper I says: "The 15 channel pairs for the time lag and cross-correlation combined with the emission measure slope represent a 31-dimensional feature space ... ." But Paper II uses 32 features. Please clarify.*

    > In @barnes_understanding_2019, we computed only 31 features: 15 time lags, 15 maximum cross-correlation values, and 1 emission measure slope. However, as part of this paper, we went back and calculated the peak temperature in each pixel of our simulated emission measure distributions such that our synthetic data set also has 32 features. This has been clarified at the end of the first paragraph in Section 3.1.

* *The Random Forest classifier was first developed by Ho (1995).*

    > We have added the reference to @ho_random_1995 in our discussion of random forest classifiers in Section 3.

* *Section 3.1 explains that the authors used 2/3 of the data to train the model, and 1/3 of the data to test it. Did the authors perform any cross validation? Or did the out-of-bag performance for the Random Forest give comparable results? If so, that might be worth mentioning.*

    > At the suggestion of the referee, we performed 5-fold random permutations cross-validation and found that the error on the test set in each case was comparable to the out-of-bag error as returned by the random forest classifier. We have added a footnote in Section 3.1 that states this explicitly.

* *There are quite a few papers on NOAA Active Region 11158. Do any of these other papers have results consistent with the results the authors present in this paper?*

    > While there are many papers discussing flaring activity in NOAA 1158, there are few dedicated to the study of quiescent heating in this region. @viall_survey_2017 and @warren_systematic_2012 examine the time lags and emission measure distributions, respectively, in this active region in the context of quiescent heating. We have already included a lengthy discussion of our consistency with the results of @viall_survey_2017. We have added an additional few sentences to the first paragraph of Section 4 discussing the consistency of our results with @warren_systematic_2012.

* *Why are the authors using such an old version of SunPy (0.9.5)? Is this to maintain the same software environment used to produce Paper I? If so, please state that in the Software section of the paper. Please also cite both the Zenodo deposit of the software used and the journal paper per Python package, e.g. "version 1.6.3 (Virtanen et al. 2021) of SciPy (Virtanen et al. 2020)". In this example, the first citation corresponds to the Zenodo deposit (http://doi.org/10.5281/zenodo.4718897) and the second to the journal paper (https://doi.org/10.1038/s41592-019-0686-2).*

    > We greatly appreciate the referee's attention to detail regarding software citations. We use v0.9.5 of sunpy in order to maintain consistency with the software environment used to produce Paper 1 [@barnes_understanding_2019]. We have noted this in the "Acknowledgement" section of the paper. Additionally, we have added the Zenodo DOI for the appropriate version of each package wherever possible and specified the version number in the citation of each software package.

* *I suggest the authors write a concise plain-language summary somewhere, maybe in the conclusion, that encapsulates the work in both Paper I and Paper II. Section 5 of Paper II does include a summary, but only summarizes the work of Paper II.*

    > We have added a single paragraph at the beginning of Section 5 that gives a concise summary of Paper 1.

\section{References}
