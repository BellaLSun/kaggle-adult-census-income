
Kaggle challenge : [Adult Census Income](https://www.kaggle.com/uciml/adult-census-income)

### Installation

- Setup environment

```
virtualenv venv
source venv/bin/activate
pip install -r requirements
```

- Jupyter python 2 kernel (if necessary)

```
python2 -m ipykernel install --user
```

- Download the data

```
mkdir data
```

Download the zip files into this folder.

- Run the jupyter notebook

```
jupyter notebook
```

### Main goal

The main goal is to predict if someone earns more or less than 50k per year based on static features of the person : the age, the occupation, the native country etc. 

### The data

There are multiple kinds of features :

- The static description
    + native country (+ continent)
    + race
    + sex
- The conjugal situation
    + marital status
    + relationship
    + single
- The educational background (education / education.num)
- The current job informations
    + hours worked per week
    + capital (can be positive or negative)
    + workclass
    + occupation

We first transformed and clean the data by :

 - Removing duplicate informations : `eduction` / `education.num`
 - Refine features : `continent`, `is_single`
 - Merge the `capital` data into one and use a logarithmic transformation
 - Convert `sex` and `income` into float data
 - Replace `?` by `None`

The exploratoring part can be seen using [exploratory.ipynb](https://github.com/tillmd/kaggle-adult-census-income/blob/master/exploratory.ipynb)

This visualization part taught us that some features seem to be useful to split the data easily but it would be difficult to be perfect.

### The prediction

#### The algorithm

We used regressors instead of classifiers to be able to plot the ROC curves and thus be more flexible on the threshold choice.

We chose to benchmark 3 algorithms:
    - The `Linear Regression` for its simplicity, quickness and performances
    - The `Random Forest` for its global performances and the computation of feature importance
    - The `Gradient Boosting` to compare bagging with boosting (and the same reasons as the RF)

We also add an aggregated regressor that uses the previous predictions and average them.

#### Performance metrics

We describe our results in [prediction.ipynb](https://github.com/tillmd/kaggle-adult-census-income/blob/master/prediction.ipynb)

Our predictions are based on a 30 KFold cross validation. We used :

 - ROC Curve
 - Confusion matrix
 - Precision / Recall 

to compare ourto have an overhaul quite good predictor, with good f-score.

The problem is to be accurate in both class (`>50k`, `<=50k`) that is why we didn't emphasize on the precision metric.

#### Future work

 - Feature engineering
     + Try to transform categorical data into numerical data
     + Find new features based on the actual one
     + Interpret FNLWGT influence
 - Be smarter with the prediction aggregation (learning overlay)
 - Use the OOB score as performance metric
 - The data is a bit imbalanced : oversampling / undersampling
 - Add unit tests

#### Appendix 

 - [FNLWGT](https://www2.census.gov/programs-surveys/sipp/guidance/SIPP_2008_USERS_Guide_Appendix_C.pdf)

FNLWGT = SSCA \* BW \* DCF \* NAF :
    
    The \# of people the census takers believe that observation represents

SSCA (the second stage calibration adjustment) :

    [...] equal to the ratio of the pre-second-stage weight and the final weight after the calibration process is completed.

BW (Base Weight) : 

    In simplified terms, a base weight of 1,000 assigned to a sampled person means that the sampled person “represents” 1,000 persons in the U.S. population. 

DCF (Duplication Control Factors) : 

    [...] value between 1 and 4 inclusive [...]

NAF (Household Noninterview Adjustment Factors) :

    The noninterview adjustment factor is intended to adjust for the presence of Type A noninterview households (households that are not interviewed because the occupants were temporarily absent, no one was home, the occupants refused participation, or the occupants could not be located). 

