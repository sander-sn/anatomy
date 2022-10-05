
# Introduction

*[to be added]*

# Quickstart
*If you haven't already, install the package via `pip install anatomy`, preferably in a new environment with Python 3.9.*

The anatomy package uses a simple workflow. An `Anatomy` object is initially estimated on your forecasting setup (using your data and your models), is then stored to disk, and can then be loaded at any future time without requiring re-estimation.

After initial estimation, an `Anatomy` can anatomize:

* forecasts produced by any combination of your models
* your original forecasts
* any loss or gain function applied to your forecasts
* an arbitrary subset of your forecasts

all of which requires *no additional computational time*.

## General structure
You may already have trained your models before you create the `Anatomy`, and the aggregate of all your models at all periods may be too large to fit into your RAM. During estimation, the `Anatomy` will therefore ask you for the specific model and dataset it needs at a given iteration by calling your mapping function:

    from anatomy import *
    
    def my_map(key: AnatomyModelProvider.PeriodKey) -> \
            AnatomyModelProvider.PeriodValue:  
        
        train, test, model = ...  # load from somewhere or generate here
        
        return AnatomyModelProvider.PeriodValue(train, test, model)

You wrap the mapping function in an `AnatomyModelProvider` alongside information about the forecasting application:
 
    my_provider = AnatomyModelProvider(  
        n_periods=..., n_features=..., model_names=[...],
        y_name=..., provider_fn=my_map
    )

and finally create the `Anatomy`:

    my_anatomy = Anatomy(provider=my_provider, n_iterations=...).precompute(  
        n_jobs=16, save_path="my_anatomy.bin"
    )

After running the above, the `Anatomy` is estimated and stored in your working directory as `my_anatomy.bin`.

# Example:

*For convenience, the examples below are contained in a single Python script available [here](https://github.com/sander-sn/anatomy/blob/main/examples/simple_dgp.py).*

To get started, we need a forecasting application. We use a linear DGP to generate our dataset consisting of 500 observations of the three predictors `x_{0,1,2}` and our target `y`:
          
    # set random seed for reproducibility:
    np.random.seed(1338)
    
    xy = pd.DataFrame(np.random.normal(0, 1, (500, 3)), columns=["x_0", "x_1", "x_2"])
    xy["y"] = xy.sum(axis=1) + np.random.normal(0, 1, 500)

    # set a unique and monotonically increasing index (default index would suffice):
    xy.index = pd.date_range("2021-04-19", "2022-08-31").map(lambda x: x.date())

For convenience, the `AnatomySubsets` includes a generator that splits your dataset into training and test sets according to your forecasting scheme. Here, we include 100 periods in our first training set, forecast the target of the next period with no gap between the training set and the forecast, extend our training set by one period, and repeat until we reach the end of our data:
    
    subsets = AnatomySubsets.generate(
        index=xy.index,
        initial_window=100,
        estimation_type=AnatomySubsets.EstimationType.EXPANDING,
        periods=1,
        gap=0
    )
	
In this example, we have not yet trained our models. We thus do so directly in our mapping function:

    def mapper(key: AnatomyModelProvider.PeriodKey) -> \
            AnatomyModelProvider.PeriodValue:  
  
        train = xy.iloc[subsets.get_train_subset(key.period)]  
        test = xy.iloc[subsets.get_test_subset(key.period)]  

        if key.model_name == "ols":
            model = train_ols(train.drop("y", axis=1), train["y"])  
        elif key.model_name == "rf":  
            model = train_rf(train.drop("y", axis=1), train["y"])  
      
        return AnatomyModelProvider.PeriodValue(train, test, model)

using `train_ols` and `train_rf`, which train a model and yield its prediction function wrapped in an `AnatomyModel`:

    from sklearn.ensemble import RandomForestRegressor  
    from sklearn.linear_model import LinearRegression
    
    def train_ols(x_train: pd.DataFrame, y_train: pd.Series) -> AnatomyModel:
        ols_model = LinearRegression().fit(x_train, y_train)  
      
        def pred_fn_ols(xs: np.ndarray) -> np.ndarray:  
            xs_df = pd.DataFrame(xs, columns=x_train.columns)  
            return np.array(ols_model.predict(xs_df)).flatten()  
      
        return AnatomyModel(pred_fn_ols)  

    def train_rf(x_train: pd.DataFrame, y_train: pd.Series) -> AnatomyModel:
        rf_model = RandomForestRegressor(random_state=1338).fit(x_train, y_train)  
      
        def pred_fn_rf(xs: np.ndarray) -> np.ndarray:  
            xs_df = pd.DataFrame(xs, columns=x_train.columns)  
            return np.array(rf_model.predict(xs_df)).flatten()  
      
        return AnatomyModel(pred_fn_rf)

We now have all we need to train the models and estimate the `Anatomy`:
 
    provider = AnatomyModelProvider(
        n_periods=subsets.n_periods,
        n_features=xy.shape[1]-1,
        model_names=["ols", "rf"],
        y_name="y",
        provider_fn=mapper
    )  
      
    anatomy = Anatomy(provider=provider, n_iterations=10).precompute(  
        n_jobs=16, save_path="anatomy.bin"  
    )

At this point, the `Anatomy` is stored as `anatomy.bin` in our working directory. We can load it at any later point using `anatomy = Anatomy.load("anatomy.bin")`.

# Anatomizing
We can now use our estimated `Anatomy` to anatomize our forecasts. In this example, we are using two models, `rf` and `ols`, as well as an equal-weighted combination of the two:

    groups = {
        "rf": ["rf"],  
        "ols": ["ols"],
        "ols+rf": ["ols", "rf"]
    }

## Anatomize the out-of-sample $R^2$ of the forecasts:
To decompose the out-of-sample $R^2$ of our forecasts produced by the two models and their combination, we use the unconditional forecasts as benchmark and provide a function transforming forecasts into out-of-sample $R^2$ to the `Anatomy`:
      
    prevailing_mean = np.array([  
        xy.iloc[subsets.get_train_subset(period=i)]["y"].mean()  
        for i in range(subsets.n_periods)  
    ])  
      
    def transform(y_hat, y):  
        return 1 - np.sum((y - y_hat) ** 2) / np.sum((y - prevailing_mean) ** 2)

    df = anatomy.explain(
        model_sets=AnatomyModelCombination(groups=groups),
        transformer=AnatomyModelOutputTransformer(transform=transform)
    )

This yields the change in out-of-sample $R^2$ attributable to each predictor:

    >>> df
                                     base_contribution       x_0       x_1       x_2
    rf     2021-07-28 -> 2022-08-31           0.000759  0.257240  0.238458  0.215062
    ols    2021-07-28 -> 2022-08-31           0.000000  0.285329  0.267940  0.249870
    ols+rf 2021-07-28 -> 2022-08-31           0.000383  0.279073  0.259201  0.240548

### *Interpretation*

The Shapley-based decomposition can be understood as a means to fairly allocate a single value (in this case the out-of-sample $R^2$) amongst multiple actors contributiong to it (the predictors in our model). This implies that the individual contributions of the actors and the contribution of the empty set of actors (`base_contribution`) sum up exactly to the original value that is decomposed.

The above depicts the individual contributions to the out-of-sample $R^2$, which can be negative, if a given predictor hurts accuracy, or positive, if a given predictor increases accuracy.  In this case, all predictors contribute positively to the out-of-sample $R^2$. In practice, predictors can hurt accuracy by reducing the out-of-sample $R^2$.

*Note: In this example, we use the prevailing mean (average of the target in the training sets) as benchmark to compute the out-of-sample R². The average forecast of an OLS model concides with this benchmark, which explains why the `base_contribution` of OLS is exactly zero.*

## ... the RMSE of the forecasts:

    def transform(y_hat, y):  
        return np.sqrt(np.mean((y - y_hat) ** 2))
    
    df = anatomy.explain(  
        model_sets=AnatomyModelCombination(groups=groups),  
        transformer=AnatomyModelOutputTransformer(transform=transform)  
    )

which yields the change in root mean squared error attributable to each predictor:

    >>> df
                                     base_contribution       x_0       x_1       x_2
    rf     2021-07-28 -> 2022-08-31           2.105513 -0.351430 -0.326065 -0.296708
    ols    2021-07-28 -> 2022-08-31           2.106313 -0.414488 -0.386125 -0.371149
    ols+rf 2021-07-28 -> 2022-08-31           2.105910 -0.397903 -0.368193 -0.350080

### *Interpretation*
The previous decomposition was that of a gain function. In this case, we use the RMSE *loss* function, implying that a predictor with a negative contribution increases forecasting accuracy. Because the RMSE cannot be negative, the `base_contribution`, which is the RMSE of the average forecasts of the models, can only be positive.

Similar to the previous case, we find that all predictors contribute positively to forecasting accuracy (by contributing negatively to the RMSE).
   
## ... the MAE:

    def transform(y_hat, y):  
        return np.mean(np.abs(y - y_hat))
    
    df = anatomy.explain(  
        model_sets=AnatomyModelCombination(groups=groups),  
        transformer=AnatomyModelOutputTransformer(transform=transform)  
    )

which yields the change in mean absolute error attributable to each predictor:

    >>> df
                                     base_contribution       x_0       x_1       x_2
    rf     2021-07-28 -> 2022-08-31           1.679359 -0.299382 -0.249591 -0.221651
    ols    2021-07-28 -> 2022-08-31           1.679946 -0.345583 -0.300613 -0.288960
    ols+rf 2021-07-28 -> 2022-08-31           1.679652 -0.330303 -0.283129 -0.262270

### *Interpretation*

The interpretation is similar to that of the RMSE.
 
## ... the SE:

    def transform(y_hat, y):  
        return (y - y_hat) ** 2
  
    df = anatomy.explain(
        model_sets=AnatomyModelCombination(groups=groups),
        transformer=AnatomyModelOutputTransformer(transform=transform)
    )

which yields the change in squared error attributable to each predictor for each forecast:

    >>> df
                       base_contribution       x_0       x_1       x_2
    rf     2021-07-28           0.026485 -0.063311 -0.115985  0.163743
           2021-07-29           0.021582  0.238569 -0.370448  0.694773
           2021-07-30           2.451742 -2.702365  1.660915 -1.407192
    ...                              ...       ...       ...       ...

*Note: The `transform` function in this case does not aggregate (returns a vector instead of a scalar). The `Anatomy` thus yields one decomposition per forecast, which is also known as a local (as opposed to global) decomposition.*

### *Interpretation*

The previous decompositions have shown consistently that all predictors increase forecasting accuracy when it is gauged over the entire period (2021-07-28 to 2022-08-31). Anatomizing instead each individual forecast reveals that this is not always true, at least not at the local level. We now see that individual predictors are contributing positively to the squared error of some forecasts (thus reducing forecast accuracy).

## ... the RMSE of the forecasts in a subperiod:

    subset = pd.date_range("2021-07-28", "2021-08-06").map(lambda x: x.date())
    	    
    def transform(y_hat, y):  
        return np.sqrt(np.mean((y - y_hat) ** 2))  
        
    df = anatomy.explain(  
        model_sets=AnatomyModelCombination(groups=groups),  
        transformer=AnatomyModelOutputTransformer(transform=transform),  
        explanation_subset=subset 
    )

which yields the change in root mean squared error in the ten-day period attributable to each predictor:

```
>>> df
                                 base_contribution       x_0       x_1       x_2
rf     2021-07-28 -> 2021-08-06           1.402950 -0.149847 -0.199615  0.118059
ols    2021-07-28 -> 2021-08-06           1.404761 -0.155275 -0.323760  0.317055
ols+rf 2021-07-28 -> 2021-08-06           1.403853 -0.169584 -0.275408  0.211672
```

### _Interpretation_

In this short subperiod of ten days, we find that our last predictor contributed positively to the RMSE (and thus negatively to forecasting accuracy).


## ... or just the raw forecasts:

    def transform(y_hat):
        return y_hat
        
    df = anatomy.explain(
        model_sets=AnatomyModelCombination(groups=groups),
        transformer=AnatomyModelOutputTransformer(transform=transform)
    )

which yields the change in the forecast attributable to each predictor:

    >>> df
                       base_contribution       x_0       x_1       x_2
    rf     2021-07-28           0.070861  0.222932  0.360182 -0.315820
           2021-07-29           0.070354  0.250389 -0.617779  0.984992
           2021-07-30           0.071163 -1.514547  0.839562 -0.835136
    ...                              ...       ...       ...       ...

### *Interpretation*

Decomposing the forecasts themselves yields contributions that bear no relation to forecasting accuracy. Hence, a negative or positive contribution means no more than a decrease or increase in the forecast at that period attributable to the given predictor, which may or may not have be good for forecasting accuracy.

***Hence: beware, a high average absolute contribution does not necessarily translate into a high gain in accuracy. That is precisely why we need to decompose the loss or gain directly, and in consequence, take into account our target and how far away our forecasts were from it.***

From the anatomized raw forecasts, we can compute the  $\text{oShapley-VI}$ by averaging over the magnitudes of the invididual contributions:

    >>> df.abs().mean(axis=0)
    base_contribution    0.078436
    x_0                  0.865098
    x_1                  0.787327
    x_2                  0.773340

## The Efficiency property

*During estimation, the `Anatomy` checks that the individual attributions of the predictors to the forecasts sum up exactly to the forecasts produced by the models. The estimation would be aborted if efficiency does not hold.*

Due to the efficiency property of Shapley values, summing the individual contributions yields the decomposed value exactly. We can check that the results that `Anatomy` yields are consistet. We can recover the RMSE from the decomposed RMSE, but we can also recover the RMSE from the decomposed forecasts. Let's make sure that they match.

We first anatomize the RMSE:

    def transform(y_hat, y):  
        return np.sqrt(np.mean((y - y_hat) ** 2))  
      
    df = anatomy.explain(  
        model_sets=AnatomyModelCombination(groups=groups),  
        transformer=AnatomyModelOutputTransformer(transform=transform)  
    )

The RMSE (or any other decomposed value) is the sum of the individual attributions, `rmse_a = df.sum(axis=1)`:

    >>> rmse_a
    rf      2021-07-28 -> 2022-08-31    1.131310
    ols     2021-07-28 -> 2022-08-31    0.934551
    ols+rf  2021-07-28 -> 2022-08-31    0.989733

We next decompose the raw forecasts and compute the RMSE from these:

    def transform(y_hat):  
        return y_hat  
  
    df = anatomy.explain(  
        model_sets=AnatomyModelCombination(groups=groups),  
        transformer=AnatomyModelOutputTransformer(transform=transform)  
    )

The forecasts are recovered as the sum of the individual attributions, `y_hat = df.sum(axis=1)`:

    >>> y_hat
    rf      2021-07-28    0.338154
            2021-07-29    0.687956
            2021-07-30   -1.438958
                               ...   

From the forecasts, we can compute the RMSE:       
      
    y_true = np.hstack([  
        xy.iloc[subsets.get_test_subset(period=i)]["y"]
        for i in range(subsets.n_periods)  
    ])  
      
    rmse_b = pd.Series({  
        key: np.sqrt(np.mean((y_true - y_hat.xs(key)) ** 2))   
        for key in groups.keys()  
    })

which yields the same RMSE as the sum of the contributions of the RMSE decomposition:

    >>> rmse_b
    rf        1.131310
    ols       0.934551
    ols+rf    0.989733

