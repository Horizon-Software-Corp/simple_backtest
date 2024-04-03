# specification

```txt
1. add_features
2. preprocess
  * normalize, fractinal difference 
3. lagging
  * Adding lagged features (past values) to the current row.
4. target definition
  * mark target column
5. split train and test
  * simple forward test
6. get model instance
7. predict
 * next close by Quantile regression
```

preprocess.py has 1. 2. 3. 4.
