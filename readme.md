# **Multiclass Classification on the Cardiotocography Dataset**

This project explores the UCI Cardiotocography (CTG) dataset using two different problem formulations:

* 3-Class (NSP): Normal / Suspect / Pathologic

* 10-Class (CLASS): FHR pattern codes 1–10

Both are implemented as standalone pipelines with EDA, model sweeps and advanced hyperparameter tuning.

For the 3-class setup, an additional cost-sensitive clinical decision layer was implemented to incorporate medical risk considerations.

# ⭐ **Results Summary**

| Problem       | Model    | Accuracy  | Macro-F1 |
|---------------|----------|-----------|----------|
| NSP 3-Class   | XGBoost  | **0.953** | **0.912**|
| CLASS 10-Class| LightGBM | **0.917** | **0.876**|
| NSP 3-Class Cost-Sensitive Decision Layer| XGBoost | **0.950** | **0.909**|
| NSP 3-Class Clinical Triage System| XGBoost | **0.953** | **0.911**|

# **Dataset Citation**

> [Cardiotocography - UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/193/cardiotocography)

# **Dataset Overview**

The Cardiotocography contains 2126 rows of fetal cardiotocograms along with 21 features for diagnostic measurements. The dataset provides two different target labels, each supporting a different machine-learning task:

* NSP (3-class): Normal, Suspect, Pathologic (clinical fetal state)

* CLASS (10-class): detailed morphological FHR pattern codes (pattern-recognition task)

### NSP Label Distribution

<img src="nsp.png" alt="NSP Label Distribution" width="600"/>

### CLASS Label Distribution

<img src="class.png" alt="CLASS Label Distribution" width="600"/>

From this, we can see that the classes are highly imbalanced for both the NSP and the CLASS label.

# **PART I — NSP 3-Class Classification**

I first performed a full baseline model sweep across multiple ML families to establish initial performance for the NSP labels (Normal / Suspect / Pathologic). The results are shown below.

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>accuracy</th>
      <th>macro_f1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>XGBoost</td>
      <td>0.953052</td>
      <td>0.912811</td>
    </tr>
    <tr>
      <th>2</th>
      <td>LightGBM</td>
      <td>0.948357</td>
      <td>0.898420</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CatBoost</td>
      <td>0.934272</td>
      <td>0.878595</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Random Forest</td>
      <td>0.929577</td>
      <td>0.858786</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Extra Trees</td>
      <td>0.924883</td>
      <td>0.840949</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Linear SVM</td>
      <td>0.892019</td>
      <td>0.791722</td>
    </tr>
    <tr>
      <th>7</th>
      <td>kNN</td>
      <td>0.882629</td>
      <td>0.757024</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Logistic Regression</td>
      <td>0.870892</td>
      <td>0.755812</td>
    </tr>
    <tr>
      <th>9</th>
      <td>MLP</td>
      <td>0.854460</td>
      <td>0.726697</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Naive Bayes</td>
      <td>0.814554</td>
      <td>0.673743</td>
    </tr>
  </tbody>
</table>
</div>

XGBoost was the strongest baseline model, achieving the highest accuracy (0.953) and macro F1 (0.9128). It also showed the most stable performance on the Suspect class, which is the most challenging and clinically important class.

Since XGBoost clearly dominated the baseline comparison, it was selected for hyperparameter optimization using Optuna.

| Model    | Accuracy  | Macro-F1 |
|----------|-----------|----------|
| XGBoost  | **0.955** | **0.915**|

Even after tuning, the macro F1 remained essentially unchanged relative to the baseline. This indicates that the baseline XGBoost model was already operating close to the performance ceiling for this tabular CTG dataset.

<img src="confusionmatrixnsp.png" alt="NSP Label Distribution" width="600"/>

# **PART II — NSP 3-Class Classification with Cost-Sensitive Decision Layer and Clinical Triage System**

Although the tuned XGBoost model performed strongly, standard argmax prediction treats all misclassifications as equally important. In a clinical setting, this is not realistic. Misclassifying a *Pathologic* case as *Normal* is substantially more dangerous than over-predicting risk.

To reflect this clinical asymmetry, two post-processing methods were implemented on top of the model’s predicted probability distribution.

## **1. Cost-Sensitive Decision Layer (Expected Risk Minimization)**

This method replaces the standard argmax prediction with a decision rule that minimizes the **expected clinical cost**.
A cost matrix was defined to penalize clinically dangerous errors more heavily:

```python
# True class = row, Predicted class = column
C = np.array([
    [0, 1, 4],   # Normal misclassified as Pathologic = severe but tolerable false alarm
    [2, 0, 3],   # Suspect misclassified as Normal = missed warning
    [6, 3, 0],   # Pathologic misclassified as Normal = catastrophic
])
```

For each sample, the model computes:

> **expected_cost = cost_matrixᵀ × probability_vector**

The class with the **lowest expected cost** is chosen.

With this system, the model achieved:

* **Accuracy:** 0.950
* **Macro F1:** 0.909

As expected, the cost-sensitive layer slightly reduces overall performance metrics, but produces more conservative behavior in borderline cases and reduces underestimation errors.

## **2. Clinical Triage System (Threshold-Based Escalation)**

A second, simpler strategy applies decision thresholds directly to the predicted probabilities:

```python
if P(Pathologic) ≥ 0.30 → predict Pathologic
elif P(Suspect) ≥ 0.45 → predict Suspect
else → predict Normal
```

This mimics real-world CTG triage, where even moderately elevated risk indicators are escalated rather than left untreated.

The triage system reached:

* **Accuracy:** 0.953
* **Macro F1:** 0.911

Notably, the accuracy remains identical to the baseline XGBoost model, while still shifting predictions upward in risk when necessary.

Key characteristics:

* **Pathologic recall is preserved**, avoiding dangerous false negatives.
* **Borderline cases are pushed upward** (Normal → Suspect, Suspect → Pathologic).
* Accuracy is maintained (0.953), showing that conservative escalation does not harm overall predictive performance.

Both methods provide alternative prediction strategies that better align with obstetric risk management. While the cost-sensitive decision layer slightly reduces overall accuracy, **both approaches create a more clinically responsible decision boundary—favoring patient safety over pure statistical optimization**.

# **PART III — CLASS 10-Class Classification**

Similar to how I handled the NSP Class, I first performed a full baseline model sweep across multiple ML families to establish initial performance for the CLASS labels (1-10). The results are shown below.

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>accuracy</th>
      <th>macro_f1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>LightGBM</td>
      <td>0.903756</td>
      <td>0.862682</td>
    </tr>
    <tr>
      <th>2</th>
      <td>XGBoost</td>
      <td>0.901408</td>
      <td>0.840438</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CatBoost</td>
      <td>0.894366</td>
      <td>0.834891</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Random Forest</td>
      <td>0.896714</td>
      <td>0.829622</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Extra Trees</td>
      <td>0.880282</td>
      <td>0.801154</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Linear SVM</td>
      <td>0.842723</td>
      <td>0.755262</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Logistic Regression</td>
      <td>0.807512</td>
      <td>0.719180</td>
    </tr>
    <tr>
      <th>8</th>
      <td>MLP</td>
      <td>0.812207</td>
      <td>0.684966</td>
    </tr>
    <tr>
      <th>9</th>
      <td>kNN</td>
      <td>0.706573</td>
      <td>0.597331</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Naive Bayes</td>
      <td>0.615023</td>
      <td>0.585003</td>
    </tr>
  </tbody>
</table>
</div>

LightGBM was the strongest baseline model, achieving the highest accuracy (0.903) and macro F1 (0.862). For this reason, it was selected for hyperparameter optimization using Optuna.

| Model    | Accuracy  | Macro-F1 |
|----------|-----------|----------|
| LightGBM  | **0.917** | **0.876**|

After the tuning, the accuracy and macro F1 were marginally improved compared to the baseline. The confusion matrix of this model is shown below.

<img src="confusionmatrixclass.png" alt="NSP Label Distribution" width="600"/>

# **How to Use This Repository**

1. **Install dependencies**

```bash
pip install -r requirements.txt
```

2. **To run the NSP Class Classification pipeline**

Execute notebooks **in order**:

```
01_EDA.ipynb  
02_NSP_Baseline.ipynb  
03_NSP_Complex.ipynb  
06_NSP_Decision_Layer.ipynb  
```

3. **To run the CLASS Class classification pipeline**

Execute notebooks **in order**:

```
01_EDA.ipynb
04_CLASS_Baseline.ipynb  
05_CLASS_Complex.ipynb  

```
