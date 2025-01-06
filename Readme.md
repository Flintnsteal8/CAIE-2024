# Identifying High-Value Repeat Buyers for Olist's E-commerce Platform (CAIE tech test 2024)
## Background
Olist is a Brazilian e-commerce marketplace like Lazada, Taobao and Shopee, it is a sales platform that connects small retailers with customers.

## Objective
Main Objective: To identify potential repeat buyers 

Full Name: Varun Sathish Nair
Email address: 222992b@mymail.nyp.edu.sg

Folder structure:
├── src
        │   ├── dataprocessing
        │   └── datamodelelling
├── saved_model
            └── logreg_model.pkl
            └── rf_model.pkl
            └── xgb_model.pkl
├── README.md
├── eda.ipynb
├── eda.pdf
├── requirements.txt
└── run.sh

Programming Language used: Python 3.7.0
List of libraries used:
    pandas
    matplotlib
    seaborn
    scikit-learn
    plotly
    XGBoost


#### Feature Engineering
- New column delivery duration made to calculate days
- on time column made to see if deliveries were infact on time or not
- Binarized repeat buyers
- converted date columns to date format for better analysis
- no need to do sentiment analysis on review message since its captured by review score

#### Key takeaways
- Review score dosent affect repeat buyer but delivery time and payment does have a minimal impact
- Delivery time has an effect since its shown that repeat buyers happen when delivery is on time or faster.
- General trend is that money spent is mostly more than total order for the month.
- Corr matrix shows product dimensions could have an effect on repeat buyers

#### Feature selection
- Plan is to use on_time, delivery time and payment_value as some features for now including high correlated features like payment_value, product weight ,height, width and length. might consider RFM analysis in the ml pipeline.

After careful consideration here is why binarization is better than RFM.

Simplicity and Interpretability:
Binarization:

   Simplified Problem: Converting the problem to binary classification (repeat buyer: yes/no) makes it more straightforward, enabling the model to focus on a single outcome. This enhances the interpretability and understanding of the model's predictions.

   Straightforward Analysis: Binary classification offers a clear and direct interpretation of results, simplifying the analysis process compared to handling multiple dimensions.

   RFM:

   Multidimensional Complexity: Using Recency, Frequency, and Monetary (RFM) values introduces complexity by adding multiple dimensions. Each dimension must be considered and weighted correctly, complicating both the modeling process and the interpretation of results.

2.Performance in Imbalanced Datasets:

   Binarization:

   Established Techniques: Well-established techniques for handling class imbalance (such as oversampling, undersampling, and class weights) are readily applicable to binary classification problems. These methods can enhance model performance when repeat buyers constitute a small portion of the dataset.
   Focused Improvement: Binary classification allows for targeted improvements in model performance through imbalance handling techniques.

   RFM:

   Dependence on Model Capability: The performance of models using RFM features in imbalanced datasets relies on the model's ability to learn complex relationships between features and the target variable. This can be challenging and less straightforward compared to binary classification.



To execute the pipeline:
1. Create VENV with python version 3.7 (inside the folder) 
2. Activate the VENV
3. download libraries using the requirements file
4. move the data csv files to the same folder () 
5. Use GitBash terminal and run (bash ./run.sh)

Logical steps of the pipeline:
    1. Data Preprocessing
        a. Feature Engineering, aggregation and dupe removal
        b. label Encoder
        c. Outlier removal and MinMaxScaler
    2. Data Modelling
        a. train_test_split with validation split
        b. Tuning parameters for XGBoost and Random Forest Classifier
        c. train using Logistic regression and RFclassifier
        d. Save Models
        e. Output Classification reports for validation and test

ML model: Undersampling of majority class is used. Experimented with oversampling techniques like smote but the results retured showed unpromising results especially with recall metric.



#### Choice of Models

The target column indicates whether a customer is a repeat buyer, making this a binary classification task. The following models were chosen:

1. **Logistic Regression**
    1. Serves as a good baseline for binary classification due to its simplicity.
    2. Easy to implement with minimal hyperparameter tuning and computational efficiency.
    3. Provides interpretable coefficients for each feature, aiding in model explainability.

2. **Random Forest**
    1. Utilizes multiple decision trees to capture complex non-linear relationships, this is crucial given the low correlation coefficients.
    2. Its ensemble nature reduces overfitting by averaging the predictions of multiple trees.
    3. Provides insights on feature importance, enhancing fine-tuning and interpretability.

3. **XGBoost**
    1. Includes regularization techniques like L1 and L2 to prevent overfitting.
    2. Known for high accuracy and often outperforms other machine learning algorithms.
    3. Famous for its computational efficiency, handling large datasets with good execution speed.

#### Classification reports

Test Report for RandomForestClassifier:
              precision    recall  f1-score   support

           0       0.96      0.59      0.73      7980
           1       0.09      0.60      0.15       514

    accuracy                           0.59      8494
   macro avg       0.52      0.60      0.44      8494
weighted avg       0.91      0.59      0.70      8494

Confusion Matrix (Test Set):
[[4718 3262]
 [ 205  309]]
Accuracy Score (Test Set): 0.5918295267247469
ROC-AUC Score (Test Set): 0.6208662463551876

Test Report for LogisticRegression:
              precision    recall  f1-score   support

           0       0.95      0.50      0.65      7980
           1       0.07      0.55      0.12       514

    accuracy                           0.50      8494
   macro avg       0.51      0.52      0.39      8494
weighted avg       0.89      0.50      0.62      8494

Confusion Matrix (Test Set):
[[3995 3985]
 [ 232  282]]
Accuracy Score (Test Set): 0.5035319048740288
ROC-AUC Score (Test Set): 0.5378012151000069

Test Report for XGBClassifier:
              precision    recall  f1-score   support

           0       0.95      0.58      0.72      7980
           1       0.08      0.54      0.14       514

    accuracy                           0.58      8494
   macro avg       0.51      0.56      0.43      8494
weighted avg       0.90      0.58      0.69      8494

Confusion Matrix (Test Set):
[[4629 3351]
 [ 234  280]]
Accuracy Score (Test Set): 0.5779373675535672
ROC-AUC Score (Test Set): 0.600127629384746


#### Metrics used in evaluation
    a. Precision: Percentage of correct positive predictions relative to total positive predictions.
    b. Recall: Percentage of correct positive predictions relative to total actual positives.
    c. F1 Score: A weighted harmonic mean of precision and recall. The closer to 1, the better the model.
    d. ROC-AUC: ROC AUC score is the area under the ROC curve. It sums up how well a model can produce relative scores to discriminate 
      between positive or negative instances across all classification thresholds.

The most important metrics are recall and roc-auc . Here’s why:

#### Importance of recall and ROC-AUC Score
Recall Focus: Recall is crucial as it measures the model's ability to identify all relevant instances of the minority class, providing a comprehensive measure of the model’s performance on detecting repeat buyers.

Handling Class Imbalance: Recall, particularly for the minority class, highlights the model's effectiveness in correctly identifying true positives, offering a more accurate assessment of model performance across classes.

Robustness of the Model: A higher recall often denotes better generalization and performance consistency across various datasets or samples. The ROC-AUC score further supports this by indicating the model’s ability to distinguish between classes across different thresholds.


Across all 3 models, it seems that randomforest performs the best based on the metrics recall and roc-auc.

RandomForestClassifier can handle imbalanced data more effectively compared to LogisticRegression and XGBClassifier due to several key characteristics and techniques inherent to the algorithm:

1. Ensemble Learning
Bagging: RandomForest is an ensemble method that relies on bagging (Bootstrap Aggregating), which helps in reducing variance and improving the generalization of the model. Each decision tree in the forest is trained on a different bootstrap sample, which introduces variability and helps the model to be less sensitive to the class imbalance.

2. Decision Trees Tree-Based Splitting: 

The data is split by decision trees, the RandomForest's base learners, according to the feature values that maximize the improvement of the class purity at each node. This implies that even examples from minority classes have an opportunity to impact the splits if they can improve the split quality significantly.

3. Class Weights Weighted Sampling: 
The class_weight parameter allows RandomForest to specify the various weights that should be applied to the various classes. The model ensures that minority classes have a proportionately stronger influence during training by adjusting weights inversely proportional to class frequencies when class_weight='balanced' is specified.

4. Adaptability in Data Distribution Handling
Handling Non-Linear correlations: Compared to Logistic Regression, which is essentially linear, RandomForest is better able to capture complicated, non-linear correlations in the data. This flexibility allows RandomForest to model imbalanced data more effectively by capturing subtle patterns that might be missed by a linear model.