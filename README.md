# DA_AI_Project

## Analysing and Predicting Job Application Callback Rates and Resume Quality

This project is a Data Analytics and Artificial Intelligence mini project focused on analyzing resume characteristics, job attributes, and candidate information to predict job application callback rates and assess resume quality using Machine Learning techniques.

The project uses a real-world dataset containing 4,870 resumes and 30 features related to job advertisements, resume structure, candidate demographics, qualifications, and hiring outcomes.

---

## Project Objectives

### 1. Callback Prediction

Predict whether a candidate is likely to receive a job callback based on:

* Resume quality
* Work experience
* Skills and certifications
* Educational qualifications
* Job requirements
* Candidate demographics
* Employment history

### 2. Resume Quality Assessment

Classify resumes as high-quality or low-quality using structured resume features and candidate attributes.

---

## Dataset Information

* Total Records: 4,870
* Total Features: 30
* Target Variables:

  * `received_callback`
  * `resume_quality`

### Important Features

* years_experience
* college_degree
* computer_skills
* special_skills
* volunteer
* military
* employment_holes
* honors
* worked_during_school
* gender
* race
* job_city
* job_industry
* job_type

---

## Data Preprocessing

The following preprocessing techniques were applied:

* Missing value handling
* Feature encoding
* One-hot encoding
* Label encoding
* Feature scaling using StandardScaler
* Class balancing using SMOTE
* Feature selection based on EDA and correlations

---

## Exploratory Data Analysis (EDA)

Key findings from the analysis:

* 91.95% of applicants did not receive callbacks.
* Only 8.05% received callbacks.
* Resume quality distribution was nearly balanced:

  * High Quality: 2446
  * Low Quality: 2424
* 3504 applicants held a college degree.
* Computer skills were the most demanded skill across jobs.
* Federal contractor jobs required higher experience and education levels.
* Applicants from Boston showed higher callback rates than applicants from Chicago.
* White applicants had higher average callback rates than black applicants.
* Female applicants had slightly higher callback rates than male applicants.
* Honors, special skills, and years of experience positively influenced callback likelihood.

---

## Machine Learning Models Used

### 1. K-Nearest Neighbors (KNN)

* Used for resume quality classification
* Optimal k selected: 5
* Features standardized using StandardScaler
* Achieved high classification performance

### 2. Decision Tree Classifier

Used for callback prediction with 21 selected features.

#### Performance Metrics

* Accuracy: 90.56%
* Precision: 88.44%
* Recall: 93.65%
* F1 Score: 90.84%
* ROC AUC Score: 0.9093

#### Confusion Matrix

* True Negatives: 774
* False Positives: 111
* False Negatives: 58
* True Positives: 849

### 3. Logistic Regression

* Baseline and multi-feature models implemented
* SMOTE applied for handling class imbalance
* Demonstrated limitations on highly imbalanced callback data

### 4. K-Means Clustering (Exploratory)

* Explored for unsupervised learning
* Elbow method showed no meaningful clustering structure
* Not adopted for final implementation

### 5. Linear Regression (Exploratory)

* Explored for callback likelihood estimation
* Determined unsuitable for binary classification tasks

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Imbalanced-learn (SMOTE)
* Jupyter Notebook
* Power BI

---

## Files Included

| File Name                | Description                            |
| ------------------------ | -------------------------------------- |
| `app.py`                 | Main application file                  |
| `Final_ML_Project.ipynb` | Complete notebook implementation       |
| `requirements.txt`       | Python dependencies                    |
| `resume.csv`             | Dataset used for training and analysis |
| `projectdashboard.pbix`  | Power BI dashboard                     |
| `knn_model.pkl`          | Saved KNN model                        |
| `tree_model.pkl`         | Saved Decision Tree model              |
| `logistic_model.pkl`     | Saved Logistic Regression model        |

---

## Ethical Considerations

This project highlights potential biases in hiring patterns and focuses on identifying unfair trends rather than reinforcing them. The analysis aims to support fair, transparent, and data-driven hiring practices.

---

## Conclusion

This project demonstrates how Machine Learning and Data Analytics techniques can be used to:

* Analyze hiring patterns
* Predict callback probabilities
* Evaluate resume quality
* Understand feature importance in recruitment systems

Among the implemented models, supervised learning approaches such as Decision Trees and KNN produced the strongest results for classification tasks.
