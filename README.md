# Binary Classification with Logistic Regression

This project demonstrates a complete machine learning workflow for binary classification using logistic regression. The workflow includes data preprocessing, model training, evaluation, threshold tuning, and explanation of the sigmoid function.

## Project Structure

- [task4.py](cci:7://file:///d:/AI%20ML%20Internship/Day4/task4.py:0:0-0:0) — Main Python script containing all steps of the workflow.
- `data.csv` — Dataset used for binary classification (e.g., breast cancer dataset).
- `requirements.txt` — List of required Python packages.

## Steps Performed

1. **Load Dataset**  
   Reads a binary classification dataset from `data.csv`.

2. **Data Preprocessing**  
   - Handles categorical features with one-hot encoding.
   - Maps target labels to numeric values.
   - Removes columns and rows with missing values.

3. **Train/Test Split & Standardization**  
   Splits data into training and test sets and standardizes features.

4. **Model Training**  
   Trains a logistic regression model on the processed data.

5. **Evaluation**  
   Evaluates the model using:
   - Confusion matrix
   - Precision
   - Recall
   - ROC-AUC score
   - ROC curve plot

6. **Threshold Tuning**  
   Shows how changing the probability threshold affects classification metrics.

7. **Sigmoid Function Explanation**  
   Explains the role of the sigmoid function in logistic regression.

## How to Run

1. **Clone the repository:**
   ```sh
   git clone [https://github.com/BFMNINJA/ai-ml-internship-day4.git](https://github.com/BFMNINJA/ai-ml-internship-day4.git)
   cd ai-ml-internship-day4
