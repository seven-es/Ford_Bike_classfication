# 🚴‍♂️ User Segmentation Using Bike Trip Data
---
This project analyzes bike-sharing data to segment users based on their behavior and trip characteristics. We use machine learning and deep learning models to classify users into meaningful categories to inform business decisions.

---

## 📁 Project Structure

The workflow is divided into the following parts:

1. **Import Libraries**  
   Load required libraries for data manipulation, visualization, machine learning, and deep learning.

2. **Preprocessing**  
   - Handle missing values  
   - Generate a new `user_segment` column based on `user_type` and trip duration  
   - Encode the target variable  
   - Extract features like hour and day of week from `start_time`  
   - Prepare feature set and scale inputs

3. **Train-Test Split & Scaling**  
   Split the dataset into training and test sets and scale the features using StandardScaler.

4. **K-Nearest Neighbors (KNN)**  
   - Perform hyperparameter tuning using `GridSearchCV`  
   - Train the best model  
   - Evaluate using accuracy, precision, recall, and confusion matrix

5. **Support Vector Machine (SVM)**  
   - Tune using linear and RBF kernels  
   - Use `GridSearchCV` with a smaller sample  
   - Evaluate final SVM model performance

6. **Deep Neural Network (DNN)**  
   - Manually define and test different architectures  
   - Apply dropout and early stopping  
   - Train and evaluate the best model

7. **Model Comparison**  
   - Compare KNN, SVM, and DNN side-by-side  
   - Display Accuracy, Precision, and Recall metrics  
   - Visualize results with bar charts

8. **Feature Importance**  
   - Analyze top features using permutation importance  
   - Compare feature contributions across models  
   - Visualize with bar plots

9. **User Segment Analysis**  
   - Summarize segment behavior (trip duration, hour, day)  
   - Use aggregation and bar plots to identify behavioral trends  
   - Map day numbers to weekday names

10. **Conclusion & Business Recommendations**  
    - Identify the best model  
    - Suggest strategic actions for each user segment:
      - Target *Potential Customers* with incentives
      - Optimize bike availability for *Regular Users*
      - Retain *Heavy Users* with loyalty perks

---

## 📊 Dataset

- File: `https://www.kaggle.com/datasets/nancyalaswad90/trip-data-ford-go-bike`
- Source: Bay Area bike-share trip data (February 2019)
- Fields used:
  - `user_type` (Subscriber/Customer/potential Customer)
  - `duration_sec`
  - `start_time`
  - `bike_id`, `start_station_id`, `end_station_id`

---

## 🎯 Target Variable

We engineered a new categorical variable: `user_segment`  
- `Regular User` – Subscriber with short trips  
- `Heavy User` – Subscriber with longer trips  
- `Potential Customer` – Non-subscriber  

Encoded as numeric labels for training.

---

## 🤖 Models Used

| Model           | Description                                 |
|-----------------|---------------------------------------------|
| **KNN (Tuned)** | GridSearchCV with different distance metrics |
| **SVM (Tuned)** | Linear and RBF kernels, small sample tuning  |
| **DNN (Tuned)** | Manual tuning of architecture, dropout, and learning rate |

---

## 📈 Evaluation Metrics

All models are evaluated using:

- Accuracy  
- Precision (Weighted)  
- Recall (Weighted)  
- Classification Report  
- Confusion Matrix  

---

## 🏆 Final Model Comparison

| Model           | Accuracy | Precision | Recall |
|-----------------|----------|-----------|--------|
| KNN (Tuned)     | 0.8280   | 0.8011    | 0.8280 |
| SVM (Tuned)     | 0.8887   | 0.7941    | 0.8887 |
| DNN (Tuned)     | 0.8883   | 0.8826    | 0.8883 |


---

## 🔍 Feature Importance

Top features contributing to user segmentation:

- `trip_duration`  
- `hour`  
- `start_station_id`  
- `dayofweek`  
- `end_station_id`


---

## 📊 Segment Behavior Summary

| User Segment        | Avg Duration | Median Duration | Min Duration | Max Duration | Avg Hour | Median Hour | Most Common Day |
|---------------------|--------------|------------------|--------------|--------------|----------|--------------|------------------|
| Heavy User          | 1109.11      | 853.0            | 600          | 74097        | 11.49    | 11.0         | Monday           |
| Potential Customer  | 1451.52      | 801.0            | 62           | 83519        | 11.43    | 11.0         | Monday           |
| Regular User        | 354.69       | 352.0            | 61           | 599          | 11.42    | 11.0         | Monday           |



---

## 💡 Business Recommendations

1. 🎁 Target *Potential Customers* with subscription incentives  
2. 🚲 Optimize bike availability for *Regular Users* based on time patterns  
3. 💎 Design loyalty offers for *Heavy Users* to increase retention  

---

## 🧪 How to Run

🐍 Make sure you have **Python 3.7 or newer** installed, then install all required packages:

1. **Clone this repository**

   Open your terminal and run:

   ```bash
   git clone https://github.com/seven-es/your-repo-name.git
   cd Ford_Bike_classfication

2. Install dependencies

   ```bash
   pip install -r requirements.txt
3. Install dependencies

   Place the dataset file fordgobike-tripdataFor201902_updated.csv in the root directory of the project

4. Run the script
   ```bash
    python bike_segmentation.py
