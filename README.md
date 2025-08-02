# Student--marks-predictor
Mini project using python and linear regression 
# Step 1: Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Step 2: Load data
data = pd.read_csv("student_data.csv")  # CSV file path
print(data.head())

# Step 3: Plot data
plt.scatter(data['Hours'], data['Marks'], color='blue')
plt.xlabel("Hours Studied")
plt.ylabel("Marks Scored")
plt.title("Hours vs Marks")
plt.show()

# Step 4: Prepare data
X = data[['Hours']]  # Independent variable
y = data['Marks']    # Dependent variable

# Step 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Step 7: Prediction
hours = float(input("Enter study hours: "))
predicted_marks = model.predict([[hours]])
print(f"Predicted Marks: {predicted_marks[0]:.2f}")