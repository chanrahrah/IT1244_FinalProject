1. Ensure you have the following libraries installed:
Use the following commands to install the required libraries:
a. pip install pandas scikit-learn matplotlib joblib imbalanced-learn
tensorflow==2.11.0
b. pip install tensorflow-cpu==2.11.0 --force-reinstall
2. Open Main Code File.ipynb in a suitable environment (e.g., Google
Colab or Jupyter Notebook).
3. Download the cleaned_data.csv file and note its directory path.
4. If you are not using Google Drive, skip the following steps in the code:
a. Section 2: For importing from Google Drive
b. Section 3: Load and Merge Datasets
5. Update the path to your cleaned_data.csv:
a. In Section 4 (Input your file dir that you stored the cleaned_data),
provide the correct directory path to your cleaned_data.csv.
b. Uncomment the relevant code to load the file.
6. Run the code sequentially:
a. Section 1: Importing Libraries
b. Section 4: Input your file dir that you stored the cleaned_data
c. Section 5: Preparing Data for Modeling
d. Section 6: Data Balancing
e. Section 7: Modeling using K-Nearest Neighbours, Logistic
Regression, and Random Forest Classifier
f. Section 8: Deep Learning Model (Feedforward Neural Network -
FNN)
g. Section 9: Plot Precision-Recall Curves for All Models
h. Section 10: Saving Final Trained Models
i. (Optional: You can comment out or delete the part that saves
the model to a directory if not needed).
i. Section 11: Evaluating on Final Trained Model with 20% of Test
Data (unseen during previous training)
7. Trained Models are available in the Models folder.
a. To import and use the trained models, follow the instructions in the
Model.py file inside the Models folder.
