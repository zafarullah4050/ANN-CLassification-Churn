ANN-CLassification-Churn
This project uses an Artificial Neural Network (ANN) built with TensorFlow/Keras to predict whether a bank customer will churn (i.e., leave the bank).

📁 Files

churn_ann_model.py – Main script for training the ANN model.

Churn_Modelling.csv – Dataset containing customer information.

model.h5 – Trained ANN model saved in HDF5 format.

scaler.pkl – StandardScaler object for feature normalization.

label_encoder_gender.pkl – LabelEncoder for encoding gender.

onehot_encoder_geo.pkl – OneHotEncoder for encoding geography.

logs/fit/ – Folder containing logs for TensorBoard visualization.

🛠 Requirements

Python 3.7+

TensorFlow

scikit-learn

pandas

numpy

pickle (standard library)

Install requirements:

pip install tensorflow scikit-learn pandas numpy

🚀 How to Run

Place the Churn_Modelling.csv file in the same folder as the script.

Run the script:

python churn_ann_model.py

It will:

Clean and preprocess the data

Encode categorical features

Scale numerical features

Build and train an ANN

Save the model and preprocessing tools

Log training process for TensorBoard

(Optional) To view training logs in TensorBoard:

tensorboard --logdir logs/fit

Then open the link in browser that it gives (usually http://localhost:6006)

📊 Features Used

Credit Score

Geography (France, Spain, Germany)

Gender (Male, Female)

Age

Tenure

Balance

Number of Products

Has Credit Card

Is Active Member

Estimated Salary

🧠 Model Architecture

Input Layer → Hidden Layer (64, ReLU)

Hidden Layer (32, ReLU)

Output Layer (1, Sigmoid)

Loss: Binary CrossentropyOptimizer: AdamMetric: Accuracy

✨ Output

model.h5 → can be loaded later for predictions

Use scaler.pkl, label_encoder_gender.pkl, and onehot_encoder_geo.pkl to transform input data before prediction

📬 Contact

For any queries or help, contact: Zafar Ullah

