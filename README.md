# AI-Based Industrial Maintenance and Process Optimization For Smart Manufacturing

## 📌 Project Overview

**AI-Based Industrial Maintenance and Process Optimization For Smart Manufacturing** is a smart manufacturing project that uses **Artificial Intelligence and Machine Learning For Smart Manufacturing** to predict machine failures, monitor machine health, analyze industrial processes, and support predictive maintenance.

The project is designed to help industries move from **traditional reactive maintenance** to **AI-driven predictive maintenance**, where potential machine failures can be identified before they cause major production interruptions.

The system uses machine sensor information such as:

* Air Temperature
* Process Temperature
* Rotational Speed
* Torque
* Tool Wear
* Machine Type
* Failure-related indicators

The project also provides an interactive **Streamlit dashboard** for machine monitoring, model analysis, process intelligence, and an AI-based assistant.

---

## 🎯 Objectives

The main objectives of this project are:

1. To analyze industrial machine sensor data.
2. To identify patterns associated with machine failures.
3. To preprocess and prepare industrial data for machine learning.
4. To train and compare multiple machine learning algorithms.
5. To select the best-performing model for machine failure prediction.
6. To provide predictive maintenance insights through an interactive dashboard.
7. To visualize machine health, risk levels, and process information.
8. To provide an AI assistant for interacting with machine and maintenance information.
9. To reduce unexpected machine downtime and improve maintenance planning.
10. To demonstrate the application of AI in smart manufacturing.

---

## 🏭 Problem Statement

Unexpected machine failures can result in:

* Production downtime
* Increased maintenance costs
* Equipment damage
* Reduced productivity
* Delayed manufacturing operations

Traditional maintenance approaches often depend on fixed schedules or maintenance after a failure occurs.

This project addresses the problem by using **machine learning-based predictive maintenance**, where available machine data is analyzed to estimate the possibility of machine failure and support timely maintenance decisions.

---

## 💡 Proposed Solution

The proposed system follows a complete machine-learning workflow:

```text
Industrial Dataset
        ↓
Data Exploration
        ↓
Data Cleaning & Preprocessing
        ↓
Feature Transformation
        ↓
Train-Test Split
        ↓
Feature Scaling
        ↓
Machine Learning Models
        ↓
Model Comparison
        ↓
Best Model Selection
        ↓
Machine Failure Prediction
        ↓
Interactive Streamlit Dashboard
        ↓
Maintenance & Process Insights
```

---

## 📊 Dataset

The project uses the **AI4I 2020 Predictive Maintenance Dataset**.

The dataset contains industrial machine information and sensor measurements that can be used to study machine failure patterns.

Important features include:

| Feature                   | Description                                |
| ------------------------- | ------------------------------------------ |
| `Type`                    | Machine/product type                       |
| `Air temperature [K]`     | Air temperature                            |
| `Process temperature [K]` | Process temperature                        |
| `Rotational speed [rpm]`  | Machine rotational speed                   |
| `Torque [Nm]`             | Machine torque                             |
| `Tool wear [min]`         | Tool usage/wear                            |
| `TWF`                     | Tool Wear Failure indicator                |
| `HDF`                     | Heat Dissipation Failure indicator         |
| `PWF`                     | Power Failure indicator                    |
| `OSF`                     | Overstrain Failure indicator               |
| `RNF`                     | Random Failure indicator                   |
| `Machine failure`         | Target variable indicating machine failure |

The dataset is stored in:

```text
dataset/
└── ai4i2020.csv
```

---

## 🤖 Machine Learning Models

The project compares three machine learning algorithms:

### 1. Logistic Regression

A classification algorithm used as a baseline model for predicting whether a machine failure will occur.

### 2. Random Forest

An ensemble learning algorithm that combines multiple decision trees to improve classification performance and robustness.

### 3. XGBoost

A powerful gradient boosting algorithm used for classification and model comparison.

The models are trained and evaluated using accuracy.

The best-performing model is automatically selected and saved for later use.

---

## ⚙️ Data Preprocessing

The preprocessing pipeline includes:

1. Loading the AI4I 2020 dataset.
2. Removing unnecessary columns such as `UDI` and `Product ID`.
3. Converting the machine `Type` values into numerical values.
4. Separating features and the target variable.
5. Splitting the dataset into training and testing sets.
6. Applying `StandardScaler` for feature scaling.
7. Preparing the processed data for machine learning.

The preprocessing implementation is available in:

```text
utils/preprocessing.py
```

---

## 🔍 Exploratory Data Analysis

The EDA module performs initial analysis of the dataset, including:

* Displaying sample records
* Checking dataset dimensions
* Checking column names
* Identifying missing values
* Checking duplicate records
* Displaying dataset information

The EDA code is available in:

```text
notebooks/eda.py
```

---

## 📈 Interactive Dashboard

The project includes an interactive **Streamlit dashboard** for predictive maintenance and process intelligence.

The dashboard contains the following major sections:

### 1. Dashboard & Digital Twin

Provides a machine fleet monitoring interface with:

* Machine health status
* Risk levels
* Sensor information
* Machine monitoring
* Predictive maintenance indicators

### 2. ML Model Performance

Provides information related to:

* Machine learning model performance
* Model comparison
* Feature importance
* Dataset insights

### 3. Process Intelligence

Provides process-level information such as:

* Failure-type analysis
* Failure trends
* Machine/process insights

### 4. AI Assistant

The project includes an AI assistant using the **Groq API** to interact with machine-related information and provide intelligent responses.

### 5. About & Team

Provides information about the project and team members.

---

## 🔐 Authentication

The dashboard includes authentication functionality using **Streamlit Authenticator**.

Users can log in through the dashboard before accessing the application.

The authentication configuration is generated using:

```text
dashboard/generate_config.py
```

Before deployment, default credentials and secret configuration values should be changed.

**Important:** Never expose real passwords, API keys, or secret keys in a public GitHub repository.

---

## 🗂️ Project Structure

```text
Project Work/
│
├── dashboard/
│   ├── app.py
│   ├── auth_config.yaml
│   └── generate_config.py
│
├── dataset/
│   └── ai4i2020.csv
│
├── models/
│   ├── __init__.py
│   └── train_model.py
│
├── notebooks/
│   └── eda.py
│
└── utils/
    ├── __init__.py
    └── preprocessing.py
```

---

## 🛠️ Technologies Used

### Programming Language

* Python

### Machine Learning

* Scikit-learn
* XGBoost

### Data Processing

* Pandas
* NumPy

### Data Visualization

* Plotly

### Dashboard

* Streamlit

### Authentication

* Streamlit Authenticator
* PyYAML

### AI Assistant

* Groq API

### Model Storage

* Joblib

### Version Control

* Git
* GitHub

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/Priyanka-HC/AI-Based-Industrial-Maintenance-and-Process-Optimization-.git
```

Navigate to the project folder:

```bash
cd AI-Based-Industrial-Maintenance-and-Process-Optimization-/Final-Project
```

Install the required Python packages:

```bash
pip install pandas numpy scikit-learn xgboost joblib streamlit plotly python-dotenv groq pyyaml streamlit-authenticator
```

---

## 🚀 Running the Project

### Step 1: Perform Exploratory Data Analysis

From the `Project Work` directory:

```bash
python notebooks/eda.py
```

### Step 2: Train the Machine Learning Models

```bash
python models/train_model.py
```

The training script compares:

```text
Logistic Regression
Random Forest
XGBoost
```

The best-performing model and scaler are saved for dashboard usage.

### Step 3: Configure Authentication

Navigate to the dashboard directory:

```bash
cd dashboard
```

Run:

```bash
python generate_config.py
```

Update the generated configuration with the required user information and secure credentials.

### Step 4: Configure the AI Assistant

Create a `.env` file and add your Groq API key:

```text
GROQ_API_KEY=your_api_key_here
```

Do not upload your real API key to GitHub.

### Step 5: Start the Dashboard

From the `dashboard` directory:

```bash
streamlit run app.py
```

The application will open in your web browser.

---

## 📌 Expected Outcome

The completed system provides an AI-powered environment for:

* Predicting machine failures
* Monitoring machine health
* Understanding machine risk levels
* Comparing machine learning models
* Analyzing industrial failure patterns
* Supporting predictive maintenance decisions
* Visualizing industrial process information
* Interacting with an AI-based maintenance assistant

---

## 🌟 Key Benefits

* Early identification of potential machine failures
* Improved maintenance planning
* Reduced unexpected downtime
* Better understanding of machine health
* Data-driven maintenance decisions
* Interactive visualization of industrial data
* Machine learning model comparison
* AI-assisted industrial process analysis

---

## 🔮 Future Enhancements

The project can be further improved by:

* Connecting the dashboard to real-time IoT sensor data.
* Implementing real-time machine failure alerts.
* Adding automated maintenance recommendations.
* Improving model evaluation using precision, recall, F1-score, and ROC-AUC.
* Implementing advanced anomaly detection.
* Adding time-series machine monitoring.
* Deploying the dashboard to a cloud platform.
* Improving authentication and user-role management.
* Adding database integration for historical machine records.
* Retraining models periodically using new machine data.

---

## 📄 License

This project is licensed under the **MIT License**.

See the [`LICENSE`](../LICENSE) file for more information.

---

## ⭐ Conclusion

The **AI-Based Industrial Maintenance and Process Optimization For Smart Manufacturing** project demonstrates how Artificial Intelligence and Machine Learning can be applied to smart manufacturing environments.

By combining industrial sensor data, data preprocessing, machine learning, predictive maintenance, process intelligence, interactive visualization, and an AI assistant, the system provides a foundation for making **data-driven maintenance decisions and improving industrial operational efficiency**.
