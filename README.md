# 🏠 Indian House Price Predictor

A simple and interactive web application built with Streamlit to predict house prices in major Indian cities. This project uses a Linear Regression model trained on synthetically generated data to provide price estimations.

## ✨ Features

  * **Interactive UI**: A clean and user-friendly interface to input house details.
  * **Dynamic Predictions**: Get real-time price estimates based on your inputs.
  * **Multiple Cities**: Supports price prediction for major Indian cities like Mumbai, Delhi, Bangalore, and more.
  * **Synthetic Data**: The model is trained on a synthetically generated dataset, making the project self-contained and easy to run without external CSV files.
  * **Data Transparency**: Includes an option to view a sample of the data used for training the model.

-----

## 🛠️ Technology Stack

This project is built using the following technologies:

  * **Python**: The core programming language.
  * **Streamlit**: For creating and serving the web application.
  * **Pandas**: For data manipulation and handling.
  * **NumPy**: For numerical operations and data generation.
  * **Scikit-learn**: For training the Linear Regression model.

-----

## ⚙️ How It Works

The application's logic is broken down into three main parts:

1.  **Data Generation**: The `generate_synthetic_data()` function programmatically creates a Pandas DataFrame with 1000 sample entries. It generates plausible data for features like `Location`, `BHK_Type`, and `Sqft`, and calculates a `Price_Lakhs` based on these features with some added randomness.
2.  **Model Training**: The `train_model()` function preprocesses the data (using one-hot encoding for the `Location` feature) and trains a `LinearRegression` model. The Streamlit decorator `@st.cache_data` is used to cache the trained model, ensuring it doesn't retrain on every user interaction, which significantly speeds up the app.
3.  **Prediction Interface**: The Streamlit GUI collects user inputs (City, BHK, Square Footage). When the user clicks the "Predict Price" button, the inputs are converted into a format the model understands (a one-hot encoded DataFrame row) and fed to the trained model to generate a price prediction.

-----

## 🚀 Getting Started

To run this project on your local machine, please follow the steps below.

### Prerequisites

  * Python 3.8 or higher
  * `pip` (Python package installer)

### Installation & Setup

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/AniketAish/PRODIGY_ML_01.git
    cd main.py
    ```

2.  **Create a virtual environment (recommended):**

      * **On macOS/Linux:**
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```
      * **On Windows:**
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```

3.  **Install the required packages:**
    Create a file named `requirements.txt` in the project directory and add the following lines to it:

    ```
    streamlit
    pandas
    numpy
    scikit-learn
    ```

    Now, install them using pip:

    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit application:**

    ```bash
    streamlit run main.py
    ```

    Your web browser should automatically open to the application's URL (usually `http://localhost:8501`).

-----

## 📋 How to Use

Once the application is running:

1.  **Select a City** from the dropdown menu.
2.  **Choose the Flat Size (BHK)** from the options.
3.  **Enter the Square Footage** of the property. The input field suggests a default value based on the selected BHK type.
4.  Click the **"Predict Price"** button.
5.  The estimated price will be displayed on the screen.

-----

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
