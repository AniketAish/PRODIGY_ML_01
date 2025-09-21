import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# --- 1. DATA GENERATION ---
# This function creates a synthetic dataset for our model.
# In a real-world scenario, you would load this from a CSV, database, or API.
def generate_synthetic_data():
    """Generates a Pandas DataFrame with synthetic house data."""
    cities = ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata', 'Hyderabad', 'Pune', 'Ahmedabad']
    bhk_options = ['1 RK', '1 BHK', '2 BHK', '3 BHK', '4 BHK', '5 BHK']
    
    # Create a base for our data
    data = {
        'Location': np.random.choice(cities, 1000),
        'BHK_Type': np.random.choice(bhk_options, 1000)
    }
    df = pd.DataFrame(data)

    # --- Feature Engineering from BHK_Type ---
    bhk_map = {
        '1 RK': (1, 1), '1 BHK': (1, 1), '2 BHK': (2, 2),
        '3 BHK': (3, 3), '4 BHK': (4, 4), '5 BHK': (5, 5)
    }
    df['Bedrooms'], df['Bathrooms'] = zip(*df['BHK_Type'].map(bhk_map))

    # --- Generate plausible Square Footage ---
    sqft_base = {
        '1 RK': 300, '1 BHK': 550, '2 BHK': 800,
        '3 BHK': 1200, '4 BHK': 1800, '5 BHK': 2500
    }
    df['Sqft'] = df['BHK_Type'].map(sqft_base) + np.random.randint(-50, 150, size=1000)

    # --- Generate Price (The Target Variable) ---
    # Create a base price and add factors for location, size, etc.
    location_premium = {
        'Mumbai': 1.8, 'Delhi': 1.6, 'Bangalore': 1.5, 'Hyderabad': 1.3,
        'Pune': 1.2, 'Chennai': 1.1, 'Kolkata': 1.0, 'Ahmedabad': 0.9
    }
    
    # Base price calculation with some randomness
    base_price_per_sqft = 8000 # A base value in INR
    price = (df['Sqft'] * base_price_per_sqft * df['Location'].map(location_premium) + \
            df['Bedrooms'] * 500000 + \
            df['Bathrooms'] * 250000 + \
            np.random.normal(0, 500000, size=1000)) # Adding some noise
            
    df['Price_Lakhs'] = price / 100000 # Convert to Lakhs
    
    # Ensure no negative prices
    df['Price_Lakhs'] = df['Price_Lakhs'].clip(lower=10) # Minimum price of 10 Lakhs

    return df

# --- 2. MODEL TRAINING ---
# Use Streamlit's caching to avoid retraining the model on every interaction.
@st.cache_data
def train_model(df):
    """Preprocesses data and trains a linear regression model."""
    # One-Hot Encode the 'Location' column
    df_encoded = pd.get_dummies(df, columns=['Location'], drop_first=True)
    
    # Define features (X) and target (y)
    features = ['Sqft', 'Bedrooms', 'Bathrooms'] + [col for col in df_encoded.columns if 'Location_' in col]
    X = df_encoded[features]
    y = df_encoded['Price_Lakhs']
    
    # Create and train the model
    model = LinearRegression()
    model.fit(X, y)
    
    return model, X.columns.tolist() # Return the model and the feature column names

# --- Main Application ---
# Generate data and train the model
df = generate_synthetic_data()
model, feature_columns = train_model(df)

# --- 3. STREAMLIT GUI ---
st.set_page_config(page_title="House Price Predictor", page_icon="🏠")
st.title('🏠 Indian House Price Predictor')
st.write("Enter the details of the house to get an estimated price.")

# Create columns for layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Location Details")
    # Get unique sorted list of cities for the dropdown
    locations = sorted(df['Location'].unique())
    selected_location = st.selectbox('Select a City', options=locations)

    st.subheader("Flat Configuration")
    bhk_types = ['1 RK', '1 BHK', '2 BHK', '3 BHK', '4 BHK', '5 BHK']
    selected_bhk = st.selectbox('Select Flat Size (BHK)', options=bhk_types)
    
with col2:
    st.subheader("Area & Size")
    # Map BHK to a plausible default square footage for a better user experience
    default_sqft = {
        '1 RK': 350, '1 BHK': 600, '2 BHK': 900, '3 BHK': 1250, '4 BHK': 1800, '5 BHK': 2500
    }.get(selected_bhk, 1000)

    sqft_input = st.number_input('Enter Square Footage', min_value=200, max_value=10000, value=default_sqft, step=50)

# 'Predict' button to trigger the calculation
if st.button('Predict Price', type="primary", use_container_width=True):
    
    # --- Prediction Logic ---
    # 1. Map BHK selection to number of bedrooms and bathrooms
    bhk_map_predict = {
        '1 RK': (1, 1), '1 BHK': (1, 1), '2 BHK': (2, 2),
        '3 BHK': (3, 3), '4 BHK': (4, 4), '5 BHK': (5, 5)
    }
    bedrooms, bathrooms = bhk_map_predict[selected_bhk]

    # 2. Create an empty DataFrame with the correct feature columns
    input_data = pd.DataFrame(0, index=[0], columns=feature_columns)
    
    # 3. Populate the DataFrame with user inputs
    input_data['Sqft'] = sqft_input
    input_data['Bedrooms'] = bedrooms
    input_data['Bathrooms'] = bathrooms
    
    # Set the one-hot encoded location column to 1
    location_column = f'Location_{selected_location}'
    if location_column in input_data.columns:
        input_data[location_column] = 1

    # 4. Make the prediction
    try:
        predicted_price = model.predict(input_data)[0]
        st.success(f"##  estimada Preço: ₹ {predicted_price:.2f} Lakhs")
        st.info("This is an estimated price based on our predictive model. Market conditions may vary.")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

# Add a small section to show some of the raw data
st.markdown("---")
if st.checkbox('Show Sample Data'):
    st.write("Here is a small sample of the data used to train the model:")
    st.dataframe(df.head())