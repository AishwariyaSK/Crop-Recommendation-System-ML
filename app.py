import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load trained model and preprocessing encoders
model = joblib.load('models/RandomForest.pkl')
label_encoder_crops = joblib.load('models/CROPS_label_encoder.pkl')
label_encoder_district = joblib.load('models/DISTRICT_label_encoder.pkl')
label_encoder_crop_type = joblib.load('models/TYPE_OF_CROP_label_encoder.pkl')
scaler = joblib.load('models/minmax_scaler.pkl')
mlb = joblib.load('models/mlb.pkl')

# Crop Type and Crop Mapping
crop_type_mapping = {
    "Root&tuber": ["Carrot", "Beetroot", "Radish", "Sweet Potato", "Tapioca", "Elephant Foot Yam"],
    "Cereals": ["Rice", "Wheat"],
    "Fibre Crop": ["Jute", "Cotton"],
    "Millets": ["Maize", "Ragi", "Samai", "Thinai", "Kudiraivali", "Varagu"],
    "Minor Vegetables": ["Asparagus", "Basella", "Lettuce", "Mint", "Palak", "Turnip"],
    "Oil Seeds": ["Groundnut", "Gingely", "Safflower", "Castor", "Sunflower"],
    "Pulses": ["Blackgram", "Greengram", "Redgram", "Soyabean"],
    "Sugar Crops": ["Sugarcane", "Sweet Sorghum"],
    "Vegetables": [
        "Tomato", "Onion", "Chillies", "Brinjal", "Capsicum", "Paprika", "Pumpkin",
        "Snake Gourd", "Ribbed Gourd", "Bottle Gourd", "Bitter Gourd", "Ash Gourd",
        "Cucumber", "Watermelon", "Muskmelon", "Chowchow", "Cluster Bean", "Peas"
    ]
}

# Available Crop Images (You should store images in 'crop_images/' folder)
# crop_images = {
#     "Rice": "crop_images/rice.jpg",
#     "Wheat": "crop_images/wheat.jpg",
#     "Maize": "crop_images/maize.jpg",
#     "Sugarcane": "crop_images/sugarcane.jpg",
#     "Tomato": "crop_images/tomato.jpg",
#     "Onion": "crop_images/onion.jpg",
#     "Brinjal": "crop_images/brinjal.jpg",
#     "Pumpkin": "crop_images/pumpkin.jpg",
#     "Capsicum": "crop_images/capsicum.jpg",
#     "Soyabean": "crop_images/soyabean.jpg",
#     # Add more crop images...
# }

# Streamlit UI
st.title("🌱 Smart Crop Recommendation System")

# Display Crop Type and Crop Mapping Table
st.subheader("📌 Crop Type Mapping")
crop_df = pd.DataFrame(list(crop_type_mapping.items()), columns=["Crop Type", "Crops"])
st.dataframe(crop_df)

st.sidebar.header("📝 Enter Details")

# Categorical Inputs (Dropdowns)
district = st.sidebar.selectbox("Select District", list(label_encoder_district.classes_))
crop_type = st.sidebar.selectbox("Select Crop Type", list(label_encoder_crop_type.classes_))

# Numerical Inputs (Sliders)
crop_duration = st.sidebar.slider("Crop Duration (Days)", min_value=20, max_value=700, value=120)
soil_ph = st.sidebar.slider("Soil pH Level", min_value=4.0, max_value=9.0, value=6.5)
temp_avg = st.sidebar.slider("Average Temperature (°C)", min_value=10.0, max_value=50.0, value=25.0)
water_required = st.sidebar.slider("Water Required (mm)", min_value=300.0, max_value=2440.0, value=200.0)
humidity_avg = st.sidebar.slider("Relative Humidity (%)", min_value=10.0, max_value=100.0, value=60.0)

# Checkbox Inputs (Soil Types)
soil_types = list(mlb.classes_)  # Get the soil type labels from MultiLabelBinarizer
selected_soils = [soil for soil in soil_types if st.sidebar.checkbox(soil, value=False)]

# Predict Button
if st.sidebar.button("🔍 Predict"):
    # Encode categorical variables
    district_encoded = label_encoder_district.transform([district])[0]
    crop_type_encoded = label_encoder_crop_type.transform([crop_type])[0]

    # Scale numerical variables
    numerical_features = np.array([[crop_duration, soil_ph, temp_avg, water_required, humidity_avg]])
    numerical_features_scaled = scaler.transform(numerical_features)[0]

    # Encode soil types using MultiLabelBinarizer
    soil_types_encoded = mlb.transform([selected_soils])[0]

    # Combine all features
    input_features = np.hstack((district_encoded, crop_type_encoded, numerical_features_scaled, soil_types_encoded)).reshape(1, -1)

    # Predict top 5 recommended crops
    probabilities = model.predict_proba(input_features)[0]
    top_n = 5  # Number of crops to display
    top_indices = np.argsort(probabilities)[-top_n:][::-1]
    top_crops = label_encoder_crops.inverse_transform(top_indices)
    top_scores = probabilities[top_indices]

    # Display results
    st.subheader("✅ Recommended Crops:")
    
    highest_confidence_crop = top_crops[0]  # Crop with highest confidence
    highest_confidence_score = top_scores[0]  # Highest confidence score
    
    for crop, score in zip(top_crops, top_scores):
        st.write(f"🌾 {crop}: {score*100:.2f}% Confidence")
        st.progress(float(score))  # Progress bar for confidence score

    # Display Image of the Top Recommended Crop
    # st.subheader("🌿 Most Recommended Crop")
    # if highest_confidence_crop in crop_images:
    #     st.image(crop_images[highest_confidence_crop], caption=highest_confidence_crop, use_column_width=True)
    # else:
    #     st.write(f"📌 No image available for {highest_confidence_crop}.")
