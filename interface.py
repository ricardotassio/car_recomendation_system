import streamlit as st
import pickle
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# Load the model and data
with open('car_recommendation_model.pkl', 'rb') as file:
    model_data = pickle.load(file)

knn_model = model_data['knn_model']
X = model_data['X']
data_model = model_data['data_model']
data_original = model_data['data_original']
fuel_dummies_columns = model_data['fuel_dummies_columns']
transmission_dummies_columns = model_data['transmission_dummies_columns']
min_price = model_data['min_price']
max_price = model_data['max_price']
unique_fuels = model_data['unique_fuels']
unique_transmissions = model_data['unique_transmissions']

# Define the recommendation function
def recommend_cars(price_min, price_max, fuel, transmission, n_neighbors=5):
    filtered_data_model = data_model[
        (data_model['price'] >= price_min) &
        (data_model['price'] <= price_max) &
        (data_original['fuel'] == fuel) &
        (data_original['transmission'] == transmission)
    ]

    filtered_data_original = data_original.loc[filtered_data_model.index]

    if filtered_data_model.empty:
        st.write("No cars found with the selected preferences.")
        return None
    else:
        filtered_data_model.reset_index(drop=True, inplace=True)
        filtered_data_original.reset_index(drop=True, inplace=True)

        filtered_data_for_knn = filtered_data_model.drop(columns=['price']).copy()
        user_features = pd.DataFrame(columns=filtered_data_for_knn.columns)

        numeric_cols = filtered_data_for_knn.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols:
            user_features.loc[0, col] = filtered_data_for_knn[col].mean()

        for col in fuel_dummies_columns:
            user_features.loc[0, col] = 1 if col == f"fuel_{fuel}" else 0

        for col in transmission_dummies_columns:
            user_features.loc[0, col] = 1 if col == f"transmission_{transmission}" else 0

        user_features.fillna(0, inplace=True)
        user_features = user_features[filtered_data_for_knn.columns]

        if len(filtered_data_for_knn) < n_neighbors:
            n_neighbors = len(filtered_data_for_knn)
            if n_neighbors == 0:
                st.write("No cars found after applying the filters.")
                return None

        knn_filtered = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
        knn_filtered.fit(filtered_data_for_knn)

        distances, indices = knn_filtered.kneighbors(user_features)

        recommended_indices = filtered_data_model.index[indices[0]]
        recommended_cars = filtered_data_original.loc[recommended_indices]

        output_columns = ['price', 'manufacturer', 'model', 'year', 'fuel', 'transmission', 'odometer', 'car_age']
        recommended_cars_display = recommended_cars[output_columns]

        return recommended_cars_display

# Streamlit Application
st.set_page_config(page_title="Car Recommendation System", layout="wide")
st.markdown(
    """
    <style>
        .header-container { 
            text-align: center; 
            padding: 2rem 0;
        }
        .metric-container {
            display: flex; 
            justify-content: space-between; 
            gap: 1rem; 
            margin-bottom: 2rem;
        }
        .metric-box {
            padding: 1rem; 
            background-color: #5A189A; 
            border-radius: 10px; 
            text-align: center; 
            font-size: 1.2rem;
            box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.1);
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            text-align: left;
            padding: 8px;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #5A189A;
            color: white;
        }
        tr:hover {
            background-color: rgba(224, 170, 255, 0.5); /* Hover color with transparency */
        }
    </style>
    """, unsafe_allow_html=True
)

# Page Header
st.markdown('<div class="header-container"><h1>Car Recommendation System</h1></div>', unsafe_allow_html=True)

# Dashboard Overview
st.markdown("### Overview")
with st.container():
    st.markdown(
        '<div class="metric-container"> \
            <div class="metric-box"> \
                <strong>Minimum Price:</strong> ${:,.2f} \
            </div> \
            <div class="metric-box"> \
                <strong>Maximum Price:</strong> ${:,.2f} \
            </div> \
            <div class="metric-box"> \
                <strong>Fuel Types:</strong> {} \
            </div> \
            <div class="metric-box"> \
                <strong>Transmissions:</strong> {} \
            </div> \
        </div>'.format(min_price, max_price, ", ".join(unique_fuels), ", ".join(unique_transmissions)), 
        unsafe_allow_html=True
    )

# Filters
st.markdown("### Filters")
col1, col2 = st.columns(2)

with col1:
    price_min_input, price_max_input = st.slider(
        "Select the price range ($)",
        min_value=int(min_price),
        max_value=int(max_price),
        value=(int(min_price), int(max_price)),
        step=1000
    )

with col2:
    fuel = st.selectbox("Fuel Type", unique_fuels)
    transmission = st.selectbox("Transmission", unique_transmissions)

# Recommend Button
st.markdown("### Recommendations")
if st.button("Recommend Cars"):
    recommendations = recommend_cars(price_min_input, price_max_input, fuel, transmission)
    if recommendations is not None and not recommendations.empty:
        st.write("### Cars recommended based on your preferences:")
        # Add table with icons
        st.markdown(
            recommendations.to_html(escape=False, index=False).replace(
                '<th>price</th>', '<th>💲 Price</th>'
            ).replace(
                '<th>manufacturer</th>', '<th>🏭 Manufacturer</th>'
            ).replace(
                '<th>model</th>', '<th>🚗 Model</th>'
            ).replace(
                '<th>year</th>', '<th>📅 Year</th>'
            ).replace(
                '<th>fuel</th>', '<th>⛽ Fuel</th>'
            ).replace(
                '<th>transmission</th>', '<th>⚙️ Transmission</th>'
            ).replace(
                '<th>odometer</th>', '<th>🛣️ Odometer</th>'
            ).replace(
                '<th>car_age</th>', '<th>📈 Car Age</th>'
            ),
            unsafe_allow_html=True
        )
    else:
        st.write("No cars found with the selected preferences.")
