# Used Cars Recommendation System

## Introduction

This project is a **Used Cars Recommendation System** that leverages machine learning to suggest vehicles to users based on their preferences, such as price range, fuel type, and transmission. The system uses a **K-Nearest Neighbors (KNN)** algorithm and is integrated with a **Streamlit-based user interface** for ease of interaction.

## Objectives

- Help users select cars that match their preferences and budgets.
- Provide alternative suggestions when no exact matches are found.
- Enhance recommendations over time as more user interactions are recorded.

## Dataset

The system uses the [Craigslist Cars and Trucks dataset](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data) available on Kaggle. This dataset includes extensive details about used cars listed on Craigslist.

## Features

- **Data Cleaning and Preparation**: Handles missing values, filters out irrelevant records, and prepares the dataset for modeling.
- **Feature Engineering**: Creates derived features such as car age and price per mile.
- **Recommendation Algorithm**: Utilizes KNN to recommend cars based on user preferences.
- **Interactive Interface**: Built with **Streamlit**, allowing users to input their preferences and view recommendations dynamically.

## How It Works

### 1. Data Preparation

- The dataset is cleaned to remove missing and irrelevant data.
- Categorical variables are encoded using **One-Hot Encoding** and numerical encoding.

### 2. Recommendation Engine

- A **KNN model** is trained on the processed dataset.
- User preferences such as price range, fuel type, and transmission are used to filter and recommend cars.

### 3. User Interface

- The **Streamlit interface** allows users to:
  - Specify preferences.
  - View recommended cars in a clean, interactive format.

### 4. Model Export

- The trained KNN model and supporting data are exported as a Pickle file (`car_recommendation_model.pkl`) for efficient reuse.

## Installation and Usage

### Requirements

- Python 3.8+
- Libraries: pandas, scikit-learn, Streamlit, matplotlib, seaborn, pickle

### Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the System

1. Download the dataset from Kaggle and place it in the project directory as `vehicles.csv`.
2. Run the Streamlit app:
   ```bash
   streamlit run interface.py
   ```
3. Open the local Streamlit URL in your browser (default: `http://localhost:8501`).

### Dataset Preprocessing

The dataset (`vehicles.csv`) is preprocessed in multiple steps to handle missing values, encode categorical variables, and engineer new features. This ensures the recommendation engine receives high-quality input data.

## Example Usage

- Input:
  - **Price Range**: $10,000–$40,000
  - **Fuel Type**: Gas
  - **Transmission**: Manual
- Output:
  A list of recommended cars that meet the specified criteria, including details like price, manufacturer, model, year, and odometer reading.

## Model Export

The trained KNN model and associated metadata are exported to `car_recommendation_model.pkl` for deployment and scalability.

## Contributions

Feel free to contribute by:

- Reporting issues.
- Suggesting improvements.
- Submitting pull requests.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

```

```
