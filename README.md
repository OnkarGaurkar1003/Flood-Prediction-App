 Flood Prediction App

This Flood Prediction App leverages advanced machine learning techniques to provide real-time flood risk predictions. Designed with a hybrid model combining LSTM (Long Short-Term Memory) and RNN (Recurrent Neural Network) layers, this app can capture sequential dependencies in weather and environmental data to predict flood likelihood accurately. It integrates a user-friendly Streamlit interface to allow users to input location-specific data or pull real-time weather data from the OpenWeatherMap API, making it suitable for a wide range of applications, from local predictions to broader regional assessments.

 Key Features:
- **Hybrid Model Architecture**: Combines LSTM and RNN layers to capture both short-term and long-term dependencies in sequential data, enhancing prediction accuracy.
- **Real-Time Data Integration**: Allows users to fetch current weather data such as temperature, humidity, wind speed, and rainfall for dynamic flood risk analysis.
- **Customizable Location-Based Predictions**: Users can select from predefined locations or input custom data for tailored predictions.
- **Visual Insights**: Displays key metrics like accuracy, sensitivity, specificity, and F1-score for model performance.
  
 Technology Stack:
- **Machine Learning**: TensorFlow/Keras for LSTM and RNN model implementation.
- **Data Processing**: StandardScaler for feature scaling and data normalization.
- **API Integration**: OpenWeatherMap API for fetching live weather data.
- **Front-End**: Streamlit for creating an intuitive and interactive user interface.

This app is built to help communities, organizations, and individuals make informed decisions by anticipating flood risks based on live data, potentially mitigating the impact of floods through timely information and action.
