from flask import Flask, request, jsonify
import pandas as pd
import mlflow.pyfunc
import logging

# Initialize Flask app
app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)  # Set the logging level
logger = logging.getLogger(__name__)  # Create a logger

# Load the model
model_path = "mlruns/0/b243a41e68d04244a4781756d0f18fb5/artifacts/iris_model"  # Replace <RUN_ID> with your actual RUN_ID
model = mlflow.pyfunc.load_model(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()  # Get JSON data from the request
        
        # Log received data
        logger.info(f"Received data: {data}")

        # Validate that 'data' and 'columns' are present
        if 'data' not in data or 'columns' not in data:
            logger.error('Invalid input format. Ensure you send both "columns" and "data".')
            return jsonify({'error': 'Invalid input format. Ensure you send both "columns" and "data".'})

        # Check if all input arrays have the same length
        if len(data['columns']) != len(data['data'][0]):
            logger.error('Mismatch between number of columns and data points.')
            return jsonify({'error': 'Mismatch between number of columns and data points.'})

        # Create DataFrame from input data
        input_data = pd.DataFrame(data['data'], columns=data['columns'])
        
        # Make predictions
        predictions = model.predict(input_data)
        
        # Log predictions
        logger.info(f"Predictions: {predictions.tolist()}")
        
        return jsonify({'predictions': predictions.tolist()})

    except Exception as e:
        logger.exception("An error occurred during prediction.")  # Log the exception
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(port=5001, debug=True)
