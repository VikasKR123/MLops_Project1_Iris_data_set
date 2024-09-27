from flask import Flask, request, jsonify
import pandas as pd
import mlflow.pyfunc
import logging

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG)

model_path = "mlruns/0/b243a41e68d04244a4781756d0f18fb5/artifacts/iris_model"
model = mlflow.pyfunc.load_model(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()  # Get JSON data from the request
        
        # Log received data
        app.logger.debug(f"Received data: {data}")

        # Validate that 'data' and 'columns' are present
        if 'data' not in data or 'columns' not in data:
            return jsonify({'error': 'Invalid input format. Ensure you send both "columns" and "data".'}), 400

        # Check if all input arrays have the same length
        if len(data['columns']) != len(data['data'][0]):
            return jsonify({'error': 'Mismatch between number of columns and data points.'}), 400

        # Create DataFrame from input data
        input_data = pd.DataFrame(data['data'], columns=data['columns'])
        
        # Make predictions
        predictions = model.predict(input_data)
        
        return jsonify({'predictions': predictions.tolist()})

    except Exception as e:
        app.logger.error(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5001, debug=True)
