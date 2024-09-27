# Use a base image
FROM python:3.8-slim

# Set the working directory
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirment.txt ./
RUN pip install --no-cache-dir -r requirment.txt

# Copy the rest of your application code
COPY . .

# Command to run your application (replace with your actual command)
CMD ["python", "app.py"]
