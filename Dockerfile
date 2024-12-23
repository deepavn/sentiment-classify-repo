# Use an official lightweight Python image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy local files to the container
COPY main.py /app/
COPY requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port (if needed, not mandatory for Cloud Run)
EXPOSE 8080

# Define the command to run your script
CMD ["python", "main.py"]
