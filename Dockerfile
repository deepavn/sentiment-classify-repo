FROM python:3.11-slim

WORKDIR /app

# Copy local files to the container
COPY main.py /app/
COPY requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8080
EXPOSE 8080

# Start the Flask app
CMD ["python", "main.py"]
