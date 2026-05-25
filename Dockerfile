FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port (Flask default)
EXPOSE 5000

# Environment variable (important for Flask)
ENV FLASK_APP=app.py

# Run app
CMD ["python", "app.py"]
