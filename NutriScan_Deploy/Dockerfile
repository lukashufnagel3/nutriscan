# Use python:3.9-slim
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Upgrade pip first to ensure it can handle large files and modern wheels
RUN pip install --no-cache-dir --upgrade pip

# Copy the requirements file
COPY requirements.txt .

# Install dependencies using the standard repository
RUN pip install --no-cache-dir -r requirements.txt

# Install fancy graphics dependencies from a separate requirements file
COPY requirements_graphic.txt .
RUN pip install --no-cache-dir -r requirements_graphic.txt

# Copy the application code
COPY . .

# Expose the port
EXPOSE 8501

# Healthcheck
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run the app
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]