# Base image must match the Python version used during training (3.13)
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install dependencies first (cached layer unless requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code, model artefact, and the model class definition.
# src/ must be present so Python can find riskstream_model.RiskStreamModel
# when joblib unpickles the artefact.
COPY main.py .
COPY models/ models/
COPY src/ src/

# Put src/ on the Python path so 'import riskstream_model' resolves correctly
ENV PYTHONPATH=/app/src

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
