FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir matplotlib scikit-learn numpy

COPY . .

EXPOSE 8080

# Serve the web app (index.html) on port 8080
CMD ["python", "-m", "http.server", "8080", "--directory", "/app"]
