# Use the full Debian image with Python
FROM python:3.12

# Set the working directory in the container
WORKDIR /app

# Update apt and install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    linux-headers-amd64 \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Copy requirements.txt and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project directory into the container
COPY . .

# Launch into bash shell
CMD ["/bin/bash"]
