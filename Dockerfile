# Use the full Debian image with Python
FROM python:3.12

# Set the working directory in the container
WORKDIR /app

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Copy requirements.txt and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install -y libasound2-dev
RUN apt-get update && apt-get install -y alsa-utils

# Copy the entire project directory into the container
COPY . .

# Launch into bash shell
CMD ["/bin/bash"]
