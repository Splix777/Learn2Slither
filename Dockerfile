# Use the full Debian image with Python
FROM python:3.12

# Set the working directory in the container
WORKDIR /app

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

RUN apt-get update && apt-get install -y \
    python3-pygame \
    libx11-dev \
    libxext-dev \
    libxrender-dev \
    libxtst-dev \
    libxi-dev \
    libncurses5-dev \
    libasound2-dev

# Copy requirements.txt and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project directory into the container
COPY . .

# Set the screen so gui can be displayed


# Launch into bash shell
CMD ["/bin/bash"]
