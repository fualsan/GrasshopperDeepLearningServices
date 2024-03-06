# Use the official Ubuntu LTS image as base image
FROM ubuntu:22.04

# Set the working directory in the container
WORKDIR /api_server

# Install system dependencies
RUN apt update && apt install -y \
    python3.11 \
    python3-pip \
    && apt clean

RUN ln -s $(which python3.11) /usr/bin/python

RUN python -m pip install --upgrade pip

COPY requirements.txt /api_server
COPY sd_1_5_generic_server.py /api_server

RUN pip install -U pip && pip install --no-cache-dir -r requirements.txt

# Expose port 8000 to the outside world
EXPOSE 8000

# Command to run the FastAPI application
CMD ['python', 'sd_1_5_generic_server.py']