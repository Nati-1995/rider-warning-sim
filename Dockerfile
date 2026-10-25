FROM cyberbotics/webots.cloud:R2025a-ubuntu22.04
ARG PROJECT_PATH
RUN mkdir -p $PROJECT_PATH
COPY . $PROJECT_PATH
RUN apt-get update && apt-get install -y python3-pip && pip install --no-cache-dir numpy matplotlib
