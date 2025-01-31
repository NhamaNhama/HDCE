FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
WORKDIR /app
RUN apt-get update && apt-get install -y \
git \
curl \
&& rm -rf /var/lib/apt/lists/

RUN pip install --no-cache-dir torch>=1.13.0

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
# CMD ["python", "src/trainer/train.py", "--config", "config/default.yaml"] 