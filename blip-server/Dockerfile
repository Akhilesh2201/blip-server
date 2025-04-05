FROM nvidia/cuda:12.1.0-runtime-ubuntu20.04

ENV TZ=UTC

RUN apt-get update && apt-get install -y \
    python3 python3-pip git curl libgl1-mesa-glx ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --upgrade pip && pip3 install -r requirements.txt

COPY . .

EXPOSE 7860

ENTRYPOINT ["bash", "run.sh"]