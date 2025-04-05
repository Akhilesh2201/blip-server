#!/bin/bash

echo "Starting FastAPI BLIP app..."
uvicorn app:app --host 0.0.0.0 --port 7860 --reload