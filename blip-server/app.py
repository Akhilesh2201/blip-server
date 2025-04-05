import os
import cv2
import torch
import logging
import asyncio
import json
import numpy as np

from PIL import Image
from io import BytesIO
from threading import Lock

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoProcessor, AutoModelForImageToText

from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from av import VideoFrame


# ========== Configuration ==========
CACHE_DIR = "/tmp/huggingface_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

os.environ["HF_HOME"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR

device = "cuda" if torch.cuda.is_available() else "cpu"

# ========== Logging ==========
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BLIPServer")
logger.info(f"Using device: {device.upper()}")

# ========== Load BLIP Model ==========
try:
    processor = AutoProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-large",
        cache_dir=CACHE_DIR
    )
    model = AutoModelForImageToText.from_pretrained(
        "Salesforce/blip-image-captioning-large",
        cache_dir=CACHE_DIR
    ).to(device)
    logger.info("BLIP model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise RuntimeError("Model initialization failed.")

# ========== FastAPI Setup ==========
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pcs = set()
caption_lock = Lock()
latest_frame = None

# ========== Caption Generation ==========
def generate_caption(image_np):
    if image_np is None:
        return "Waiting for frames..."

    try:
        resized = cv2.resize(image_np, (640, 480))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(rgb)

        inputs = processor(images=image_pil, return_tensors="pt").to(device)

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_length=30, num_beams=5)

        caption = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return f"BLIP: {caption}"
    except Exception as e:
        logger.error(f"Caption generation error: {e}")
        return "Caption unavailable"

# ========== WebRTC Video Processing ==========
class VideoTransformTrack(VideoStreamTrack):
    def __init__(self, track):
        super().__init__()
        self.track = track

    async def recv(self):
        global latest_frame
        try:
            frame = await self.track.recv()
            img = frame.to_ndarray(format="bgr24")

            with caption_lock:
                latest_frame = img.copy()

            caption = generate_caption(latest_frame)

            cv2.putText(img, caption, (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            new_frame = VideoFrame.from_ndarray(img, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            return new_frame
        except Exception as e:
            logger.error(f"Error in video frame: {e}")
            raise

# ========== API: REST Endpoint ==========
@app.post("/caption")
async def caption_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image_np = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        if image_np is None:
            raise ValueError("Could not decode image")
        caption = generate_caption(image_np)
        return {"caption": caption}
    except Exception as e:
        logger.error(f"Error in /caption: {e}")
        raise HTTPException(status_code=500, detail="Caption generation failed")

# ========== API: WebRTC Signaling ==========
@app.post("/offer")
async def handle_offer(sdp: dict):
    try:
        offer = RTCSessionDescription(sdp["sdp"], sdp["type"])
        pc = RTCPeerConnection()
        pcs.add(pc)

        @pc.on("iceconnectionstatechange")
        async def on_ice_change():
            logger.info(f"ICE state: {pc.iceConnectionState}")
            if pc.iceConnectionState in ["failed", "disconnected"]:
                await pc.close()
                pcs.discard(pc)

        @pc.on("track")
        def on_track(track):
            logger.info(f"Track received: {track.kind}")
            if track.kind == "video":
                pc.addTrack(VideoTransformTrack(track))

        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return {
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type
        }
    except Exception as e:
        logger.error(f"WebRTC error: {e}")
        raise HTTPException(status_code=500, detail="WebRTC negotiation failed")

# ========== API: WebSocket ==========
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            await asyncio.sleep(1)
            with caption_lock:
                frame_copy = latest_frame.copy() if latest_frame is not None else None
            caption = generate_caption(frame_copy)
            await websocket.send_text(json.dumps({
                "caption": caption,
                "device": device.upper(),
                "status": "ok"
            }))
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close(code=1011)

# ========== Shutdown Cleanup ==========
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down server...")
    for pc in pcs:
        await pc.close()
    pcs.clear()

# ========== Health Check ==========
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "device": device.upper(),
        "model_loaded": True
    }
