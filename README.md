# DC-chess

Chess real-board -> digital-board detection project.

Structure:
- detection: YOLOv8 detector training and detection scripts
- classifier: Keras/TensorFlow classifier training & utils
- preprocessing: board warp & tile extraction utilities
- inference: full end-to-end pipeline and helper scripts
- web: optional FastAPI wrapper

Follow the repo README and notebook for step-by-step workflow.


# Navigate to your project
cd ~/Desktop/dc\ project/dc-vs-chess-detection/DC-chess/DC-chess

1. pip install -r requirements.txt

backend
2. python3 backend.py

# Start a simple web server
3. python3 -m http.server 8000
```

Then open your browser and go to:
```
http://localhost:8000/frontend.html
