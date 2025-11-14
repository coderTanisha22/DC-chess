from fastapi import FastAPI, File, UploadFile
import uvicorn
from inference.full_pipeline import predict_from_warped
import os

app = FastAPI()

@app.post("/predict-warped")
async def predict_warped(file: UploadFile = File(...)):
    tmp = f"/tmp/{file.filename}"
    with open(tmp, "wb") as f:
        f.write(await file.read())
    # Expect a warped board image (square). For full pipeline, integrate detection->warp step before sending
    from inference.full_pipeline import predict_from_warped
    board, fen = predict_from_warped(tmp)
    os.remove(tmp)
    return {"fen": fen, "board": board}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8765)
