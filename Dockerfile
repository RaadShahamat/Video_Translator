# Optional: the command-line version with NO network at all (strongest isolation).
# CPU only. The web page (app.py) is not meant for this: a browser has to reach it.
#
# Build once (internet ON):
#   docker build -t local-translator .
#
# Run (network fully removed):
#   docker run --rm --network none \
#     -v "$PWD/models:/app/models:ro" \
#     -v "$PWD/videos:/data" \
#     local-translator /data/my_video.mp4 --target English

FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pipeline.py translate_video.py ./

ENTRYPOINT ["python", "translate_video.py"]
