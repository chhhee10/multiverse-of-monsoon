# A judge-like runtime for local testing only (the organisers use their own image).
# Python 3.12, CPU torch and the server libraries the PS lists, plus the team kit for its schema validator.
FROM python:3.12-slim
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir numpy fastapi uvicorn requests \
    "monsoon-r1-kit @ https://github.com/OfficialAbhinavSingh/yug-monsoon-kit/archive/689c8798bbff7f318a1e74878a6985abc646080a.zip"
