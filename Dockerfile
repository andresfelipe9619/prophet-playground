# A container for the dashboard, for any host that runs one: Render, Railway,
# Fly.io, Cloud Run, a VPS. Streamlit Community Cloud does not use this file —
# it installs requirements.txt itself. See docs/deployment.md.
#
# The one thing worth knowing: Streamlit is a **long-running server holding a
# websocket per viewer**, not a request/response function. That is why this is a
# container and not a serverless handler, and it is the whole reason Vercel is
# not on the list of hosts above.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first, so editing a page module does not reinstall xgboost.
#
# requirements.txt, the same file a local checkout installs — there is no
# separate deploy list. There was one briefly, and it silently omitted Prophet,
# which meant the hosted app offered five models where the real one offers six.
# A dependency list that can differ from what the app needs will eventually
# differ from what the app needs. requirements-test.txt is a different question
# (what pytest imports) and stays separate.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# $PORT is what Render, Railway, Fly and Cloud Run all hand the process; 8501 is
# Streamlit's own default, for a plain `docker run -p 8501:8501`. The address
# must be 0.0.0.0: bound to localhost the container answers only itself.
ENV PORT=8501
EXPOSE 8501

# Shell form on purpose — $PORT has to be expanded at start time, and the exec
# form would pass the literal string "$PORT" to Streamlit.
CMD streamlit run dashboard/app.py \
      --server.port=${PORT} \
      --server.address=0.0.0.0 \
      --server.headless=true
