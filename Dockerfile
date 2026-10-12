FROM python:3.12-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

ENV MOVIEPY_MCP_MEDIA_ROOT=/media \
    MOVIEPY_MCP_TRANSPORT=http \
    MOVIEPY_MCP_HOST=0.0.0.0 \
    MOVIEPY_MCP_PORT=8000

VOLUME ["/media"]
EXPOSE 8000

# MOVIEPY_MCP_TOKEN must be set at runtime. Bind stays on the container
# interface; put a reverse proxy in front if this is reachable beyond localhost.
CMD ["moviepy-mcp"]
