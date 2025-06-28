# Use Python 3.11 slim image as base
FROM python:3.11-slim-bullseye

# Upgrade system packages to address vulnerabilities
RUN apt-get update && apt-get upgrade -y && apt-get clean

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV UV_CACHE_DIR=/app/.uv-cache

# Install system dependencies and uv
RUN apt update && apt install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv /root/.local/bin/uv /usr/local/bin/uv

# Copy dependency files
COPY pyproject.toml ./

# Install Python dependencies using uv
RUN uv sync

# Copy the application code
COPY arxiv_searcher/ ./arxiv_searcher/

# Create a non-root user for security
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Expose port (if the MCP server runs on a specific port)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import arxiv_searcher.arxiv_mcp; print('OK')" || exit 1

# Default command to run the MCP server
CMD ["uv", "run", "arxiv_searcher/arxiv_mcp.py"]
