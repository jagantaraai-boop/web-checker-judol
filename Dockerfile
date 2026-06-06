# Build stage - compile dependencies
FROM python:3.11-slim as builder

WORKDIR /tmp

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and build wheels
COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /tmp/wheels -r requirements.txt

# Runtime stage
FROM python:3.11-slim

WORKDIR /app

# Install runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    wget \
    libgconf-2-4 \
    libglib2.0-0 \
    libx11-6 \
    libx11-xcb1 \
    libxcb1 \
    libxext6 \
    libxrender1 \
    libnss3 \
    libgbm1 \
    libpangocairo-1.0-0 \
    libpango-1.0-0 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libxss1 \
    libxkbcommon0 \
    libgdk-pixbuf2.0-0 \
    libasound2 \
    && rm -rf /var/lib/apt/lists/*

# Copy wheels dari builder stage
COPY --from=builder /tmp/wheels /wheels

# Copy requirements
COPY requirements.txt .

# Install Python packages dari wheels
RUN pip install --no-cache-dir /wheels/* && \
    rm -rf /wheels

# Install playwright dan browsers
RUN pip install --no-cache-dir playwright && \
    playwright install chromium

# Copy project files
COPY . .

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT:-3000}/ || exit 1

# Environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV PORT=3000

# Expose port (Railway akan override ini dengan PORT env var)
EXPOSE ${PORT}

# Run aplikasi
CMD ["python", "app.py"]
