FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

# Health/metrics server listens on HEALTH_PORT (default 8080)
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import os,urllib.request,sys;port=os.environ.get('HEALTH_PORT','8080');url='http://127.0.0.1:%s/healthz'%port;urllib.request.urlopen(url,timeout=2).read();sys.exit(0)" || exit 1

CMD ["python", "bot.py"]


