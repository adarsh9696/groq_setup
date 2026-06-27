FROM python:3.12
WORKDIR /app
COPY requirements-api.txt .
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install -r requirements-api.txt
COPY . .
CMD uvicorn fast_api:app --host 0.0.0.0 --port $PORT