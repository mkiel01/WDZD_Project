FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential curl

COPY . .

RUN pip3 install -r requirements.txt --no-cache-dir

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
