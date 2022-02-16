FROM python:3.8.2

RUN apt-get update -y
EXPOSE 80

COPY . /app
WORKDIR /app

RUN pip install -r requirements.txt
CMD ["python", "inference.py"]
