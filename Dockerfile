# Dockerfile for the app, Image, Container
# basis-image with Python 3.12
FROM python:3.12-slim

# create a working directory in the container:
WORKDIR /app

# copy the requirements into this working directory:
COPY requirements.txt requirements.txt

# install requirements with pip
RUN pip install -r requirements.txt

# copy the rest of the code into the image
COPY . .

# define environment variables to correctly start Flask
ENV FLASK_APP=src/api/run.py
ENV FLASK_ENV=production
ENV FLASK_RUN_HOST=0.0.0.0

# expose Port for Flask
EXPOSE 5000

# start the Flask-App
CMD ["python", "-m", "src.api.run"]
