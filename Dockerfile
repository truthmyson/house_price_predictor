# create the base image using the python version for the project
FROM python:3.13.7-slim-bookworm

# create a working directory
WORKDIR /app

# copy all my source code into the workig directory '/app'
COPY . /app

# we will install all the packages or requirements
RUN pip install --no-cache-dir -r requirements.txt

# we have to expose the port we want to run our application on
EXPOSE 2662

# command to run our application
CMD ["python", "app.py"]