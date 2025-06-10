#! /bin/bash
pipenv install numpy pyarrow scikit-learn==1.5.0 flask pandas gunicorn requests
docker build -t mlops-hw5 .
docker run -ti mlops-hw5 /bin/bash
