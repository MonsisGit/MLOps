# Base image
FROM python:3.7-slim

# install python 
RUN apt update && \
apt install --no-install-recommends -y build-essential gcc && \
apt clean && rm -rf /var/lib/apt/lists/*


#Copying stuff
COPY mlops_cookiecutter/requirements.txt requirements.txt
COPY mlops_cookiecutter/setup.py setup.py
COPY mlops_cookiecutter/src/ src/
#COPY mlops_cookiecutter/data/ data/
COPY mlops_cookiecutter/models/ models/
COPY mlops_cookiecutter/reports/ reports/
#COPY .dvc/ .dvc/
COPY mlops_cookiecutter/data.dvc data.dvc
COPY mlops_cookiecutter/tests/ tests/
#COPY .git/ .git/

#install modules
#WORKDIR /
#RUN pip install -r requirements.txt --no-cache-dir
RUN pip install dvc[gc]

RUN dvc init --no-scm
RUN dvc remote add -d storage gs://mlops-bucket-simon/
RUN dvc pull --remote storage
#RUN dvc config core.no_scm true

#Entrypoint
ENTRYPOINT ["python", "-u", "src/models/train_model.py"]
