FROM python:3

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

VOLUME ["/data"]
VOLUME ["/output"]

ENV INPUT=""
ENV OUTPUT=""

COPY . .
CMD python3 irregulars_neureka_codebase/ioChallenge.py --pathIn "/data/$INPUT" --pathOut "/output/$OUTPUT"


# # syntax=docker/dockerfile:1

# FROM continuumio/miniconda3

# # Create the environment:
# COPY environment.yml .
# RUN conda env create -f environment.yml

# # Make RUN commands use the new environment:
# SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

# # Demonstrate the environment is activated:
# RUN echo "Make sure timescoring is installed:"
# RUN python -c "import timescoring"

# # Create a data volume
# VOLUME ["/data"]
# VOLUME ["/output"]

# # Define environment variables
# ENV INPUT=""
# ENV OUTPUT=""

# # The code to run when container is started:
# COPY irregulars_neureka_codebase/ioChallenge.py .
# ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "myenv", "python", "irregulars_neureka_codebase/ioChallenge.py", "--pathIn", "/data/$INPUT" ,"--pathOut" ,"/output/$OUTPUT"]