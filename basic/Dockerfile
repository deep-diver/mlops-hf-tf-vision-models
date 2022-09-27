FROM tensorflow/tfx:1.9.1

WORKDIR /pipeline
COPY ./ ./

RUN pip install -r requirements.txt

ENV PYTHONPATH="/pipeline:${PYTHONPATH}"