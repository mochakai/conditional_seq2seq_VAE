FROM floydhub/pytorch:0.4.0-gpu.cuda9cudnn7-py3

RUN pip3 install pandas

WORKDIR /workspace
