FROM python:3.7
COPY . /app
WORKDIR /app
RUN pip install -r requiements.txt
ENTRYPOINT ["python"]
CMD ["mlaas.py"]