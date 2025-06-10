FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

RUN pip install -U pip
RUN pip install pipenv 

WORKDIR /app

COPY starter.py .

RUN pip install pandas pyarrow scikit-learn

# Default command to run the script with yellow May 2023 data
CMD ["python", "starter.py", "yellow", "2023", "05"]