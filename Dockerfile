FROM python:3.7

WORKDIR /app

COPY requirements.txt ./requirements.txt

COPY /models /app

COPY /app /app

# EXPOSE 8080

RUN pip install -r requirements.txt

CMD streamlit run /app/dashbord.py --server.port $PORT

# ENTRYPOINT ["streamlit", "run"]
#
# CMD ["hello.py"]


