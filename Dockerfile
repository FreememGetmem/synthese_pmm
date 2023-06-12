FROM python:3.7

WORKDIR /app

COPY requirements.txt ./requirements.txt

COPY models/ /app/models/

COPY app/ /app/app/

COPY src/visualization/ /app/visualization/

RUN ls -R /app

# EXPOSE 8080

RUN python3 -m pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

CMD streamlit run /app/app/dashbord.py --server.port $PORT

# ENTRYPOINT ["streamlit", "run"]
#
# CMD ["hello.py"]


