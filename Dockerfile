FROM python:3.11

WORKDIR /app

COPY requirements.txt ./requirements.txt

COPY models/ /app/models/

COPY app/ /app/app/

COPY src/visualization/ /app/visualization/

RUN ls -R /app

# EXPOSE 8080

RUN pip cache purge
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD streamlit run /app/app/dashbord.py --server.port $PORT

# ENTRYPOINT ["streamlit", "run"]
#
# CMD ["hello.py"]
