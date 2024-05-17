FROM mysql:latest

# tema de permisos
RUN chown -R mysql:root /var/lib/mysql/

ARG MYSQL_DATABASE
ARG MYSQL_ROOT_USER
ARG MYSQL_ROOT_PASSWORD

ENV MYSQL_DATABASE=$MYSQL_DATABASE
ENV MYSQL_ROOT_USER=$MYSQL_ROOT_USER
ENV MYSQL_ROOT_PASSWORD=$MYSQL_ROOT_PASSWORD

# al final lo hemos puesto en el docker compose
# COPY init_db.sql /docker-entrypoint-initdb.d/

EXPOSE 3306