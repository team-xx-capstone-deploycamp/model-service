#!/bin/bash

until PGPASSWORD=${LUIGI_DB_PASSWORD} psql -h ${LUIGI_DB_HOST} -U ${LUIGI_DB_USER} -d ${LUIGI_DB_NAME} -c '\q'; do
  echo "Postgres is unavailable - sleeping"
  sleep 1
done

echo "Postgres is up"

echo "Starting Luigi scheduler..."
luigid --address 0.0.0.0 --port 8082 --background --pidfile /var/run/luigi.pid
echo "Luigi scheduler started with PID $(cat /var/run/luigi.pid)"

tail -f /dev/null
