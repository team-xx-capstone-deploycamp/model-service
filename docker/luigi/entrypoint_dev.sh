#!/bin/bash

until PGPASSWORD=luigipass psql -h postgres -U luigi -d luigi -c '\q'; do
  echo "Postgres is unavailable - sleeping"
  sleep 1
done

echo "Postgres is up"

echo "Starting Luigi scheduler..."
luigid --address 0.0.0.0 --port 8082 --background --pidfile /var/run/luigi.pid
echo "Luigi scheduler started with PID $(cat /var/run/luigi.pid)"

tail -f /dev/null
