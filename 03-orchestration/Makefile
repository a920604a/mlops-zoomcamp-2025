PLATFORM ?= linux/amd64


up:
	docker compose up  -d



## Stops all services in docker-compose
stop:
	docker compose stop

down:
	docker compose down --volumes --remove-orphans

clean:
	sudo rm -rf postgres-db-volume/

restart:
	docker compose restart
init:
	docker compose up airflow-init -d