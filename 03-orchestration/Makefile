PLATFORM ?= linux/amd64

UID := $(shell id -u)
GID := $(shell id -g)


## Stops all services in docker-compose
stop:
	docker compose stop

down:
	docker compose down --volumes --remove-orphans

clean:
	sudo rm -rf postgres_data/ mlflow/ mlflow_db/

restart:
	docker compose restart

up:
	mkdir -p mlflow mlflow_db
	chmod -R 777 mlflow mlflow_db
	UID=$(UID) GID=$(GID) docker compose up -d