PLATFORM ?= linux/amd64

UID := $(shell id -u)
GID := $(shell id -g)


## Stops all services in docker-compose
stop:
	docker compose stop

down:
	docker compose down --volumes --remove-orphans

clean:
	sudo rm -rf postgres_data/ mlflow_pg_data/ mlflow_artifacts/

restart:
	docker compose restart

up:
	mkdir -p mlflow_artifacts/ 
	chmod -R 777 mlflow_artifacts
	UID=$(UID) GID=$(GID) docker compose up -d