PLATFORM ?= linux/amd64
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
	docker compose up -d