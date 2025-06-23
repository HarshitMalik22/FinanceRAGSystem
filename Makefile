.PHONY: help install setup build start stop restart logs test clean lint format check-env

# Define variables
DOCKER_COMPOSE := docker-compose -f docker-compose.yml
DOCKER_COMPOSE_DEV := docker-compose -f docker-compose.yml -f docker-compose.override.yml
DOCKER_COMPOSE_PROD := docker-compose -f docker-compose.yml -f docker-compose.prod.yml

# Default target
help:
	@echo "Available targets:"
	@echo "  help      - Show this help message"
	@echo "  install   - Install project dependencies"
	@echo "  setup     - Setup development environment"
	@echo "  build     - Build all services"
	@echo "  start     - Start all services in detached mode"
	@echo "  stop      - Stop all services"
	@echo "  restart   - Restart all services"
	@echo "  logs      - Show logs for all services"
	@echo "  test      - Run tests"
	@echo "  lint      - Run linters"
	@echo "  format    - Format code"
	@echo "  clean     - Remove temporary files and containers"
	@echo "  clean-all - Remove all build artifacts and containers"

# Install project dependencies
install:
	@echo "Installing project dependencies..."
	cd backend && pip install -e .
	cd frontend && npm install

# Setup development environment
setup: check-env
	@echo "Setting up development environment..."
	cp .env.example .env
	@echo "Please update the .env file with your configuration"

# Build all services
build:
	@echo "Building all services..."
	$(DOCKER_COMPOSE) build

# Start all services in development mode
dev: check-env
	@echo "Starting development environment..."
	$(DOCKER_COMPOSE_DEV) up -d

# Start all services in production mode
prod: check-env
	@echo "Starting production environment..."
	$(DOCKER_COMPOSE_PROD) up -d

# Stop all services
stop:
	@echo "Stopping all services..."
	$(DOCKER_COMPOSE) down

# Restart all services
restart: stop start

# Show logs for all services
logs:
	$(DOCKER_COMPOSE) logs -f

# Run tests
test:
	@echo "Running tests..."
	docker-compose run --rm backend pytest tests/

# Run linters
lint:
	@echo "Running linters..."
	docker-compose run --rm backend flake8 .
	docker-compose run --rm frontend npm run lint

# Format code
format:
	@echo "Formatting code..."
	docker-compose run --rm backend black .
	docker-compose run --rm backend isort .
	docker-compose run --rm frontend npm run format

# Clean temporary files and containers
clean:
	@echo "Cleaning up..."
	docker-compose down -v --remove-orphans
	docker-compose rm -f
	docker volume rm $(shell docker volume ls -q -f "dangling=true") 2>/dev/null || true

# Clean all build artifacts and containers
clean-all: clean
	@echo "Removing all build artifacts..."
	docker system prune -a -f --volumes
	cd frontend && rm -rf node_modules build .next out

# Check if .env file exists
check-env:
	@if [ ! -f .env ]; then \
		echo "Error: .env file not found. Please run 'make setup' first."; \
		exit 1; \
	fi

# Run database migrations
migrate:
	@echo "Running database migrations..."
	docker-compose run --rm backend alembic upgrade head

# Create new database migration
migration:
	@if [ -z "$(name)" ]; then \
		echo "Error: Please specify a migration name with 'make migration name=your_migration_name'"; \
		exit 1; \
	fi
	docker-compose run --rm backend alembic revision --autogenerate -m "$(name)"

# Run backend shell
shell:
	docker-compose run --rm backend python -m IPython

# Run frontend development server
frontend-dev:
	cd frontend && npm start

# Run backend development server
backend-dev:
	cd backend && python -m flask run --host=0.0.0.0 --port=8080 --reload

# Show service status
status:
	$(DOCKER_COMPOSE) ps

# Show resource usage
stats:
	docker stats $(docker-compose ps -q)

# Follow logs for a specific service
logs-%:
	$(DOCKER_COMPOSE) logs -f $*

# Helpers for common tasks

# Install pre-commit hooks
install-hooks:
	pre-commit install

# Run security audit
audit:
	docker-compose run --rm backend safety check
	docker-compose run --rm frontend npm audit

# Generate API documentation
docs:
	@echo "Generating API documentation..."
	docker-compose run --rm backend sphinx-apidoc -o docs/source/ backend/
	docker-compose run --rm backend sphinx-build -b html docs/source/ docs/build/

# Open API documentation
open-docs:
	open docs/build/index.html

# Run performance tests
perf-test:
	@echo "Running performance tests..."
	k6 run loadtest/script.js

# Check for outdated dependencies
outdated:
	@echo "Checking for outdated dependencies..."
	docker-compose run --rm backend pip list --outdated
	docker-compose run --rm frontend npm outdated

# Update dependencies
update:
	@echo "Updating dependencies..."
	docker-compose build --no-cache
	docker-compose run --rm backend pip install -U pip
	docker-compose run --rm backend pip install -U -r requirements.txt
	docker-compose run --rm frontend npm update

# Backup database
backup-db:
	@echo "Creating database backup..."
	@mkdir -p backups
	docker-compose exec -T db pg_dump -U postgres financerag > backups/backup_$(shell date +%Y%m%d_%H%M%S).sql

# Restore database
restore-db:
	@if [ -z "$(file)" ]; then \
		echo "Error: Please specify a backup file with 'make restore-db file=backups/backup_YYYYMMDD_HHMMSS.sql'"; \
		exit 1; \
	fi
	docker-compose exec -T db psql -U postgres -d financerag -f /backups/$(file)

# Show help for all make targets
help-all:
	@$(MAKE) -pRrq -f $(lastword $(MAKEFILE_LIST)) : 2>/dev/null | \
	awk -v RS= -F: '/^# File/,/^# Finished Make data base/ {if ($$1 !~ "^[#.]") {print $$1}}' | \
	sort | egrep -v -e '^[^[:alnum:]]' -e "^_" | column -t

# Include any custom makefile includes
-include Makefile.local
