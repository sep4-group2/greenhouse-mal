# SEP4 MAL 

This project contains the Machine learning aspect of the Semester project:

- **mal-model**: ML component that trains and builds a classification model
- **mal-api**: REST API that serves the trained model for predictions

## Quick Start

### Start the System

```sh
docker compose -f docker-compose-local.yml up -d --build
```

### Train the Model

```sh
docker compose -f docker-compose-local.yml --profile build-model up mal-model
```

### Stop the System

```sh
docker compose -f docker-compose-local.yml down
```
