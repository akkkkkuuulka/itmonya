#!/bin/bash
gunicorn main:app --workers 10 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8080 