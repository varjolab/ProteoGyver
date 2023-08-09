#!/bin/bash

# Start dash app
cd /proteogyver
echo "starting dash app"
gunicorn -b 0.0.0.0:8050 app:server --log-level debug --timeout 1200 