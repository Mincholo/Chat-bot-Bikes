#!/bin/bash
exec gunicorn -w 1 -b 0.0.0.0:5000 bot_api:app
