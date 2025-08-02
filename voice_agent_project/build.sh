#!/usr/bin/env bash
# Exit on error
set -o errexit

# Install Python dependencies
pip install -r requirements.txt

# Generate Prisma Client
prisma generate
 
# Run migrations
python -m prisma db push   

# Collect Django static files (if applicable)
python manage.py collectstatic --no-input

# Run database migrations (optional, if using Django migrations)
python manage.py migrate