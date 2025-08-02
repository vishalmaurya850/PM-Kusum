#!/usr/bin/env bash
# Exit on error
set -o errexit

# Install Python dependencies
pip install -r requirements.txt

# Generate Prisma Client
prisma generate
 
# Run migrations
python -m prisma db push