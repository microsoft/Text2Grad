#!/bin/bash

# Get the current host IP address
HOST_IP=$(hostname -i)
echo "Current host IP: $HOST_IP"

# Create hostfile directory (if it doesn't exist)
mkdir -p ./hostfile

# Write IP address to hostfile
echo "$HOST_IP slots=8" > ./hostfile/hostfile
echo "IP address written to ./hostfile/hostfile"

# Run deepspeed training
deepspeed --hostfile ./hostfile/hostfile --include 10.18.32.97:0,1,2,3,5,6,7 --master_port 29502 train_rm.py --deepspeed
