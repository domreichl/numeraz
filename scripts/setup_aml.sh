#!/bin/bash

PROJECT_DIR=${PWD##*/}
LOCATION="germanywestcentral"
COMPUTE_SIZE="Standard_E4ds_v4"

echo "Installing Azure CLI"
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

echo "Installing ML Extension"
az extension remove -n ml
az extension add -n ml -y

echo "Logging in"
az login

echo "Creating a resource group"
az group create -n "${PROJECT_DIR}-rg" -l ${LOCATION}

echo "Creating a workspace"
az ml workspace create -n "${PROJECT_DIR}-ws" -g "${PROJECT_DIR}-rg" -l ${LOCATION}

echo "Creating a compute instance"
az ml compute create -n "${PROJECT_DIR}-ci01" -t ComputeInstance --size ${COMPUTE_SIZE} -g "${PROJECT_DIR}-rg" -w "${PROJECT_DIR}-ws"

cat <<EOF > resources.json
{
  "resource_group": "${PROJECT_DIR}-rg",
  "workspace_name": "${PROJECT_DIR}-ws",
  "compute_instance": "${PROJECT_DIR}-ci01"
}
EOF
