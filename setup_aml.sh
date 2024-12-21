#!/bin/bash

PROJECT_DIR=${PWD##*/}
LOCATION="germanywestcentral"
COMPUTE_SIZE="Standard_E4ds_v4"
CONDA_FILE="conda.yml"
ENV_IMAGE="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04"

read -p "Azure Subscription ID: " SUBSCRIPTION_ID

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

echo "Creating a virtual environment"
envName="${PROJECT_DIR}-env"
az ml environment create --name ${env_name} --conda-file $CONDA_FILE -g "${PROJECT_DIR}-rg" -w "${PROJECT_DIR}-ws" --image $ENV_IMAGE

cat <<EOF > resources.json
{
  "subscription_id": "${SUBSCRIPTION_ID}",
  "resource_group": "${PROJECT_DIR}-rg",
  "workspace_name": "${PROJECT_DIR}-ws",
  "compute_instance": "${PROJECT_DIR}-ci01",
  "environment_name": "${PROJECT_DIR}-env"
}
EOF


echo "Creating service principal for role-based access control for GitHub Actions"
servicePrincipalName="Azure-ARM-${envName}-${PROJECT_DIR}"
az ad sp create-for-rbac --name $servicePrincipalName --role "Contributor" --scopes /subscriptions/$SUBSCRIPTION_ID --json-auth 
