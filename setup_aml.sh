#!/bin/bash

PROJECT_DIR=${PWD##*/}
LOCATION="germanywestcentral"
COMPUTE_SIZE="Standard_E4ds_v4"
CONDA_FILE="conda.yml"
ENV_IMAGE="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04"

rgName="${PROJECT_DIR}-rg"
wsName="${PROJECT_DIR}-ws"
ciName="${PROJECT_DIR}-ci01"
envName="${PROJECT_DIR}-env"

read -p "Azure Subscription ID: " SUBSCRIPTION_ID

echo "Installing Azure CLI"
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

echo "Installing ML Extension"
az extension remove -n ml
az extension add -n ml -y

echo "Logging in"
az login

echo "Creating a resource group"
az group create -n ${rgName} -l ${LOCATION}

echo "Creating a workspace"
az ml workspace create -n ${wsName} -g ${rgName} -l ${LOCATION}

echo "Creating a compute instance"
az ml compute create -n ${ciName} -t ComputeInstance --size ${COMPUTE_SIZE} -g ${rgName} -w ${wsName}

echo "Creating a virtual environment"
az ml environment create --name ${env_name} --conda-file $CONDA_FILE -g ${rgName} -w ${wsName} --image $ENV_IMAGE

cat <<EOF > src/.env
SUBSCRIPTION_ID=${SUBSCRIPTION_ID}
RESOURCE_GROUP=${rgName}
WORKSPACE_NAME=${wsName}
COMPUTE_INSTANCE=${ciName}
ENVIRONMENT_NAME=${envName}
EOF


echo "Creating service principal for role-based access control for GitHub Actions"
servicePrincipalName="Azure-ARM-${envName}-${PROJECT_DIR}"
az ad sp create-for-rbac --name $servicePrincipalName --role "Contributor" --scopes /subscriptions/$SUBSCRIPTION_ID --json-auth 
