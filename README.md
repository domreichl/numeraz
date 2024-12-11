# numeraz

## Setup
1. Install project with dependencies: `sh setup_naz.sh`
1. Create Azure subscription
1. Create Azure ML resources: `sh setup_aml.sh`
1. Create Numerai data asset: `naz job create_data_asset`

## API Usage

### Components
- Run a component job: `naz job <component_name>`
- Register a component: `naz component <component_name>`

### Pipelines
- Run a pipeline job: `naz pipeline <pipeline_name>`

### Updates
- Update the virtual environment: `naz update conda`
