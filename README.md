# Hazards-Prioritization
For paper "Hazards Prioritization With Cognitive Attention Maps for Supporting Driving Decision-Making"

## Prerequisites

- Docker and Docker Compose installed
- NVIDIA Docker support (for GPU acceleration)
- Input data prepared according to the expected format
- At least 8GB of RAM available for Docker

## Quick Start

1. Extract and Load Docker Image
unzip hazard-prioritization-package.zip
cd hazard-prioritization-package

docker load < hazard-prioritization.tar.gz

2. Create the necessary directories for mounting volumes:

```bash
mkdir -p model input output bitmask
```

3. Place your data in the appropriate directories:
   - Place the model in the `model/` directory
   - Place your features.json file in the `input/` directory
   - Place your bitmask images in the `bitmask/` directory

4. Run the conversion step to prepare TFRecords:

```bash
docker-compose run --rm hazard-prioritization convert
```

5. Run the prediction:

```bash
docker-compose run --rm hazard-prioritization predict
```

6. Check the output directory for results.

## Customizing Paths

You can customize paths by modifying the environment variables in the `docker-compose.yaml` file:

```yaml
environment:
  - MODEL_DIR=/data/model
  - FCE_JSON_FILE=/data/input/features.json
  - BITMASK_FOLDER=/data/bitmask
  - PREDICTION_OUTPUT_DIR=/data/output
  - TEST_TFRECORD_DIR=/data/input/tfrecords
```

## Directory Structure

- `/data/model`: Contains the trained model
- `/data/input`: Contains input files including features.json
- `/data/input/tfrecords`: Will contain generated TFRecord files
- `/data/bitmask`: Contains bitmask images
- `/data/output`: Where prediction results will be saved

## Data Format Requirements

For successful application, your simulator needs to produce data in the following formats:

### 1. Bitmask Images

- Format: PNG files
- Requirements: Each object should have a unique segment ID in the alpha channel
- Naming: `<image_id>.png` where `<image_id>` matches the IDs in your JSON data

### 2. Feature JSON Format

Your simulator should generate a JSON file with this structure:

```json
{
  "FCE_Result": [
    {
      "image_id": "frame_001",
      "global_vis_degree": 0.85,
      "Instances_Amount": 12,
      "Nearest_Instance_Distance": 5.2,
      "result_info": [
        {
          "category_id": 35,
          "segment_id": 1035,
          "Matrix_X": [10.5, 2500, 45.0, 35, 0.95]
        },
        {
          "category_id": 31,
          "segment_id": 1031,
          "Matrix_X": [15.2, 1200, 30.0, 31, 0.85]
        }
      ]
    }
  ]
}
```

Where:
- `Matrix_X` contains: [distance, size, direction, category, visibility]
- `category_id` follows the BDD10K format (31=person, 35=car, etc.)
- `segment_id` should match the ID in the bitmask image alpha channel
- `INSTANCE_IDS` [31, 32, 33, 34, 35, 37, 39, 40] #BDD10K


- `Label dictionary for BDD10K` LABEL_DICT = {
    7: 'road', 8: 'sidewalk', 10: 'building', 15: 'wall', 
    11: 'fence', 20: 'pole', 25: 'traffic light', 26: 'traffic sign', 
    29: 'vegetation', 28: 'terrain', 30:'sky', 31: 'person', 
    32: 'rider', 35: 'car', 40: 'truck', 34: 'bus', 
    39: 'train', 37: 'motorcycle', 33: 'bicycle'}

## Integration

The guide below explains how to integrate the containerized Hazard Prioritization System with your existing simulation platform.

### Approach 1: File-Based Integration

This approach uses shared directories for data exchange between your simulator and the hazard prioritization system.

1. **Set up shared directories**:
   ```bash
   mkdir -p simulator-integration/input
   mkdir -p simulator-integration/output
   mkdir -p simulator-integration/model
   mkdir -p simulator-integration/bitmask
   ```

2. **Configure your simulator to output**:
   - Segmentation bitmasks to the `bitmask` directory
   - Feature data (JSON format) to the `input` directory

3. **Update the docker-compose.yml file** to mount these directories:
   ```yaml
   volumes:
     - ./simulator-integration/model:/data/model
     - ./simulator-integration/input:/data/input
     - ./simulator-integration/output:/data/output
     - ./simulator-integration/bitmask:/data/bitmask
   ```

4. **Create a simple shell script or batch file** to automate the process:
   ```bash
   #!/bin/bash
   
   # First run your simulator to generate input data
   ./run_simulator.sh --output_dir=./simulator-integration
   
   # Then process the data with the hazard prioritization system
   docker-compose run --rm hazard-prioritization convert
   docker-compose run --rm hazard-prioritization predict
   
   # Optional: Post-process the results
   ./process_results.sh --input_dir=./simulator-integration/output
   ```

### Approach 2: API-Based Integration

For a more tightly coupled integration, you can expose the Docker container as an API service.

1. **Create a custom docker-compose.yml file** for API mode:
   ```yaml
   version: '3'
   
   services:
     hazard-prioritization-api:
       image: hazard-prioritization
       ports:
         - "5000:5000"
       volumes:
         - ./model:/data/model
         - ./shared:/data/shared
       environment:
         - MODEL_DIR=/data/model
         - SHARED_DIR=/data/shared
         - API_MODE=true
       command: api
   ```

2. **Add an API server** to the Docker container by creating a new file `api.py`:
   ```python
   from flask import Flask, request, jsonify
   import os
   import json
   import sys
   import logging
   from PIL import Image
   import numpy as np
   
   # Import your modules
   import config
   import prediction
   from dataset import read_json_file
   
   app = Flask(__name__)
   
   @app.route('/predict', methods=['POST'])
   def predict():
       try:
           # Get data from request
           data = request.json
           
           # Process the data (customize this part to fit your specific needs)
           shared_dir = os.environ.get('SHARED_DIR', '/data/shared')
           
           # Save input data to shared directory
           with open(os.path.join(shared_dir, 'input.json'), 'w') as f:
               json.dump(data, f)
           
           # Run prediction
           results = prediction.run_predictions(config)
           
           # Return results
           return jsonify({'status': 'success', 'results': results.to_dict()})
       
       except Exception as e:
           logging.error(f"API error: {e}")
           return jsonify({'status': 'error', 'message': str(e)}), 500
   
   if __name__ == '__main__':
       app.run(host='0.0.0.0', port=5000)
   ```

3. **Update entrypoint.sh** to support API mode:
   ```bash
   #!/bin/bash
   # Activate conda environment
   source /opt/conda/etc/profile.d/conda.sh
   conda activate Yoki-Hazard-38
   
   # Available commands
   if [ "$1" = "convert" ]; then
       echo "Running JSON to TFRecord conversion..."
       python json_to_tfrecord.py --input "$FCE_JSON_FILE" --output_dir "$TEST_TFRECORD_DIR"
   elif [ "$1" = "predict" ]; then
       echo "Running hazard prediction..."
       python main.py
   elif [ "$1" = "api" ]; then
       echo "Starting API server..."
       python api.py
   else
       # Help text...
   fi
   ```

4. **Call the API from your simulator**:
   ```python
   import requests
   import json
   
   def process_simulation_frame(frame_data, segmentation_data):
       # Prepare the data
       payload = {
           'image_id': frame_data['id'],
           'global_vis_degree': calculate_visibility(frame_data),
           'Instances_Amount': len(segmentation_data['instances']),
           'result_info': prepare_object_data(segmentation_data)
       }
       
       # Call the hazard prioritization API
       response = requests.post('http://localhost:5000/predict', json=payload)
       
       if response.status_code == 200:
           results = response.json()
           # Process the results
           return results['results']
       else:
           print(f"Error: {response.text}")
           return None
   ```

## Programming Language Integration

### Python

```python
import subprocess
import os
import pandas as pd

def run_hazard_prioritization(input_dir, output_dir):
    """Run the hazard prioritization Docker container."""
    # Ensure directories exist
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert and predict
    subprocess.run(["docker-compose", "run", "--rm", "hazard-prioritization", "convert"])
    subprocess.run(["docker-compose", "run", "--rm", "hazard-prioritization", "predict"])
    
    # Load and return results
    results_file = os.path.join(output_dir, 'getPredictResults.xlsx')
    if os.path.exists(results_file):
        return pd.read_excel(results_file)
    else:
        raise FileNotFoundError(f"Results file not found at {results_file}")

# Example usage in your simulator loop
for frame in simulation_frames:
    # Generate input data
    generate_bitmasks(frame, "bitmask")
    generate_json(frame, "input/features.json")
    
    # Run hazard prioritization
    results = run_hazard_prioritization("input", "output")
    
    # Use results in your simulator
    update_simulator_visuals(frame, results)
```

### C++

```cpp
#include <iostream>
#include <string>
#include <filesystem>
#include <cstdlib>

namespace fs = std::filesystem;

bool runHazardPrioritization(const std::string& inputDir, const std::string& outputDir) {
    // Ensure directories exist
    fs::create_directories(inputDir);
    fs::create_directories(outputDir);
    
    // Run Docker commands
    int convertResult = std::system("docker-compose run --rm hazard-prioritization convert");
    if (convertResult != 0) {
        std::cerr << "Error running conversion step" << std::endl;
        return false;
    }
    
    int predictResult = std::system("docker-compose run --rm hazard-prioritization predict");
    if (predictResult != 0) {
        std::cerr << "Error running prediction step" << std::endl;
        return false;
    }
    
    // Check if output file exists
    std::string resultsFile = outputDir + "/getPredictResults.xlsx";
    if (!fs::exists(resultsFile)) {
        std::cerr << "Results file not found at " << resultsFile << std::endl;
        return false;
    }
    
    return true;
}

// Example usage in simulator
void simulationLoop() {
    for (int frame = 0; frame < totalFrames; frame++) {
        // Generate input data
        generateBitmasks(frame, "bitmask");
        generateJson(frame, "input/features.json");
        
        // Run hazard prioritization
        bool success = runHazardPrioritization("input", "output");
        
        if (success) {
            // Process results
            // Note: You'll need a library to read Excel files in C++
            processResults("output/getPredictResults.xlsx");
        }
    }
}
```

## Troubleshooting

If you encounter any issues:

1. Check that all input files are in the correct locations
2. Verify that the model is properly saved and compatible
3. Ensure you have sufficient disk space for output files

## Citation

If you use this hazard prioritization package in your research or project, please cite it as follows:

```
@ARTICLE{10570394,
  author={Huang, Yaoqi and Wang, Xiuying},
  journal={IEEE Transactions on Intelligent Transportation Systems}, 
  title={Hazards Prioritization With Cognitive Attention Maps for Supporting Driving Decision-Making}, 
  year={2024},
  volume={25},
  number={11},
  pages={16221-16234},
  keywords={Hazards;Visualization;Semantics;Resource management;Appraisal;Autonomous vehicles;Road safety;Image analysis;Advanced driver assistance systems;Decision making;Attention map;autonomous vehicles;cognition;road safety;scene understanding},
  doi={10.1109/TITS.2024.3413675}}

```


