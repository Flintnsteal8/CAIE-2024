#!/bin/bash

# Function to handle errors
error_exit() {
    echo "$1" 1>&2
    exit 1
}

# Logging function
log() {
    echo "$(date +"%Y-%m-%d %T") - $1"
}

# Step 1: Set up the environment
#setup_environment() {
 #   log "Setting up the environment..."
    # Check if virtual environment exists, if not create it
  #  if [ ! -d "venv" ]; then
   #     python -3.7 -m venv venv || error_exit "Failed to create virtual environment."
    #fi
    # Activate virtual environment
    #source venv/bin/activate || error_exit "Failed to activate virtual environment."
    #log "Environment setup complete."
#}

# Step 2: Data Processing
data_processing() {
    log "Starting data processing..."
    python src/DataProcessing.py || error_exit "Data processing failed."
    log "Data processing complete."
}

# Step 3: Data Modelling
data_modelling() {
    log "Starting data modelling..."
    python src/DataModelling.py || error_exit "Data modelling failed."
    log "Data modelling complete."
}

# Clean up (if any temporary files are generated)
cleanup() {
    log "Cleaning up..."
    # Add cleanup commands if necessary
    log "Cleanup complete."
}

# Main execution
main() {
    log "Script execution started."
    data_processing
    data_modelling
    cleanup
    log "Script execution completed successfully."
}

# Execute main function
main
