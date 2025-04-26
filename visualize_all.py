#!/usr/bin/env python3
"""
Run all visualization scripts to generate comprehensive analytics
"""

import os
import sys
from pathlib import Path
import subprocess
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("visualization")

def run_visualization_script(script_path):
    """Run a visualization script and log the result"""
    logger.info(f"Running visualization script: {script_path}")
    
    try:
        result = subprocess.run(
            [sys.executable, script_path], 
            check=True,
            capture_output=True,
            text=True
        )
        logger.info(f"Successfully ran {script_path}")
        logger.info(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running {script_path}")
        logger.error(f"Return code: {e.returncode}")
        logger.error(f"Output: {e.stdout}")
        logger.error(f"Error: {e.stderr}")
        return False

def main():
    """Run all visualization scripts"""
    base_path = Path(__file__).parent
    
    # List of visualization scripts to run
    visualization_scripts = [
        base_path / "visualizations" / "economic_analysis.py",
        base_path / "visualizations" / "model_performance.py"
    ]
    
    # Run each script
    success_count = 0
    for script in visualization_scripts:
        if run_visualization_script(script):
            success_count += 1
    
    # Report overall status
    logger.info(f"Completed {success_count}/{len(visualization_scripts)} visualization scripts")
    
    if success_count == len(visualization_scripts):
        logger.info("All visualizations successfully generated!")
    else:
        logger.warning("Some visualization scripts failed. Check the logs for details.")

if __name__ == "__main__":
    main() 