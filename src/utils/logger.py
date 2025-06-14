import logging
import logging.config
import yaml
from pathlib import Path


def setup_logger(name: str) -> logging.Logger:
    """Set up logger with configuration."""
    
    # Load logging configuration
    config_path = Path("config/logging_config.yaml")
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logging.config.dictConfig(config)
    else:
        # Default configuration
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('research_assistant.log')
            ]
        )
    
    return logging.getLogger(name)
