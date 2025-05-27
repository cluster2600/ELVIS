"""
Script to run the ELVIS Trading Bot REST API
"""

import os
import sys
import argparse
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from trading.api.app import app
from utils.logger_config import setup_logging

# Setup logging
logger = setup_logging(
    app_name="API",
    log_level="INFO",
    enable_file_logging=True
)


def main():
    parser = argparse.ArgumentParser(description="Run ELVIS Trading Bot REST API")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                       help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5000,
                       help="Port to bind to (default: 5000)")
    parser.add_argument("--debug", action="store_true",
                       help="Run in debug mode")
    parser.add_argument("--workers", type=int, default=1,
                       help="Number of worker processes (default: 1)")
    
    args = parser.parse_args()
    
    logger.info(f"Starting ELVIS Trading Bot API on {args.host}:{args.port}")
    logger.info("API Documentation available at: http://localhost:{}/api/docs".format(args.port))
    
    if args.debug:
        # Development server with debug mode
        logger.warning("Running in DEBUG mode - do not use in production!")
        app.run(
            host=args.host,
            port=args.port,
            debug=True
        )
    else:
        # Production server using gunicorn
        try:
            import gunicorn.app.base
            
            class StandaloneApplication(gunicorn.app.base.BaseApplication):
                def __init__(self, app, options=None):
                    self.options = options or {}
                    self.application = app
                    super().__init__()
                
                def load_config(self):
                    for key, value in self.options.items():
                        if key in self.cfg.settings and value is not None:
                            self.cfg.set(key.lower(), value)
                
                def load(self):
                    return self.application
            
            options = {
                'bind': f'{args.host}:{args.port}',
                'workers': args.workers,
                'worker_class': 'sync',
                'timeout': 120,
                'keepalive': 5,
                'accesslog': '-',
                'errorlog': '-',
                'loglevel': 'info',
                'capture_output': True,
                'enable_stdio_inheritance': True,
            }
            
            logger.info(f"Starting production server with {args.workers} workers")
            StandaloneApplication(app, options).run()
            
        except ImportError:
            logger.warning("Gunicorn not installed. Running with Flask development server.")
            logger.warning("For production, install gunicorn: pip install gunicorn")
            app.run(
                host=args.host,
                port=args.port,
                debug=False
            )


if __name__ == "__main__":
    main()
