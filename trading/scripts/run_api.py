"""
Script to run the ELVIS Trading Bot REST API
"""

import argparse
import logging
import os
import sys

# Add parent directory to path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from utils.logger_config import setup_logging


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the ELVIS paper-compatibility control API"
    )
    parser.add_argument(
        "--host",
        type=str,
        default=os.getenv("API_HOST", "127.0.0.1"),
        help="Host to bind to (default: API_HOST or 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("API_PORT", "5000")),
        help="Port to bind to (default: API_PORT or 5000)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=os.getenv("API_DEBUG", "").lower() == "true",
        help="Run in debug mode (default: API_DEBUG or false)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.getenv("API_WORKERS", "1")),
        help="Number of worker processes (default: API_WORKERS or 1)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)

    # Import only after argument parsing so --help remains dependency-free. The
    # control API itself still fails closed when API_SECRET_KEY is absent.
    from trading.api.app import app

    logger = setup_logging(app_name="API", log_level="INFO", enable_file_logging=True)

    logger.info(f"Starting ELVIS Trading Bot API on {args.host}:{args.port}")
    logger.info(
        "API Documentation available at: http://localhost:{}/api/docs".format(args.port)
    )

    if args.debug:
        # Development server with debug mode
        logger.warning("Running in DEBUG mode - do not use in production!")
        app.run(host=args.host, port=args.port, debug=True)
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
                "bind": f"{args.host}:{args.port}",
                "workers": args.workers,
                "worker_class": "sync",
                "timeout": 120,
                "keepalive": 5,
                "accesslog": "-",
                "errorlog": "-",
                "loglevel": "info",
                "capture_output": True,
                "enable_stdio_inheritance": True,
            }

            logger.info(f"Starting production server with {args.workers} workers")
            StandaloneApplication(app, options).run()

        except ImportError:
            logger.warning(
                "Gunicorn not installed. Running with Flask development server."
            )
            logger.warning("For production, install gunicorn: pip install gunicorn")
            app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
