import logging

from src.api.batch_app import create_app
from waitress import serve
from src.utils import setup_logging, load_app_config

# entry point for the application. This script initializes the Flask app and starts the server using Waitress.
if __name__ == "__main__":
    # set up logging
    setup_logging()
    app_logger = logging.getLogger("app_logger")
    # create the Flask application
    app = create_app(logger=app_logger, is_test=False)
    # load config
    app_config = load_app_config(is_test=False)

    # get host and port configuration
    host = app_config["flask"].get("host", "0.0.0.0")
    port = int(app_config["flask"].get("port", 5000))

    app_logger.info(f"Starting app on {host}:{port}...")

    # start the Flask application using the Waitress WSGI server
    serve(app, host=host, port=port)
