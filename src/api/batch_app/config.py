import os


class Config:
    # define the SECRET_Key for securely signing the session cookie
    SECRET_KEY = os.environ.get("SECRET_KEY") or "a_secret_key"
    # disable the signaling support of SQLAlchemy for object modifications
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    # limit maximum upload to 20 MB
    MAX_CONTENT_LENGTH = 20 * 1024 * 1024
