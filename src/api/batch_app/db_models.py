from flask_sqlalchemy import SQLAlchemy
import datetime

# initialize the SQLAlchemy object
db = SQLAlchemy()


class ImageClassifications(db.Model):
    """
    Model representing information on ingested images and image classification results stored in a database.

    Attributes:
        - img_id: Integer - Unique identifier for each image (Primary Key).
        - image_name: String - Name of the image file (Unique).
        - timestamp_ingestion: DateTime - Timestamp when the image was ingested into the system.
        - timestamp_prediction: DateTime - Timestamp when the prediction was made.
        - predicted_class: String - the class predicted for the image.
        - prob_tshirt_top: Float - Probability of the image being classified as a T-shirt/top.
        - prob_trouser: Float - Probability of the image being classified as a trouser.
        - prob_pullover: Float - Probability of the image being classified as a pullover.
        - prob_dress: Float - Probability of the image being classified as a dress.
        - prob_coat: Float - Probability of the image being classified as a coat.
        - prob_sandal: Float - Probability of the image being classified as a sandal.
        - prob_shirt: Float - Probability of the image being classified as a shirt.
        - prob_sneaker: Float - Probability of the image being classified as a sneaker.
        - prob_bag: Float - Probability of the image being classified as a bag.
        - prob_ankle_boot: Float - Probability of the image being classified as an ankle boot.
    """
    # define columns for the ImageClassifications table
    img_id = db.Column(db.Integer, primary_key=True)
    image_name = db.Column(db.String(255), unique=True)
    timestamp_ingestion = db.Column(db.DateTime, default=datetime.datetime.now(), nullable=False)
    timestamp_prediction = db.Column(db.DateTime, nullable=True)
    predicted_class = db.Column(db.String(80), nullable=True)
    prob_tshirt_top = db.Column(db.Float, nullable=True)
    prob_trouser = db.Column(db.Float, nullable=True)
    prob_pullover = db.Column(db.Float, nullable=True)
    prob_dress = db.Column(db.Float, nullable=True)
    prob_coat = db.Column(db.Float, nullable=True)
    prob_sandal = db.Column(db.Float, nullable=True)
    prob_shirt = db.Column(db.Float, nullable=True)
    prob_sneaker = db.Column(db.Float, nullable=True)
    prob_bag = db.Column(db.Float, nullable=True)
    prob_ankle_boot = db.Column(db.Float, nullable=True)

    def __repr__(self) -> str:
        """
        Provide a string representation of the ImageClassifications instance for debugging.
        """
        return f"<Image {self.image_name}>"
