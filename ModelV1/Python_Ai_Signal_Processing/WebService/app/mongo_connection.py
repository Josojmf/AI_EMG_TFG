import logging
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError, OperationFailure, ConfigurationError
from dotenv import load_dotenv
import os

# Cargar variables de entorno
load_dotenv()

# Variables de configuración de MongoDB
db_username = os.getenv("DB_USERNAME")
db_password = os.getenv("DB_PASSWORD")
db_cluster = os.getenv("DB_CLUSTER")  # ← Esto sigue siendo 'localhost' en tu .env
db_name = os.getenv("DB_NAME")

logger = logging.getLogger('app_logger')


def connect_to_mongo():
    """
    Connect to MongoDB and return the client, database, and users collection.
    """
    try:
        # Para entorno local con Docker Compose, usamos "mongo" como hostname
        mongo_uri = f"mongodb://{db_username}:{db_password}@mongo:27017/"

        # Establecer conexión
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        db = client[db_name]
        users_collection = db["Users"]

        # Test de conexión
        client.admin.command("ping")
        logger.info("Connected to MongoDB successfully.")

        return client, db, users_collection

    except (ServerSelectionTimeoutError, OperationFailure, ConfigurationError) as e:
        logger.error(f"Failed to connect to MongoDB: {e}", exc_info=True)
        return None, None, None
