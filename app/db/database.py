import motor.motor_asyncio
from typing import Optional
from app.config import settings # We will update this file next

class MongoDatabase:
    """
    A class to manage the MongoDB client and database instances.
    This acts as a global state holder for our connection.
    """
    client: Optional[motor.motor_asyncio.AsyncIOMotorClient] = None
    db: Optional[motor.motor_asyncio.AsyncIOMotorDatabase] = None

# Create a single, global instance that our app will use
db = MongoDatabase()

async def connect_to_database():
    """
    Called by the 'lifespan' event in main.py.
    Initializes the MongoDB client and database object.
    """
    print("--- DB: Connecting to MongoDB... ---")
    try:
        db.client = motor.motor_asyncio.AsyncIOMotorClient(
            settings.MONGO_CONNECTION_STRING
        )
        # Ping the server to verify connection
        await db.client.admin.command('ping') 
        db.db = db.client[settings.MONGO_DB_NAME]
        print(f"--- DB: Connected to MongoDB, database: '{settings.MONGO_DB_NAME}' ---")
    except Exception as e:
        print(f"--- DB: FAILED to connect to MongoDB. Error: {e} ---")
        # In a real app, you might want to exit if the DB connection fails
        raise

async def close_database_connection():
    """
    Called by the 'lifespan' event in main.py.
    Closes the MongoDB connection.
    """
    print("--- DB: Closing MongoDB connection... ---")
    if db.client:
        db.client.close()
        print("--- DB: MongoDB connection closed. ---")

async def get_db() -> motor.motor_asyncio.AsyncIOMotorDatabase:
    """
    A FastAPI dependency that injects the database object into
    service classes (like SessionService).
    """
    if db.db is None:
        # This should not happen if the lifespan event is working
        print("--- DB: ERROR! Database is not initialized. ---")
        raise Exception("Database not initialized. Check application startup logs.")
    return db.db