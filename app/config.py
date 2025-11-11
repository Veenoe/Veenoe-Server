from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """
    Loads environment variables from a .env file.
    """
    # Sarvam API Key
    SARVAM_API_KEY: str
    
    # Google Gemini API Key
    GOOGLE_API_KEY: str
    
    # MongoDB Settings
    MONGO_CONNECTION_STRING: str
    MONGO_DB_NAME: str

    class Config:
        env_file = ".env"

# Create a single instance to be imported by other modules
settings = Settings()