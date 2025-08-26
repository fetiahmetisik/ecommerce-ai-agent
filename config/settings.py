import os
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    """Application settings with validation"""
    
    # Google Cloud
    google_cloud_project: str = Field(..., env="GOOGLE_CLOUD_PROJECT")
    google_credentials_path: Optional[str] = Field(None, env="GOOGLE_APPLICATION_CREDENTIALS")
    location: str = Field("us-central1", env="LOCATION")
    
    # Vertex AI
    vertex_ai_model: str = Field("gemini-1.5-pro", env="VERTEX_AI_MODEL")
    vertex_ai_temperature: float = Field(0.7, env="VERTEX_AI_TEMPERATURE")
    vertex_ai_max_tokens: int = Field(8192, env="VERTEX_AI_MAX_TOKENS")
    
    # Firestore
    firestore_database: str = Field("(default)", env="FIRESTORE_DATABASE")
    firestore_products_collection: str = Field("products", env="FIRESTORE_COLLECTION_PRODUCTS")
    firestore_users_collection: str = Field("users", env="FIRESTORE_COLLECTION_USERS")
    firestore_orders_collection: str = Field("orders", env="FIRESTORE_COLLECTION_ORDERS")
    
    # Cloud Storage
    gcs_bucket_name: str = Field(..., env="GCS_BUCKET_NAME")
    gcs_product_images_path: str = Field("product-images/", env="GCS_PRODUCT_IMAGES_PATH")
    
    # Vision API
    vision_api_enabled: bool = Field(True, env="VISION_API_ENABLED")
    vision_confidence_threshold: float = Field(0.7, env="VISION_CONFIDENCE_THRESHOLD")
    
    # Translation
    translation_api_enabled: bool = Field(True, env="TRANSLATION_API_ENABLED")
    default_language: str = Field("tr", env="DEFAULT_LANGUAGE")
    supported_languages: List[str] = Field(
        default_factory=lambda: ["en", "tr", "de", "fr", "es", "ar", "ru", "zh", "ja", "ko"],
        env="SUPPORTED_LANGUAGES"
    )
    
    # Redis
    redis_host: str = Field("localhost", env="REDIS_HOST")
    redis_port: int = Field(6379, env="REDIS_PORT")
    redis_db: int = Field(0, env="REDIS_DB")
    redis_password: Optional[str] = Field(None, env="REDIS_PASSWORD")
    
    # API
    api_host: str = Field("0.0.0.0", env="API_HOST")
    api_port: int = Field(8000, env="API_PORT")
    api_version: str = Field("v1", env="API_VERSION")
    api_key: str = Field(..., env="API_KEY")
    cors_origins: List[str] = Field(
        default_factory=lambda: ["http://localhost:3000"],
        env="CORS_ORIGINS"
    )
    
    # Agent
    agent_max_iterations: int = Field(10, env="AGENT_MAX_ITERATIONS")
    agent_timeout_seconds: int = Field(30, env="AGENT_TIMEOUT_SECONDS")
    enable_memory: bool = Field(True, env="ENABLE_MEMORY")
    memory_type: str = Field("firestore", env="MEMORY_TYPE")
    
    # Monitoring
    enable_monitoring: bool = Field(True, env="ENABLE_MONITORING")
    otel_endpoint: str = Field("http://localhost:4317", env="OTEL_EXPORTER_OTLP_ENDPOINT")
    prometheus_port: int = Field(8001, env="PROMETHEUS_PORT")
    
    # Development
    debug: bool = Field(False, env="DEBUG")
    log_level: str = Field("INFO", env="LOG_LEVEL")
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()