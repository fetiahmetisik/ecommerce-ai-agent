"""
Pytest configuration and fixtures for E-Commerce AI Agent tests
"""

import pytest
import asyncio
import os
from typing import Generator, AsyncGenerator
from unittest.mock import Mock, AsyncMock, patch

# Set test environment
os.environ["TESTING"] = "true"
os.environ["DEBUG"] = "true"
os.environ["LOG_LEVEL"] = "DEBUG"
os.environ["GOOGLE_CLOUD_PROJECT"] = "test-project"
os.environ["API_KEY"] = "test-api-key"
os.environ["FIRESTORE_DATABASE"] = "test-db"
os.environ["GCS_BUCKET_NAME"] = "test-bucket"

from fastapi.testclient import TestClient
from main import app
from config import settings

@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def client() -> TestClient:
    """Create a test client for the FastAPI app"""
    return TestClient(app)

@pytest.fixture
def auth_headers() -> dict:
    """Authentication headers for API requests"""
    return {"Authorization": f"Bearer {settings.api_key}"}

# Mock Google Cloud clients
@pytest.fixture
def mock_firestore_client():
    """Mock Firestore client"""
    with patch('google.cloud.firestore.Client') as mock_client:
        mock_instance = Mock()
        mock_client.return_value = mock_instance
        
        # Mock collection and document methods
        mock_collection = Mock()
        mock_document = Mock()
        mock_instance.collection.return_value = mock_collection
        mock_collection.document.return_value = mock_document
        
        yield mock_instance

@pytest.fixture
def mock_vision_client():
    """Mock Vision API client"""
    with patch('google.cloud.vision.ImageAnnotatorClient') as mock_client:
        mock_instance = Mock()
        mock_client.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def mock_storage_client():
    """Mock Cloud Storage client"""
    with patch('google.cloud.storage.Client') as mock_client:
        mock_instance = Mock()
        mock_client.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def mock_vertex_ai():
    """Mock Vertex AI"""
    with patch('vertexai.init'), \
         patch('vertexai.generative_models.GenerativeModel') as mock_model:
        mock_instance = Mock()
        mock_model.return_value = mock_instance
        
        # Mock generate_content method
        mock_response = Mock()
        mock_response.text = '{"test": "response"}'
        mock_instance.generate_content.return_value = mock_response
        
        yield mock_instance

@pytest.fixture
def mock_langchain_llm():
    """Mock LangChain LLM"""
    with patch('langchain_google_vertexai.VertexAI') as mock_llm:
        mock_instance = Mock()
        mock_llm.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def mock_crewai_agent():
    """Mock CrewAI Agent"""
    with patch('crewai.Agent') as mock_agent:
        mock_instance = Mock()
        mock_agent.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def mock_crewai_task():
    """Mock CrewAI Task"""
    with patch('crewai.Task') as mock_task:
        mock_instance = AsyncMock()
        mock_instance.execute.return_value = "Test task result"
        mock_task.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def mock_crewai_crew():
    """Mock CrewAI Crew"""
    with patch('crewai.Crew') as mock_crew:
        mock_instance = Mock()
        mock_instance.kickoff.return_value = "Test crew result"
        mock_crew.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def sample_product_data():
    """Sample product data for testing"""
    return {
        "name": "Test Product",
        "description": "A test product for unit testing",
        "category": "Electronics",
        "brand": "TestBrand",
        "sku": "TEST-001",
        "price": 99.99,
        "currency": "TRY",
        "stock_quantity": 100,
        "colors": ["black", "white"],
        "sizes": ["M", "L"],
        "tags": ["test", "electronics"],
        "status": "active",
        "is_featured": False
    }

@pytest.fixture
def sample_user_data():
    """Sample user data for testing"""
    return {
        "email": "test@example.com",
        "name": "Test User",
        "age": 30,
        "gender": "other",
        "language": "en",
        "preferences": {
            "favorite_categories": ["Electronics"],
            "favorite_brands": ["TestBrand"],
            "price_range": {"min": 0, "max": 200}
        }
    }

@pytest.fixture
def sample_order_data():
    """Sample order data for testing"""
    return {
        "user_id": "test-user-123",
        "order_number": "ORD-TEST-001",
        "items": [
            {
                "product_id": "test-product-123",
                "product_name": "Test Product",
                "quantity": 2,
                "unit_price": 99.99,
                "total_price": 199.98
            }
        ],
        "subtotal": 199.98,
        "tax_amount": 36.00,
        "total_amount": 235.98,
        "currency": "TRY",
        "shipping_address": {
            "name": "Test User",
            "address_line1": "123 Test St",
            "city": "Test City",
            "postal_code": "12345",
            "country": "Turkey"
        },
        "status": "pending"
    }

@pytest.fixture
def sample_image_data():
    """Sample image data for testing"""
    # Create a simple test image (1x1 pixel PNG)
    import base64
    # Minimal PNG image data (1x1 transparent pixel)
    png_data = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xdb\x00\x00\x00\x00IEND\xaeB`\x82'
    return png_data

@pytest.fixture
def mock_redis():
    """Mock Redis client"""
    with patch('redis.Redis') as mock_redis:
        mock_instance = Mock()
        mock_redis.return_value = mock_instance
        
        # Mock Redis methods
        mock_instance.get.return_value = None
        mock_instance.set.return_value = True
        mock_instance.delete.return_value = 1
        mock_instance.exists.return_value = False
        
        yield mock_instance

@pytest.fixture(autouse=True)
def mock_all_external_services(
    mock_firestore_client,
    mock_vision_client,
    mock_storage_client,
    mock_vertex_ai,
    mock_langchain_llm,
    mock_crewai_agent,
    mock_crewai_task,
    mock_crewai_crew,
    mock_redis
):
    """Automatically mock all external services for all tests"""
    pass

# Helper functions for tests
def create_mock_firestore_document(data: dict, doc_id: str = "test-doc-id"):
    """Create a mock Firestore document"""
    mock_doc = Mock()
    mock_doc.exists = True
    mock_doc.id = doc_id
    mock_doc.to_dict.return_value = data
    return mock_doc

def create_mock_firestore_query_result(data_list: list):
    """Create a mock Firestore query result"""
    mock_docs = []
    for i, data in enumerate(data_list):
        mock_doc = create_mock_firestore_document(data, f"doc-{i}")
        mock_docs.append(mock_doc)
    return mock_docs

@pytest.fixture
def mock_vision_response():
    """Mock Vision API response"""
    from google.cloud.vision import AnnotateImageResponse, EntityAnnotation, ImageProperties, DominantColorsAnnotation, ColorInfo, Color, LocalizedObjectAnnotation, BoundingPoly, NormalizedVertex
    
    mock_response = Mock(spec=AnnotateImageResponse)
    
    # Mock label annotations
    mock_label = Mock(spec=EntityAnnotation)
    mock_label.description = "test_label"
    mock_label.score = 0.9
    mock_response.label_annotations = [mock_label]
    
    # Mock object annotations
    mock_object = Mock(spec=LocalizedObjectAnnotation)
    mock_object.name = "test_object"
    mock_object.score = 0.8
    mock_response.localized_object_annotations = [mock_object]
    
    # Mock image properties
    mock_color = Mock(spec=ColorInfo)
    mock_color.color = Mock(spec=Color)
    mock_color.color.red = 255
    mock_color.color.green = 0
    mock_color.color.blue = 0
    mock_color.score = 0.7
    mock_color.pixel_fraction = 0.5
    
    mock_dominant_colors = Mock(spec=DominantColorsAnnotation)
    mock_dominant_colors.colors = [mock_color]
    
    mock_image_props = Mock(spec=ImageProperties)
    mock_image_props.dominant_colors = mock_dominant_colors
    mock_response.image_properties_annotation = mock_image_props
    
    # Mock web detection
    mock_web_entity = Mock()
    mock_web_entity.description = "test_web_entity"
    mock_web_detection = Mock()
    mock_web_detection.web_entities = [mock_web_entity]
    mock_response.web_detection = mock_web_detection
    
    # Mock text annotations
    mock_text = Mock()
    mock_text.description = "test text"
    mock_response.text_annotations = [mock_text]
    
    return mock_response