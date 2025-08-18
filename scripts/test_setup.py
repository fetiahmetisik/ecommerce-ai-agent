"""
Test setup script to verify all components are working
"""

import asyncio
import logging
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_configuration():
    """Test configuration loading"""
    logger.info("Testing configuration...")
    
    try:
        assert settings.google_cloud_project, "Google Cloud project not configured"
        assert settings.api_key, "API key not configured"
        logger.info("✓ Configuration loaded successfully")
        return True
    except Exception as e:
        logger.error(f"✗ Configuration test failed: {e}")
        return False

async def test_imports():
    """Test all imports work correctly"""
    logger.info("Testing imports...")
    
    try:
        # Test agent imports
        from agents.orchestrator import ECommerceOrchestrator
        from agents.visual_search_agent import VisualSearchAgent
        from agents.recommendation_agent import RecommendationAgent
        from agents.inventory_agent import InventoryAgent
        
        # Test model imports
        from models.schemas import Product, User, Order
        
        # Test utility imports
        from utils.monitoring import metrics_collector
        from utils.logging_config import setup_logging
        
        logger.info("✓ All imports successful")
        return True
    except Exception as e:
        logger.error(f"✗ Import test failed: {e}")
        return False

async def test_google_cloud_connectivity():
    """Test Google Cloud service connectivity"""
    logger.info("Testing Google Cloud connectivity...")
    
    try:
        # Test Firestore
        from google.cloud import firestore
        client = firestore.Client(project=settings.google_cloud_project)
        # Simple connectivity test
        collections = list(client.collections())
        logger.info(f"✓ Firestore connected - {len(collections)} collections found")
        
        # Test Cloud Storage
        from google.cloud import storage
        storage_client = storage.Client(project=settings.google_cloud_project)
        # List buckets to test connectivity
        buckets = list(storage_client.list_buckets())
        logger.info(f"✓ Cloud Storage connected - {len(buckets)} buckets found")
        
        return True
    except Exception as e:
        logger.error(f"✗ Google Cloud connectivity test failed: {e}")
        logger.info("Note: This is expected if running without proper GCP credentials")
        return False

async def test_fastapi_app():
    """Test FastAPI app initialization"""
    logger.info("Testing FastAPI app...")
    
    try:
        from main import app
        assert app is not None, "FastAPI app not initialized"
        logger.info("✓ FastAPI app initialized successfully")
        return True
    except Exception as e:
        logger.error(f"✗ FastAPI app test failed: {e}")
        return False

async def test_agent_initialization():
    """Test agent initialization (mocked)"""
    logger.info("Testing agent initialization...")
    
    try:
        # Mock the external dependencies for testing
        import unittest.mock as mock
        
        with mock.patch('google.cloud.firestore.Client'), \
             mock.patch('google.cloud.vision.ImageAnnotatorClient'), \
             mock.patch('google.cloud.storage.Client'), \
             mock.patch('vertexai.init'), \
             mock.patch('vertexai.generative_models.GenerativeModel'):
            
            from agents.visual_search_agent import VisualSearchAgent
            from agents.recommendation_agent import RecommendationAgent
            from agents.inventory_agent import InventoryAgent
            
            # Test agent creation
            visual_agent = VisualSearchAgent()
            rec_agent = RecommendationAgent()
            inv_agent = InventoryAgent()
            
            logger.info("✓ All agents initialized successfully (mocked)")
            return True
    except Exception as e:
        logger.error(f"✗ Agent initialization test failed: {e}")
        return False

async def test_monitoring_setup():
    """Test monitoring setup"""
    logger.info("Testing monitoring setup...")
    
    try:
        from utils.monitoring import init_monitoring, metrics_collector, health_checker
        
        # Test metrics collector
        assert metrics_collector is not None
        
        # Test health checker
        assert health_checker is not None
        
        logger.info("✓ Monitoring components initialized")
        return True
    except Exception as e:
        logger.error(f"✗ Monitoring test failed: {e}")
        return False

async def test_logging_setup():
    """Test logging setup"""
    logger.info("Testing logging setup...")
    
    try:
        from utils.logging_config import setup_logging, get_logger
        
        # Test logging setup
        setup_logging()
        
        # Test logger creation
        test_logger = get_logger("test")
        test_logger.info("Test log message")
        
        logger.info("✓ Logging setup successful")
        return True
    except Exception as e:
        logger.error(f"✗ Logging test failed: {e}")
        return False

async def test_database_models():
    """Test database model creation"""
    logger.info("Testing database models...")
    
    try:
        from models.schemas import Product, User, Order, OrderItem, ShippingAddress
        
        # Test product creation
        product = Product(
            name="Test Product",
            category="Electronics",
            brand="TestBrand",
            sku="TEST-001",
            price=99.99
        )
        assert product.name == "Test Product"
        
        # Test user creation
        user = User(
            email="test@example.com",
            name="Test User"
        )
        assert user.email == "test@example.com"
        
        # Test order creation
        shipping_address = ShippingAddress(
            name="Test User",
            address_line1="123 Test St",
            city="Test City",
            postal_code="12345",
            country="Test Country"
        )
        
        order_item = OrderItem(
            product_id="test-product",
            product_name="Test Product",
            quantity=1,
            unit_price=99.99,
            total_price=99.99
        )
        
        order = Order(
            user_id="test-user",
            order_number="TEST-001",
            items=[order_item],
            subtotal=99.99,
            total_amount=99.99,
            shipping_address=shipping_address
        )
        assert order.order_number == "TEST-001"
        
        logger.info("✓ Database models working correctly")
        return True
    except Exception as e:
        logger.error(f"✗ Database models test failed: {e}")
        return False

async def run_all_tests():
    """Run all tests"""
    logger.info("🧪 Starting E-Commerce AI Agent test suite...")
    logger.info("=" * 60)
    
    tests = [
        ("Configuration", test_configuration),
        ("Imports", test_imports),
        ("Database Models", test_database_models),
        ("FastAPI App", test_fastapi_app),
        ("Agent Initialization", test_agent_initialization),
        ("Logging Setup", test_logging_setup),
        ("Monitoring Setup", test_monitoring_setup),
        ("Google Cloud Connectivity", test_google_cloud_connectivity),
    ]
    
    results = {}
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results[test_name] = result
            if result:
                passed += 1
        except Exception as e:
            logger.error(f"✗ {test_name} test crashed: {e}")
            results[test_name] = False
        
        print()  # Add spacing between tests
    
    # Print summary
    logger.info("=" * 60)
    logger.info("🎯 TEST SUMMARY")
    logger.info("=" * 60)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{test_name:<25} : {status}")
    
    logger.info("-" * 60)
    logger.info(f"Total: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        logger.info("🎉 All tests passed! The application is ready to run.")
    elif passed >= total * 0.7:
        logger.info("⚠️  Most tests passed. Some optional features may not work.")
    else:
        logger.error("❌ Many tests failed. Please check configuration and dependencies.")
    
    return passed, total

async def main():
    """Main test function"""
    try:
        passed, total = await run_all_tests()
        
        if passed >= total * 0.7:
            logger.info("\n🚀 Next steps:")
            logger.info("1. Copy .env.example to .env and configure your settings")
            logger.info("2. Set up Google Cloud credentials")
            logger.info("3. Run: python main.py")
            logger.info("4. Access API docs at: http://localhost:8000/docs")
        
        # Exit with appropriate code
        sys.exit(0 if passed == total else 1)
        
    except KeyboardInterrupt:
        logger.info("\n🛑 Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"💥 Test suite crashed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())