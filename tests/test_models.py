"""
Tests for data models and schemas
"""

import pytest
from datetime import datetime
from pydantic import ValidationError

from models.schemas import (
    Product, User, Order, OrderItem, ShippingAddress,
    UserEvent, StockHistory, PriceHistory, Alert,
    Supplier, RecommendationContext, Recommendation,
    ProductStatus, OrderStatus, AlertType, AlertPriority
)

class TestProductModel:
    """Tests for Product model"""
    
    def test_valid_product_creation(self):
        """Test creating a valid product"""
        product_data = {
            "name": "Test Product",
            "description": "A test product",
            "category": "Electronics",
            "brand": "TestBrand",
            "sku": "TEST-001",
            "price": 99.99,
            "currency": "TRY"
        }
        
        product = Product(**product_data)
        
        assert product.name == "Test Product"
        assert product.price == 99.99
        assert product.currency == "TRY"
        assert product.stock_quantity == 0  # Default value
        assert product.status == ProductStatus.ACTIVE
        assert isinstance(product.created_at, datetime)
    
    def test_product_with_all_fields(self):
        """Test product with all optional fields"""
        product_data = {
            "name": "Complete Product",
            "description": "A complete test product",
            "category": "Electronics",
            "subcategory": "Smartphones",
            "brand": "TestBrand",
            "sku": "COMPLETE-001",
            "price": 899.99,
            "currency": "TRY",
            "discount_price": 799.99,
            "discount_percentage": 11.11,
            "stock_quantity": 50,
            "reserved_quantity": 5,
            "warehouse_locations": ["WH-001", "WH-002"],
            "supplier_id": "SUP-001",
            "colors": ["black", "white", "blue"],
            "sizes": ["S", "M", "L", "XL"],
            "materials": ["plastic", "metal"],
            "tags": ["smartphone", "electronics", "mobile"],
            "seasons": ["all"],
            "images": ["img1.jpg", "img2.jpg"],
            "thumbnail": "thumb.jpg",
            "is_featured": True,
            "is_new": True,
            "trending_score": 85.5,
            "rating": 4.5,
            "review_count": 123
        }
        
        product = Product(**product_data)
        
        assert product.subcategory == "Smartphones"
        assert product.discount_price == 799.99
        assert len(product.colors) == 3
        assert len(product.warehouse_locations) == 2
        assert product.is_featured is True
        assert product.trending_score == 85.5
    
    def test_product_validation_errors(self):
        """Test product validation errors"""
        # Missing required fields
        with pytest.raises(ValidationError):
            Product()
        
        # Invalid price
        with pytest.raises(ValidationError):
            Product(
                name="Test",
                category="Test",
                brand="Test",
                sku="TEST",
                price=-10.0  # Negative price should be invalid
            )
    
    def test_product_status_enum(self):
        """Test product status enum validation"""
        product_data = {
            "name": "Test Product",
            "category": "Electronics",
            "brand": "TestBrand",
            "sku": "TEST-001",
            "price": 99.99,
            "status": "active"
        }
        
        product = Product(**product_data)
        assert product.status == ProductStatus.ACTIVE
        
        # Test invalid status
        product_data["status"] = "invalid_status"
        with pytest.raises(ValidationError):
            Product(**product_data)

class TestUserModel:
    """Tests for User model"""
    
    def test_valid_user_creation(self):
        """Test creating a valid user"""
        user_data = {
            "email": "test@example.com",
            "name": "Test User"
        }
        
        user = User(**user_data)
        
        assert user.email == "test@example.com"
        assert user.name == "Test User"
        assert user.language == "tr"  # Default value
        assert user.is_active is True
        assert isinstance(user.created_at, datetime)
    
    def test_user_with_preferences(self):
        """Test user with preferences"""
        user_data = {
            "email": "test@example.com",
            "name": "Test User",
            "age": 25,
            "gender": "female",
            "location": {"city": "Istanbul", "country": "Turkey"},
            "preferences": {
                "favorite_categories": ["Electronics", "Fashion"],
                "favorite_brands": ["Apple", "Nike"],
                "price_range": {"min": 100, "max": 1000}
            },
            "segments": ["young_adult", "tech_savvy"]
        }
        
        user = User(**user_data)
        
        assert user.age == 25
        assert len(user.preferences.favorite_categories) == 2
        assert user.preferences.price_range["max"] == 1000
        assert "tech_savvy" in user.segments
    
    def test_user_email_validation(self):
        """Test user email validation"""
        # Valid email
        user = User(email="valid@example.com", name="Test")
        assert user.email == "valid@example.com"
        
        # Invalid email should still work (basic validation)
        # Note: Pydantic doesn't validate email format by default unless specified

class TestOrderModel:
    """Tests for Order model"""
    
    def test_valid_order_creation(self):
        """Test creating a valid order"""
        shipping_address = ShippingAddress(
            name="John Doe",
            address_line1="123 Test St",
            city="Test City",
            postal_code="12345",
            country="Turkey"
        )
        
        order_item = OrderItem(
            product_id="prod-123",
            product_name="Test Product",
            quantity=2,
            unit_price=50.0,
            total_price=100.0
        )
        
        order_data = {
            "user_id": "user-123",
            "order_number": "ORD-001",
            "items": [order_item],
            "subtotal": 100.0,
            "total_amount": 118.0,
            "shipping_address": shipping_address
        }
        
        order = Order(**order_data)
        
        assert order.user_id == "user-123"
        assert order.order_number == "ORD-001"
        assert len(order.items) == 1
        assert order.total_amount == 118.0
        assert order.status == OrderStatus.PENDING
    
    def test_order_item_validation(self):
        """Test order item validation"""
        # Valid order item
        item = OrderItem(
            product_id="prod-123",
            product_name="Test Product",
            quantity=1,
            unit_price=99.99,
            total_price=99.99
        )
        
        assert item.quantity == 1
        assert item.total_price == 99.99
        
        # Invalid quantity
        with pytest.raises(ValidationError):
            OrderItem(
                product_id="prod-123",
                product_name="Test Product",
                quantity=0,  # Should be positive
                unit_price=99.99,
                total_price=99.99
            )

class TestEventModel:
    """Tests for UserEvent model"""
    
    def test_user_event_creation(self):
        """Test creating user events"""
        event_data = {
            "user_id": "user-123",
            "event_type": "product_view",
            "product_id": "prod-123",
            "category": "Electronics",
            "event_data": {"duration": 30, "source": "search"},
            "session_id": "session-123"
        }
        
        event = UserEvent(**event_data)
        
        assert event.user_id == "user-123"
        assert event.event_type == "product_view"
        assert event.event_data["duration"] == 30
        assert isinstance(event.timestamp, datetime)

class TestHistoryModels:
    """Tests for history tracking models"""
    
    def test_stock_history(self):
        """Test stock history model"""
        history_data = {
            "product_id": "prod-123",
            "previous_quantity": 100,
            "new_quantity": 80,
            "change": -20,
            "operation": "subtract",
            "reason": "Sale"
        }
        
        history = StockHistory(**history_data)
        
        assert history.product_id == "prod-123"
        assert history.change == -20
        assert history.operation == "subtract"
        assert isinstance(history.timestamp, datetime)
    
    def test_price_history(self):
        """Test price history model"""
        history_data = {
            "product_id": "prod-123",
            "previous_price": 100.0,
            "new_price": 90.0,
            "change_amount": -10.0,
            "change_percentage": -10.0,
            "reason": "Promotion"
        }
        
        history = PriceHistory(**history_data)
        
        assert history.product_id == "prod-123"
        assert history.change_amount == -10.0
        assert history.change_percentage == -10.0
        assert isinstance(history.effective_date, datetime)

class TestAlertModel:
    """Tests for Alert model"""
    
    def test_alert_creation(self):
        """Test creating alerts"""
        alert_data = {
            "product_id": "prod-123",
            "alert_type": AlertType.LOW_STOCK,
            "priority": AlertPriority.HIGH,
            "title": "Low Stock Alert",
            "message": "Product stock is running low",
            "details": {"current_stock": 5, "threshold": 10}
        }
        
        alert = Alert(**alert_data)
        
        assert alert.alert_type == AlertType.LOW_STOCK
        assert alert.priority == AlertPriority.HIGH
        assert alert.details["current_stock"] == 5
        assert alert.status == "active"
        assert alert.acknowledged is False
    
    def test_alert_enum_validation(self):
        """Test alert enum validation"""
        # Valid alert type
        alert = Alert(
            alert_type="low_stock",
            priority="high",
            title="Test Alert",
            message="Test message"
        )
        assert alert.alert_type == AlertType.LOW_STOCK
        
        # Invalid alert type
        with pytest.raises(ValidationError):
            Alert(
                alert_type="invalid_type",
                priority="high",
                title="Test Alert",
                message="Test message"
            )

class TestSupplierModel:
    """Tests for Supplier model"""
    
    def test_supplier_creation(self):
        """Test creating supplier"""
        supplier_data = {
            "name": "Test Supplier Ltd.",
            "contact": {
                "email": "contact@testsupplier.com",
                "phone": "+90-555-123-4567",
                "address": "123 Supplier St, Istanbul"
            },
            "lead_time_days": 14,
            "minimum_order_quantity": 50,
            "payment_terms": "Net 45",
            "reliability_score": 0.95,
            "is_preferred": True
        }
        
        supplier = Supplier(**supplier_data)
        
        assert supplier.name == "Test Supplier Ltd."
        assert supplier.lead_time_days == 14
        assert supplier.reliability_score == 0.95
        assert supplier.is_preferred is True
        assert isinstance(supplier.created_at, datetime)

class TestRecommendationModels:
    """Tests for recommendation models"""
    
    def test_recommendation_context(self):
        """Test recommendation context"""
        context_data = {
            "user_id": "user-123",
            "session_id": "session-456",
            "current_page": "product_detail",
            "search_query": "wireless headphones",
            "cart_items": ["prod-1", "prod-2"],
            "recently_viewed": ["prod-3", "prod-4", "prod-5"]
        }
        
        context = RecommendationContext(**context_data)
        
        assert context.user_id == "user-123"
        assert context.search_query == "wireless headphones"
        assert len(context.cart_items) == 2
        assert len(context.recently_viewed) == 3
    
    def test_recommendation(self):
        """Test individual recommendation"""
        rec_data = {
            "product_id": "prod-123",
            "score": 0.85,
            "reason": "Based on your purchase history",
            "algorithm": "collaborative_filtering",
            "context": {"category_match": True}
        }
        
        recommendation = Recommendation(**rec_data)
        
        assert recommendation.product_id == "prod-123"
        assert recommendation.score == 0.85
        assert recommendation.algorithm == "collaborative_filtering"
        assert recommendation.context["category_match"] is True

def test_model_serialization():
    """Test model serialization to dict"""
    product = Product(
        name="Test Product",
        category="Electronics",
        brand="TestBrand",
        sku="TEST-001",
        price=99.99
    )
    
    product_dict = product.dict()
    
    assert isinstance(product_dict, dict)
    assert product_dict["name"] == "Test Product"
    assert product_dict["price"] == 99.99
    assert "created_at" in product_dict

def test_model_json_serialization():
    """Test model JSON serialization"""
    product = Product(
        name="Test Product",
        category="Electronics",
        brand="TestBrand",
        sku="TEST-001",
        price=99.99
    )
    
    json_str = product.json()
    
    assert isinstance(json_str, str)
    assert "Test Product" in json_str
    assert "99.99" in json_str