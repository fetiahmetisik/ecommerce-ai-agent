"""
Setup Firestore collections and indexes for E-Commerce AI Agent
"""

import logging
from google.cloud import firestore
from google.cloud.firestore import FieldFilter
from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_collections():
    """Setup Firestore collections with initial data and indexes"""
    try:
        # Initialize Firestore client
        db = firestore.Client(project=settings.google_cloud_project)
        
        logger.info("Setting up Firestore collections...")
        
        # Setup products collection
        setup_products_collection(db)
        
        # Setup users collection
        setup_users_collection(db)
        
        # Setup orders collection
        setup_orders_collection(db)
        
        # Setup other collections
        setup_support_collections(db)
        
        logger.info("Firestore setup completed successfully")
        
    except Exception as e:
        logger.error(f"Error setting up Firestore: {str(e)}")
        raise

def setup_products_collection(db):
    """Setup products collection with sample data"""
    logger.info("Setting up products collection...")
    
    products_ref = db.collection(settings.firestore_products_collection)
    
    # Sample products
    sample_products = [
        {
            "name": "Wireless Bluetooth Headphones",
            "description": "Premium quality wireless headphones with noise cancellation",
            "category": "Electronics",
            "subcategory": "Audio",
            "brand": "TechBrand",
            "sku": "WBH-001",
            "price": 299.99,
            "currency": "TRY",
            "discount_price": 249.99,
            "discount_percentage": 16.67,
            "stock_quantity": 150,
            "reserved_quantity": 5,
            "warehouse_locations": ["WH-IST-001", "WH-ANK-002"],
            "colors": ["black", "white", "blue"],
            "sizes": ["One Size"],
            "materials": ["plastic", "metal", "leather"],
            "tags": ["wireless", "bluetooth", "headphones", "audio", "music"],
            "seasons": ["all"],
            "images": [
                "https://example.com/images/headphones-1.jpg",
                "https://example.com/images/headphones-2.jpg"
            ],
            "thumbnail": "https://example.com/images/headphones-thumb.jpg",
            "status": "active",
            "is_featured": True,
            "is_new": True,
            "trending_score": 85.5,
            "rating": 4.5,
            "review_count": 234,
            "sales_count": 89,
            "views_count": 1250
        },
        {
            "name": "Summer Cotton Dress",
            "description": "Lightweight cotton dress perfect for summer days",
            "category": "Fashion",
            "subcategory": "Women's Clothing",
            "brand": "FashionCo",
            "sku": "SCD-001",
            "price": 89.99,
            "currency": "TRY",
            "stock_quantity": 75,
            "reserved_quantity": 2,
            "warehouse_locations": ["WH-IST-001"],
            "colors": ["red", "blue", "green", "yellow"],
            "sizes": ["XS", "S", "M", "L", "XL"],
            "materials": ["cotton"],
            "tags": ["dress", "summer", "cotton", "casual", "women"],
            "seasons": ["spring", "summer"],
            "images": [
                "https://example.com/images/dress-1.jpg",
                "https://example.com/images/dress-2.jpg"
            ],
            "thumbnail": "https://example.com/images/dress-thumb.jpg",
            "status": "active",
            "is_featured": False,
            "is_new": False,
            "trending_score": 72.3,
            "rating": 4.2,
            "review_count": 156,
            "sales_count": 67,
            "views_count": 890
        },
        {
            "name": "Gaming Laptop",
            "description": "High-performance gaming laptop with RTX graphics",
            "category": "Electronics",
            "subcategory": "Computers",
            "brand": "GameTech",
            "sku": "GL-001",
            "price": 2499.99,
            "currency": "TRY",
            "stock_quantity": 25,
            "reserved_quantity": 1,
            "warehouse_locations": ["WH-ANK-002"],
            "colors": ["black"],
            "sizes": ["15.6 inch"],
            "materials": ["aluminum", "plastic"],
            "tags": ["laptop", "gaming", "computer", "rtx", "high-performance"],
            "seasons": ["all"],
            "images": [
                "https://example.com/images/laptop-1.jpg",
                "https://example.com/images/laptop-2.jpg"
            ],
            "thumbnail": "https://example.com/images/laptop-thumb.jpg",
            "status": "active",
            "is_featured": True,
            "is_new": False,
            "trending_score": 91.2,
            "rating": 4.7,
            "review_count": 89,
            "sales_count": 23,
            "views_count": 567
        }
    ]
    
    # Add sample products
    for i, product in enumerate(sample_products):
        doc_ref = products_ref.document(f"product_{i+1}")
        doc_ref.set(product)
        logger.info(f"Added product: {product['name']}")

def setup_users_collection(db):
    """Setup users collection with sample data"""
    logger.info("Setting up users collection...")
    
    users_ref = db.collection(settings.firestore_users_collection)
    
    # Sample users
    sample_users = [
        {
            "email": "john.doe@example.com",
            "name": "John Doe",
            "age": 28,
            "gender": "male",
            "location": {"city": "Istanbul", "country": "Turkey"},
            "language": "tr",
            "preferences": {
                "favorite_categories": ["Electronics", "Sports"],
                "favorite_brands": ["TechBrand", "SportsCorp"],
                "color_preferences": ["black", "blue"],
                "price_range": {"min": 100, "max": 1000}
            },
            "segments": ["tech_enthusiast", "young_adult"],
            "total_purchases": 5,
            "total_spent": 1250.50,
            "lifetime_value": 2000.0,
            "is_active": True,
            "is_verified": True,
            "newsletter_subscribed": True
        },
        {
            "email": "jane.smith@example.com",
            "name": "Jane Smith",
            "age": 32,
            "gender": "female",
            "location": {"city": "Ankara", "country": "Turkey"},
            "language": "en",
            "preferences": {
                "favorite_categories": ["Fashion", "Beauty"],
                "favorite_brands": ["FashionCo", "BeautyBrand"],
                "color_preferences": ["red", "pink", "white"],
                "price_range": {"min": 50, "max": 500}
            },
            "segments": ["fashion_lover", "frequent_buyer"],
            "total_purchases": 12,
            "total_spent": 2100.75,
            "lifetime_value": 3500.0,
            "is_active": True,
            "is_verified": True,
            "newsletter_subscribed": True
        }
    ]
    
    # Add sample users
    for i, user in enumerate(sample_users):
        doc_ref = users_ref.document(f"user_{i+1}")
        doc_ref.set(user)
        logger.info(f"Added user: {user['name']}")

def setup_orders_collection(db):
    """Setup orders collection with sample data"""
    logger.info("Setting up orders collection...")
    
    orders_ref = db.collection(settings.firestore_orders_collection)
    
    # Sample orders
    sample_orders = [
        {
            "user_id": "user_1",
            "order_number": "ORD-2024-001",
            "items": [
                {
                    "product_id": "product_1",
                    "product_name": "Wireless Bluetooth Headphones",
                    "quantity": 1,
                    "unit_price": 249.99,
                    "total_price": 249.99,
                    "variant": {"color": "black"}
                }
            ],
            "subtotal": 249.99,
            "tax_amount": 44.99,
            "shipping_amount": 15.00,
            "discount_amount": 0.0,
            "total_amount": 309.98,
            "currency": "TRY",
            "shipping_address": {
                "name": "John Doe",
                "address_line1": "123 Tech Street",
                "city": "Istanbul",
                "postal_code": "34000",
                "country": "Turkey",
                "phone": "+90-555-123-4567"
            },
            "shipping_method": "Standard",
            "payment_method": "Credit Card",
            "payment_status": "completed",
            "status": "delivered"
        }
    ]
    
    # Add sample orders
    for i, order in enumerate(sample_orders):
        doc_ref = orders_ref.document(f"order_{i+1}")
        doc_ref.set(order)
        logger.info(f"Added order: {order['order_number']}")

def setup_support_collections(db):
    """Setup supporting collections"""
    logger.info("Setting up support collections...")
    
    # Setup suppliers collection
    suppliers_ref = db.collection("suppliers")
    sample_supplier = {
        "name": "TechSupplier Ltd.",
        "contact": {
            "email": "sales@techsupplier.com",
            "phone": "+90-212-555-0001",
            "address": "Industrial Zone, Istanbul, Turkey"
        },
        "lead_time_days": 7,
        "minimum_order_quantity": 10,
        "payment_terms": "Net 30",
        "reliability_score": 0.95,
        "quality_score": 0.92,
        "delivery_performance": 0.98,
        "is_active": True,
        "is_preferred": True
    }
    suppliers_ref.document("supplier_1").set(sample_supplier)
    
    # Create empty collections for other data
    collections_to_create = [
        "user_events",
        "stock_history", 
        "price_history",
        "alerts",
        "recommendations",
        "search_queries"
    ]
    
    for collection_name in collections_to_create:
        # Create a placeholder document to initialize collection
        placeholder_ref = db.collection(collection_name).document("placeholder")
        placeholder_ref.set({"_placeholder": True, "created_at": firestore.SERVER_TIMESTAMP})
        logger.info(f"Initialized collection: {collection_name}")

if __name__ == "__main__":
    setup_collections()