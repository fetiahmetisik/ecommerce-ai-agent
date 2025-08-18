"""
Database schemas and models for Firestore collections
"""

from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum

class ProductStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    OUT_OF_STOCK = "out_of_stock"
    DISCONTINUED = "discontinued"

class OrderStatus(str, Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    PROCESSING = "processing"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"
    REFUNDED = "refunded"

class AlertType(str, Enum):
    LOW_STOCK = "low_stock"
    OUT_OF_STOCK = "out_of_stock"
    PRICE_CHANGE = "price_change"
    DEMAND_SPIKE = "demand_spike"
    SYSTEM_ERROR = "system_error"

class AlertPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

# Product Models
class Product(BaseModel):
    """Product schema for Firestore"""
    name: str = Field(..., description="Product name")
    description: str = Field("", description="Product description")
    category: str = Field(..., description="Product category")
    subcategory: Optional[str] = Field(None, description="Product subcategory")
    brand: str = Field(..., description="Product brand")
    sku: str = Field(..., description="Stock keeping unit")
    
    # Pricing
    price: float = Field(..., description="Current price")
    currency: str = Field("TRY", description="Currency code")
    discount_price: Optional[float] = Field(None, description="Discounted price")
    discount_percentage: float = Field(0, description="Discount percentage")
    price_tier: str = Field("standard", description="Price tier")
    
    # Inventory
    stock_quantity: int = Field(0, description="Current stock quantity")
    reserved_quantity: int = Field(0, description="Reserved stock quantity")
    warehouse_locations: List[str] = Field(default_factory=list, description="Warehouse locations")
    supplier_id: Optional[str] = Field(None, description="Supplier identifier")
    
    # Attributes
    colors: List[str] = Field(default_factory=list, description="Available colors")
    sizes: List[str] = Field(default_factory=list, description="Available sizes")
    materials: List[str] = Field(default_factory=list, description="Materials")
    tags: List[str] = Field(default_factory=list, description="Product tags")
    seasons: List[str] = Field(default_factory=list, description="Applicable seasons")
    
    # Images and media
    images: List[str] = Field(default_factory=list, description="Image URLs")
    thumbnail: Optional[str] = Field(None, description="Thumbnail image URL")
    videos: List[str] = Field(default_factory=list, description="Video URLs")
    
    # Analytics
    views_count: int = Field(0, description="Number of views")
    sales_count: int = Field(0, description="Number of sales")
    trending_score: float = Field(0.0, description="Trending score")
    rating: float = Field(0.0, description="Average rating")
    review_count: int = Field(0, description="Number of reviews")
    
    # SEO and search
    meta_title: Optional[str] = Field(None, description="SEO title")
    meta_description: Optional[str] = Field(None, description="SEO description")
    search_keywords: List[str] = Field(default_factory=list, description="Search keywords")
    
    # Status and tracking
    status: ProductStatus = Field(ProductStatus.ACTIVE, description="Product status")
    is_featured: bool = Field(False, description="Is featured product")
    is_new: bool = Field(False, description="Is new product")
    is_bestseller: bool = Field(False, description="Is bestseller")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    stock_updated_at: Optional[datetime] = Field(None, description="Stock last updated")
    price_updated_at: Optional[datetime] = Field(None, description="Price last updated")
    
    # Competitor data
    competitor_prices: Dict[str, float] = Field(default_factory=dict, description="Competitor prices")
    
    class Config:
        use_enum_values = True

# User Models
class UserPreferences(BaseModel):
    """User preferences model"""
    favorite_categories: List[str] = Field(default_factory=list)
    favorite_brands: List[str] = Field(default_factory=list)
    color_preferences: List[str] = Field(default_factory=list)
    style_preferences: List[str] = Field(default_factory=list)
    price_range: Dict[str, float] = Field(default_factory=dict)
    size_preferences: Dict[str, str] = Field(default_factory=dict)

class User(BaseModel):
    """User schema for Firestore"""
    email: str = Field(..., description="User email")
    name: str = Field(..., description="User name")
    phone: Optional[str] = Field(None, description="Phone number")
    
    # Demographics
    age: Optional[int] = Field(None, description="User age")
    gender: Optional[str] = Field(None, description="User gender")
    location: Optional[Dict[str, str]] = Field(None, description="User location")
    language: str = Field("tr", description="Preferred language")
    
    # Preferences
    preferences: UserPreferences = Field(default_factory=UserPreferences)
    computed_preferences: Dict[str, Any] = Field(default_factory=dict)
    segments: List[str] = Field(default_factory=list, description="User segments")
    
    # Analytics
    total_purchases: int = Field(0, description="Total number of purchases")
    total_spent: float = Field(0.0, description="Total amount spent")
    last_purchase_date: Optional[datetime] = Field(None, description="Last purchase date")
    lifetime_value: float = Field(0.0, description="Customer lifetime value")
    
    # Engagement
    registration_date: datetime = Field(default_factory=datetime.now)
    last_login: Optional[datetime] = Field(None, description="Last login timestamp")
    login_count: int = Field(0, description="Number of logins")
    
    # Status
    is_active: bool = Field(True, description="Is user active")
    is_verified: bool = Field(False, description="Is email verified")
    newsletter_subscribed: bool = Field(False, description="Newsletter subscription")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

# Order Models
class OrderItem(BaseModel):
    """Order item model"""
    product_id: str = Field(..., description="Product identifier")
    product_name: str = Field(..., description="Product name")
    quantity: int = Field(..., description="Quantity ordered")
    unit_price: float = Field(..., description="Unit price at time of order")
    total_price: float = Field(..., description="Total price for this item")
    variant: Optional[Dict[str, str]] = Field(None, description="Product variant (size, color, etc.)")

class ShippingAddress(BaseModel):
    """Shipping address model"""
    name: str = Field(..., description="Recipient name")
    address_line1: str = Field(..., description="Address line 1")
    address_line2: Optional[str] = Field(None, description="Address line 2")
    city: str = Field(..., description="City")
    state: Optional[str] = Field(None, description="State/Province")
    postal_code: str = Field(..., description="Postal code")
    country: str = Field(..., description="Country")
    phone: Optional[str] = Field(None, description="Phone number")

class Order(BaseModel):
    """Order schema for Firestore"""
    user_id: str = Field(..., description="User identifier")
    order_number: str = Field(..., description="Unique order number")
    
    # Items
    items: List[OrderItem] = Field(..., description="Order items")
    
    # Pricing
    subtotal: float = Field(..., description="Subtotal amount")
    tax_amount: float = Field(0.0, description="Tax amount")
    shipping_amount: float = Field(0.0, description="Shipping amount")
    discount_amount: float = Field(0.0, description="Discount amount")
    total_amount: float = Field(..., description="Total order amount")
    currency: str = Field("TRY", description="Currency code")
    
    # Shipping
    shipping_address: ShippingAddress = Field(..., description="Shipping address")
    shipping_method: Optional[str] = Field(None, description="Shipping method")
    tracking_number: Optional[str] = Field(None, description="Tracking number")
    estimated_delivery: Optional[datetime] = Field(None, description="Estimated delivery date")
    actual_delivery: Optional[datetime] = Field(None, description="Actual delivery date")
    
    # Payment
    payment_method: Optional[str] = Field(None, description="Payment method")
    payment_status: str = Field("pending", description="Payment status")
    payment_id: Optional[str] = Field(None, description="Payment transaction ID")
    
    # Status and tracking
    status: OrderStatus = Field(OrderStatus.PENDING, description="Order status")
    notes: Optional[str] = Field(None, description="Order notes")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    shipped_at: Optional[datetime] = Field(None)
    delivered_at: Optional[datetime] = Field(None)
    
    class Config:
        use_enum_values = True

# Event Models
class UserEvent(BaseModel):
    """User event tracking model"""
    user_id: str = Field(..., description="User identifier")
    event_type: str = Field(..., description="Event type (view, click, purchase, etc.)")
    product_id: Optional[str] = Field(None, description="Product identifier if applicable")
    category: Optional[str] = Field(None, description="Product category if applicable")
    
    # Event data
    event_data: Dict[str, Any] = Field(default_factory=dict, description="Additional event data")
    session_id: Optional[str] = Field(None, description="Session identifier")
    user_agent: Optional[str] = Field(None, description="User agent string")
    ip_address: Optional[str] = Field(None, description="IP address")
    referrer: Optional[str] = Field(None, description="Referrer URL")
    
    # Timestamps
    timestamp: datetime = Field(default_factory=datetime.now)

# History Models
class StockHistory(BaseModel):
    """Stock level history model"""
    product_id: str = Field(..., description="Product identifier")
    previous_quantity: int = Field(..., description="Previous stock quantity")
    new_quantity: int = Field(..., description="New stock quantity")
    change: int = Field(..., description="Change amount")
    operation: str = Field(..., description="Operation type")
    reason: str = Field("", description="Reason for change")
    timestamp: datetime = Field(default_factory=datetime.now)

class PriceHistory(BaseModel):
    """Price change history model"""
    product_id: str = Field(..., description="Product identifier")
    previous_price: float = Field(..., description="Previous price")
    new_price: float = Field(..., description="New price")
    change_amount: float = Field(..., description="Price change amount")
    change_percentage: float = Field(..., description="Price change percentage")
    reason: str = Field("", description="Reason for change")
    effective_date: datetime = Field(default_factory=datetime.now, description="When change takes effect")
    timestamp: datetime = Field(default_factory=datetime.now)

# Alert Models
class Alert(BaseModel):
    """Alert model for inventory and system notifications"""
    product_id: Optional[str] = Field(None, description="Product identifier if applicable")
    alert_type: AlertType = Field(..., description="Type of alert")
    priority: AlertPriority = Field(..., description="Alert priority")
    
    # Alert content
    title: str = Field(..., description="Alert title")
    message: str = Field(..., description="Alert message")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional alert details")
    
    # Status
    status: str = Field("active", description="Alert status")
    acknowledged: bool = Field(False, description="Has been acknowledged")
    acknowledged_by: Optional[str] = Field(None, description="Who acknowledged the alert")
    acknowledged_at: Optional[datetime] = Field(None, description="When acknowledged")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    expires_at: Optional[datetime] = Field(None, description="When alert expires")
    
    class Config:
        use_enum_values = True

# Supplier Models
class Supplier(BaseModel):
    """Supplier information model"""
    name: str = Field(..., description="Supplier name")
    contact: Dict[str, str] = Field(default_factory=dict, description="Contact information")
    
    # Terms
    lead_time_days: int = Field(7, description="Lead time in days")
    minimum_order_quantity: int = Field(1, description="Minimum order quantity")
    payment_terms: str = Field("Net 30", description="Payment terms")
    
    # Performance
    reliability_score: float = Field(0.9, description="Reliability score (0-1)")
    quality_score: float = Field(0.9, description="Quality score (0-1)")
    delivery_performance: float = Field(0.9, description="Delivery performance (0-1)")
    
    # Tracking
    total_orders: int = Field(0, description="Total orders placed")
    last_order_date: Optional[datetime] = Field(None, description="Last order date")
    
    # Status
    is_active: bool = Field(True, description="Is supplier active")
    is_preferred: bool = Field(False, description="Is preferred supplier")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

# Recommendation Models
class RecommendationContext(BaseModel):
    """Context for recommendation generation"""
    user_id: str = Field(..., description="User identifier")
    session_id: Optional[str] = Field(None, description="Session identifier")
    current_page: Optional[str] = Field(None, description="Current page context")
    search_query: Optional[str] = Field(None, description="Current search query")
    cart_items: List[str] = Field(default_factory=list, description="Items in cart")
    recently_viewed: List[str] = Field(default_factory=list, description="Recently viewed products")

class Recommendation(BaseModel):
    """Individual recommendation model"""
    product_id: str = Field(..., description="Recommended product ID")
    score: float = Field(..., description="Recommendation score")
    reason: str = Field(..., description="Reason for recommendation")
    algorithm: str = Field(..., description="Algorithm used")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")

class RecommendationSet(BaseModel):
    """Set of recommendations for a user"""
    user_id: str = Field(..., description="User identifier")
    recommendations: List[Recommendation] = Field(..., description="List of recommendations")
    context: RecommendationContext = Field(..., description="Recommendation context")
    algorithm_version: str = Field("1.0", description="Algorithm version")
    generated_at: datetime = Field(default_factory=datetime.now)
    expires_at: Optional[datetime] = Field(None, description="When recommendations expire")

# Search Models
class SearchQuery(BaseModel):
    """Search query tracking model"""
    user_id: Optional[str] = Field(None, description="User identifier if logged in")
    query: str = Field(..., description="Search query text")
    category: Optional[str] = Field(None, description="Category filter")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Applied filters")
    results_count: int = Field(0, description="Number of results returned")
    clicked_products: List[str] = Field(default_factory=list, description="Products clicked from results")
    session_id: Optional[str] = Field(None, description="Session identifier")
    timestamp: datetime = Field(default_factory=datetime.now)

# Collection names constants
class Collections:
    PRODUCTS = "products"
    USERS = "users"
    ORDERS = "orders"
    USER_EVENTS = "user_events"
    STOCK_HISTORY = "stock_history"
    PRICE_HISTORY = "price_history"
    ALERTS = "alerts"
    SUPPLIERS = "suppliers"
    RECOMMENDATIONS = "recommendations"
    SEARCH_QUERIES = "search_queries"