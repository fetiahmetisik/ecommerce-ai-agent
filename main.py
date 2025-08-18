import logging
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
import json
import base64

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from agents.orchestrator import ECommerceOrchestrator
from agents.visual_search_agent import VisualSearchAgent
from agents.recommendation_agent import RecommendationAgent
from agents.inventory_agent import InventoryAgent
from config import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="E-Commerce AI Agent API",
    description="Multi-agent AI system for e-commerce operations",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Global agent instances
orchestrator: Optional[ECommerceOrchestrator] = None
visual_agent: Optional[VisualSearchAgent] = None
recommendation_agent: Optional[RecommendationAgent] = None
inventory_agent: Optional[InventoryAgent] = None

# Pydantic Models
class CustomerQuery(BaseModel):
    user_id: str = Field(..., description="User identifier")
    query: str = Field(..., description="Customer query text")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")

class VisualSearchRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
    query: Optional[str] = Field(None, description="Optional text query")

class RecommendationRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")
    limit: int = Field(10, description="Maximum number of recommendations")

class InventoryUpdate(BaseModel):
    product_id: str = Field(..., description="Product identifier")
    quantity: int = Field(..., description="Quantity to update")
    operation: str = Field("set", description="Operation: set, add, or subtract")
    reason: str = Field("", description="Reason for update")

class PriceUpdate(BaseModel):
    product_id: str = Field(..., description="Product identifier")
    new_price: float = Field(..., description="New price")
    reason: str = Field("", description="Reason for price change")

class BulkOperation(BaseModel):
    operation_type: str = Field(..., description="Type of bulk operation")
    data: Dict[str, Any] = Field(..., description="Operation data")

class WorkflowRequest(BaseModel):
    workflow_name: str = Field(..., description="Name of the workflow")
    parameters: Dict[str, Any] = Field(..., description="Workflow parameters")

# Authentication dependency
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return credentials.credentials

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize agents on startup"""
    global orchestrator, visual_agent, recommendation_agent, inventory_agent
    
    try:
        logger.info("Initializing AI agents...")
        
        # Initialize all agents
        orchestrator = ECommerceOrchestrator()
        visual_agent = VisualSearchAgent()
        recommendation_agent = RecommendationAgent()
        inventory_agent = InventoryAgent()
        
        logger.info("All agents initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize agents: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down AI agents...")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        if orchestrator:
            health_report = await orchestrator.monitor_system_health()
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "agents": health_report
            }
        else:
            return {"status": "initializing"}
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {"status": "unhealthy", "error": str(e)}

# Customer query endpoint
@app.post("/api/v1/query")
async def process_customer_query(
    request: CustomerQuery,
    token: str = Depends(verify_token)
):
    """Process customer query using the orchestrator"""
    try:
        if not orchestrator:
            raise HTTPException(status_code=503, detail="Orchestrator not initialized")
        
        logger.info(f"Processing query for user {request.user_id}")
        
        result = await orchestrator.process_customer_query(
            user_id=request.user_id,
            query=request.query,
            context=request.context
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing customer query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Visual search endpoint
@app.post("/api/v1/visual-search")
async def visual_search(
    request: VisualSearchRequest,
    image: UploadFile = File(...),
    token: str = Depends(verify_token)
):
    """Process visual search request"""
    try:
        if not visual_agent:
            raise HTTPException(status_code=503, detail="Visual agent not initialized")
        
        # Validate image
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Read image data
        image_data = await image.read()
        
        logger.info(f"Processing visual search for user {request.user_id}")
        
        result = await visual_agent.process_search_request(
            image_data=image_data,
            user_query=request.query
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing visual search: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Combined query with image
@app.post("/api/v1/query-with-image")
async def process_query_with_image(
    user_id: str,
    query: str,
    image: UploadFile = File(...),
    context: Optional[str] = None,
    token: str = Depends(verify_token)
):
    """Process customer query with image using orchestrator"""
    try:
        if not orchestrator:
            raise HTTPException(status_code=503, detail="Orchestrator not initialized")
        
        # Validate image
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Read image data
        image_data = await image.read()
        
        # Parse context if provided
        parsed_context = None
        if context:
            try:
                parsed_context = json.loads(context)
            except:
                parsed_context = {"raw_context": context}
        
        logger.info(f"Processing query with image for user {user_id}")
        
        result = await orchestrator.process_customer_query(
            user_id=user_id,
            query=query,
            context=parsed_context,
            image_data=image_data
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing query with image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Recommendations endpoint
@app.post("/api/v1/recommendations")
async def get_recommendations(
    request: RecommendationRequest,
    token: str = Depends(verify_token)
):
    """Get personalized recommendations"""
    try:
        if not recommendation_agent:
            raise HTTPException(status_code=503, detail="Recommendation agent not initialized")
        
        logger.info(f"Generating recommendations for user {request.user_id}")
        
        result = await recommendation_agent.generate_recommendations(
            user_id=request.user_id,
            context=request.context,
            limit=request.limit
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Inventory endpoints
@app.get("/api/v1/inventory/{product_id}")
async def get_inventory_status(
    product_id: str,
    token: str = Depends(verify_token)
):
    """Get inventory status for a product"""
    try:
        if not inventory_agent:
            raise HTTPException(status_code=503, detail="Inventory agent not initialized")
        
        result = await inventory_agent.check_stock_level(product_id)
        
        if result.get("error"):
            raise HTTPException(status_code=404, detail=result["error"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting inventory status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/v1/inventory")
async def update_inventory(
    request: InventoryUpdate,
    token: str = Depends(verify_token)
):
    """Update inventory levels"""
    try:
        if not inventory_agent:
            raise HTTPException(status_code=503, detail="Inventory agent not initialized")
        
        result = await inventory_agent.update_stock(
            product_id=request.product_id,
            quantity=request.quantity,
            operation=request.operation,
            reason=request.reason
        )
        
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error", "Update failed"))
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating inventory: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/inventory/{product_id}/forecast")
async def get_demand_forecast(
    product_id: str,
    days_ahead: int = 7,
    token: str = Depends(verify_token)
):
    """Get demand forecast for a product"""
    try:
        if not inventory_agent:
            raise HTTPException(status_code=503, detail="Inventory agent not initialized")
        
        result = await inventory_agent.forecast_demand(
            product_id=product_id,
            days_ahead=days_ahead
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting demand forecast: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/inventory/{product_id}/reorder-point")
async def get_reorder_point(
    product_id: str,
    lead_time_days: int = 7,
    service_level: float = 0.95,
    token: str = Depends(verify_token)
):
    """Calculate reorder point for a product"""
    try:
        if not inventory_agent:
            raise HTTPException(status_code=503, detail="Inventory agent not initialized")
        
        result = await inventory_agent.calculate_reorder_point(
            product_id=product_id,
            lead_time_days=lead_time_days,
            service_level=service_level
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error calculating reorder point: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Price endpoints
@app.get("/api/v1/pricing/{product_id}")
async def get_pricing_info(
    product_id: str,
    token: str = Depends(verify_token)
):
    """Get pricing information for a product"""
    try:
        if not inventory_agent:
            raise HTTPException(status_code=503, detail="Inventory agent not initialized")
        
        result = await inventory_agent.check_price(product_id)
        
        if result.get("error"):
            raise HTTPException(status_code=404, detail=result["error"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting pricing info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/v1/pricing")
async def update_pricing(
    request: PriceUpdate,
    token: str = Depends(verify_token)
):
    """Update product pricing"""
    try:
        if not inventory_agent:
            raise HTTPException(status_code=503, detail="Inventory agent not initialized")
        
        result = await inventory_agent.update_price(
            product_id=request.product_id,
            new_price=request.new_price,
            reason=request.reason
        )
        
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error", "Update failed"))
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating pricing: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/pricing/{product_id}/history")
async def get_price_history(
    product_id: str,
    days: int = 30,
    token: str = Depends(verify_token)
):
    """Get price history for a product"""
    try:
        if not inventory_agent:
            raise HTTPException(status_code=503, detail="Inventory agent not initialized")
        
        result = await inventory_agent.get_price_history(
            product_id=product_id,
            days=days
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting price history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Bulk operations endpoint
@app.post("/api/v1/bulk")
async def handle_bulk_operation(
    request: BulkOperation,
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token)
):
    """Handle bulk operations"""
    try:
        if not orchestrator:
            raise HTTPException(status_code=503, detail="Orchestrator not initialized")
        
        # Process in background for large operations
        background_tasks.add_task(
            process_bulk_operation,
            request.operation_type,
            request.data
        )
        
        return {
            "success": True,
            "message": "Bulk operation started",
            "operation_type": request.operation_type
        }
        
    except Exception as e:
        logger.error(f"Error starting bulk operation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def process_bulk_operation(operation_type: str, data: Dict[str, Any]):
    """Process bulk operation in background"""
    try:
        result = await orchestrator.handle_bulk_operation(operation_type, data)
        logger.info(f"Bulk operation completed: {result}")
    except Exception as e:
        logger.error(f"Bulk operation failed: {str(e)}")

# Workflow endpoints
@app.post("/api/v1/workflows")
async def execute_workflow(
    request: WorkflowRequest,
    token: str = Depends(verify_token)
):
    """Execute predefined workflows"""
    try:
        if not orchestrator:
            raise HTTPException(status_code=503, detail="Orchestrator not initialized")
        
        logger.info(f"Executing workflow: {request.workflow_name}")
        
        result = await orchestrator.execute_complex_workflow(
            workflow_name=request.workflow_name,
            parameters=request.parameters
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error executing workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Monitoring endpoints
@app.get("/api/v1/monitoring/inventory")
async def monitor_inventory(
    category: Optional[str] = None,
    token: str = Depends(verify_token)
):
    """Monitor inventory status"""
    try:
        if not inventory_agent:
            raise HTTPException(status_code=503, detail="Inventory agent not initialized")
        
        result = await inventory_agent.monitor_inventory(category=category)
        
        return result
        
    except Exception as e:
        logger.error(f"Error monitoring inventory: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/monitoring/system")
async def monitor_system(
    token: str = Depends(verify_token)
):
    """Monitor overall system health"""
    try:
        if not orchestrator:
            raise HTTPException(status_code=503, detail="Orchestrator not initialized")
        
        result = await orchestrator.monitor_system_health()
        
        return result
        
    except Exception as e:
        logger.error(f"Error monitoring system: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# User profile endpoints
@app.get("/api/v1/users/{user_id}/profile")
async def get_user_profile(
    user_id: str,
    token: str = Depends(verify_token)
):
    """Get user profile and preferences"""
    try:
        if not recommendation_agent:
            raise HTTPException(status_code=503, detail="Recommendation agent not initialized")
        
        result = await recommendation_agent.get_user_profile(user_id)
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting user profile: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/users/{user_id}/history")
async def get_user_history(
    user_id: str,
    history_type: str = "purchase",
    limit: int = 50,
    token: str = Depends(verify_token)
):
    """Get user purchase or browsing history"""
    try:
        if not recommendation_agent:
            raise HTTPException(status_code=503, detail="Recommendation agent not initialized")
        
        if history_type == "purchase":
            result = await recommendation_agent.get_purchase_history(user_id, limit)
        elif history_type == "browsing":
            result = await recommendation_agent.get_browsing_history(user_id, limit)
        else:
            raise HTTPException(status_code=400, detail="Invalid history type")
        
        return {"user_id": user_id, "history_type": history_type, "data": result}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting user history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "timestamp": datetime.now().isoformat()}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "timestamp": datetime.now().isoformat()}
    )

# Main entry point
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )