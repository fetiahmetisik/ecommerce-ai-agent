"""
Inventory and Price Tracking Agent
Monitors stock levels, price changes, and manages inventory alerts
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
import asyncio

from google.cloud import firestore
from google.cloud import pubsub_v1
import vertexai
from vertexai.generative_models import GenerativeModel
from crewai import Agent, Task
from langchain.tools import Tool
from langchain_google_vertexai import VertexAI
import pandas as pd
import numpy as np

from config import settings

logging.basicConfig(level=getattr(logging, settings.log_level))
logger = logging.getLogger(__name__)

class InventoryAgent:
    """Agent for inventory management and price tracking"""
    
    def __init__(self):
        """Initialize Inventory Agent"""
        self.firestore_client = firestore.Client(project=settings.google_cloud_project)
        self.publisher = pubsub_v1.PublisherClient()
        
        # Initialize Vertex AI
        vertexai.init(project=settings.google_cloud_project, location=settings.location)
        self.model = GenerativeModel(settings.vertex_ai_model)
        
        # Initialize LLM for agent
        self.llm = VertexAI(
            model_name=settings.vertex_ai_model,
            temperature=0.3,  # Lower temperature for accuracy
            max_output_tokens=settings.vertex_ai_max_tokens,
            project=settings.google_cloud_project,
            location=settings.location
        )
        
        # Create the agent
        self.agent = self._create_agent()
        
        # Alert thresholds
        self.thresholds = {
            "low_stock": 10,
            "out_of_stock": 0,
            "price_drop": 0.1,  # 10% drop
            "price_increase": 0.2,  # 20% increase
            "demand_spike": 2.0  # 2x normal demand
        }
    
    def _create_agent(self) -> Agent:
        """Create and configure the inventory agent"""
        return Agent(
            role="Inventory Manager & Price Analyst",
            goal="Optimize inventory levels and track price changes to maximize availability and profitability",
            backstory="""You are an expert inventory manager with deep knowledge of 
            supply chain management, demand forecasting, and pricing strategies. 
            You can predict stock shortages, identify pricing opportunities, and 
            ensure optimal inventory levels. You excel at analyzing trends and 
            making data-driven recommendations.""",
            tools=self._create_tools(),
            llm=self.llm,
            verbose=settings.debug,
            max_iter=settings.agent_max_iterations,
            memory=settings.enable_memory
        )
    
    def _create_tools(self) -> List[Tool]:
        """Create tools for the inventory agent"""
        return [
            Tool(
                name="check_stock_level",
                func=self.check_stock_level,
                description="Check current stock level for a product"
            ),
            Tool(
                name="get_stock_history",
                func=self.get_stock_history,
                description="Get historical stock levels for trend analysis"
            ),
            Tool(
                name="update_stock",
                func=self.update_stock,
                description="Update stock levels for a product"
            ),
            Tool(
                name="check_price",
                func=self.check_price,
                description="Check current price of a product"
            ),
            Tool(
                name="get_price_history",
                func=self.get_price_history,
                description="Get price history for a product"
            ),
            Tool(
                name="update_price",
                func=self.update_price,
                description="Update product price"
            ),
            Tool(
                name="forecast_demand",
                func=self.forecast_demand,
                description="Forecast future demand for a product"
            ),
            Tool(
                name="calculate_reorder_point",
                func=self.calculate_reorder_point,
                description="Calculate optimal reorder point"
            ),
            Tool(
                name="get_supplier_info",
                func=self.get_supplier_info,
                description="Get supplier information for a product"
            ),
            Tool(
                name="create_alert",
                func=self.create_alert,
                description="Create inventory or price alert"
            )
        ]
    
    async def check_stock_level(self, product_id: str) -> Dict[str, Any]:
        """
        Check current stock level for a product
        
        Args:
            product_id: Product identifier
            
        Returns:
            Stock information
        """
        try:
            product_ref = self.firestore_client.collection(
                settings.firestore_products_collection
            ).document(product_id)
            
            product_doc = product_ref.get()
            
            if not product_doc.exists:
                return {"error": "Product not found"}
            
            product = product_doc.to_dict()
            stock_info = {
                "product_id": product_id,
                "name": product.get("name", "Unknown"),
                "current_stock": product.get("stock_quantity", 0),
                "reserved_stock": product.get("reserved_quantity", 0),
                "available_stock": product.get("stock_quantity", 0) - product.get("reserved_quantity", 0),
                "warehouse_locations": product.get("warehouse_locations", []),
                "status": self._determine_stock_status(product.get("stock_quantity", 0)),
                "last_updated": product.get("stock_updated_at", datetime.now().isoformat())
            }
            
            # Check for alerts
            if stock_info["available_stock"] <= self.thresholds["low_stock"]:
                stock_info["alert"] = "LOW_STOCK"
                if stock_info["available_stock"] == 0:
                    stock_info["alert"] = "OUT_OF_STOCK"
            
            return stock_info
            
        except Exception as e:
            logger.error(f"Error checking stock level: {str(e)}")
            return {"error": str(e)}
    
    def _determine_stock_status(self, quantity: int) -> str:
        """Determine stock status based on quantity"""
        if quantity == 0:
            return "out_of_stock"
        elif quantity <= self.thresholds["low_stock"]:
            return "low_stock"
        elif quantity > 100:
            return "high_stock"
        else:
            return "normal"
    
    async def get_stock_history(
        self, 
        product_id: str, 
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Get historical stock levels
        
        Args:
            product_id: Product identifier
            days: Number of days of history
            
        Returns:
            Stock history data
        """
        try:
            # Get from stock_history collection
            history_ref = self.firestore_client.collection("stock_history")
            
            cutoff_date = datetime.now() - timedelta(days=days)
            
            query = history_ref.where("product_id", "==", product_id)\
                             .where("timestamp", ">=", cutoff_date)\
                             .order_by("timestamp")
            
            history = []
            for doc in query.stream():
                record = doc.to_dict()
                history.append(record)
            
            # Analyze trends
            if history:
                df = pd.DataFrame(history)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                
                trend_analysis = {
                    "average_stock": df['quantity'].mean(),
                    "min_stock": df['quantity'].min(),
                    "max_stock": df['quantity'].max(),
                    "stock_volatility": df['quantity'].std(),
                    "stockout_days": len(df[df['quantity'] == 0]),
                    "trend": self._calculate_trend(df['quantity'].values)
                }
            else:
                trend_analysis = {}
            
            return {
                "history": history,
                "analysis": trend_analysis
            }
            
        except Exception as e:
            logger.error(f"Error getting stock history: {str(e)}")
            return {"history": [], "analysis": {}}
    
    def _calculate_trend(self, values: np.ndarray) -> str:
        """Calculate trend from time series data"""
        if len(values) < 2:
            return "insufficient_data"
        
        # Simple linear regression
        x = np.arange(len(values))
        z = np.polyfit(x, values, 1)
        slope = z[0]
        
        if slope > 0.5:
            return "increasing"
        elif slope < -0.5:
            return "decreasing"
        else:
            return "stable"
    
    async def update_stock(
        self, 
        product_id: str, 
        quantity: int, 
        operation: str = "set",
        reason: str = ""
    ) -> Dict[str, Any]:
        """
        Update stock levels
        
        Args:
            product_id: Product identifier
            quantity: Quantity to update
            operation: 'set', 'add', or 'subtract'
            reason: Reason for update
            
        Returns:
            Update result
        """
        try:
            product_ref = self.firestore_client.collection(
                settings.firestore_products_collection
            ).document(product_id)
            
            # Use transaction for consistency
            transaction = self.firestore_client.transaction()
            
            @firestore.transactional
            def update_in_transaction(transaction, product_ref):
                snapshot = product_ref.get(transaction=transaction)
                
                if not snapshot.exists:
                    raise ValueError("Product not found")
                
                current_stock = snapshot.get("stock_quantity") or 0
                
                if operation == "set":
                    new_stock = quantity
                elif operation == "add":
                    new_stock = current_stock + quantity
                elif operation == "subtract":
                    new_stock = max(0, current_stock - quantity)
                else:
                    raise ValueError(f"Invalid operation: {operation}")
                
                transaction.update(product_ref, {
                    "stock_quantity": new_stock,
                    "stock_updated_at": datetime.now(),
                    "stock_updated_by": "inventory_agent"
                })
                
                # Log the change
                history_ref = self.firestore_client.collection("stock_history").document()
                transaction.set(history_ref, {
                    "product_id": product_id,
                    "timestamp": datetime.now(),
                    "previous_quantity": current_stock,
                    "new_quantity": new_stock,
                    "change": new_stock - current_stock,
                    "operation": operation,
                    "reason": reason
                })
                
                return {
                    "previous_stock": current_stock,
                    "new_stock": new_stock,
                    "change": new_stock - current_stock
                }
            
            result = update_in_transaction(transaction, product_ref)
            
            # Check if alert needed
            if result["new_stock"] <= self.thresholds["low_stock"]:
                await self.create_alert(
                    product_id, 
                    "low_stock" if result["new_stock"] > 0 else "out_of_stock",
                    {"current_stock": result["new_stock"]}
                )
            
            return {
                "success": True,
                "product_id": product_id,
                **result
            }
            
        except Exception as e:
            logger.error(f"Error updating stock: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def check_price(self, product_id: str) -> Dict[str, Any]:
        """
        Check current price of a product
        
        Args:
            product_id: Product identifier
            
        Returns:
            Price information
        """
        try:
            product_ref = self.firestore_client.collection(
                settings.firestore_products_collection
            ).document(product_id)
            
            product_doc = product_ref.get()
            
            if not product_doc.exists:
                return {"error": "Product not found"}
            
            product = product_doc.to_dict()
            
            return {
                "product_id": product_id,
                "name": product.get("name", "Unknown"),
                "current_price": product.get("price", 0),
                "currency": product.get("currency", "USD"),
                "discount_price": product.get("discount_price"),
                "discount_percentage": product.get("discount_percentage", 0),
                "price_tier": product.get("price_tier", "standard"),
                "competitor_prices": product.get("competitor_prices", {}),
                "last_updated": product.get("price_updated_at", datetime.now().isoformat())
            }
            
        except Exception as e:
            logger.error(f"Error checking price: {str(e)}")
            return {"error": str(e)}
    
    async def get_price_history(
        self, 
        product_id: str, 
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Get price history for a product
        
        Args:
            product_id: Product identifier
            days: Number of days of history
            
        Returns:
            Price history and analysis
        """
        try:
            history_ref = self.firestore_client.collection("price_history")
            
            cutoff_date = datetime.now() - timedelta(days=days)
            
            query = history_ref.where("product_id", "==", product_id)\
                             .where("timestamp", ">=", cutoff_date)\
                             .order_by("timestamp")
            
            history = []
            for doc in query.stream():
                record = doc.to_dict()
                history.append(record)
            
            # Analyze price trends
            if history:
                df = pd.DataFrame(history)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                
                analysis = {
                    "average_price": df['price'].mean(),
                    "min_price": df['price'].min(),
                    "max_price": df['price'].max(),
                    "price_volatility": df['price'].std(),
                    "price_changes": len(df) - 1,
                    "total_change": df['price'].iloc[-1] - df['price'].iloc[0] if len(df) > 1 else 0,
                    "trend": self._calculate_trend(df['price'].values)
                }
                
                # Detect significant changes
                if len(df) > 1:
                    recent_change = (df['price'].iloc[-1] - df['price'].iloc[-2]) / df['price'].iloc[-2]
                    if abs(recent_change) > self.thresholds["price_drop"]:
                        analysis["alert"] = "significant_price_change"
                        analysis["change_percentage"] = recent_change * 100
            else:
                analysis = {}
            
            return {
                "history": history,
                "analysis": analysis
            }
            
        except Exception as e:
            logger.error(f"Error getting price history: {str(e)}")
            return {"history": [], "analysis": {}}
    
    async def update_price(
        self, 
        product_id: str, 
        new_price: float,
        reason: str = "",
        effective_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Update product price
        
        Args:
            product_id: Product identifier
            new_price: New price
            reason: Reason for price change
            effective_date: When the price change takes effect
            
        Returns:
            Update result
        """
        try:
            product_ref = self.firestore_client.collection(
                settings.firestore_products_collection
            ).document(product_id)
            
            # Use transaction
            transaction = self.firestore_client.transaction()
            
            @firestore.transactional
            def update_in_transaction(transaction, product_ref):
                snapshot = product_ref.get(transaction=transaction)
                
                if not snapshot.exists:
                    raise ValueError("Product not found")
                
                current_price = snapshot.get("price") or 0
                
                # Calculate price change percentage
                if current_price > 0:
                    change_percentage = (new_price - current_price) / current_price
                else:
                    change_percentage = 0
                
                # Update product
                update_data = {
                    "price": new_price,
                    "price_updated_at": datetime.now(),
                    "price_updated_by": "inventory_agent"
                }
                
                if effective_date and effective_date > datetime.now():
                    update_data["scheduled_price"] = new_price
                    update_data["scheduled_price_date"] = effective_date
                
                transaction.update(product_ref, update_data)
                
                # Log price change
                history_ref = self.firestore_client.collection("price_history").document()
                transaction.set(history_ref, {
                    "product_id": product_id,
                    "timestamp": datetime.now(),
                    "previous_price": current_price,
                    "new_price": new_price,
                    "change_amount": new_price - current_price,
                    "change_percentage": change_percentage,
                    "reason": reason,
                    "effective_date": effective_date or datetime.now()
                })
                
                return {
                    "previous_price": current_price,
                    "new_price": new_price,
                    "change_percentage": change_percentage * 100
                }
            
            result = update_in_transaction(transaction, product_ref)
            
            # Check for significant price changes
            if abs(result["change_percentage"]) > self.thresholds["price_drop"] * 100:
                await self.create_alert(
                    product_id,
                    "price_change",
                    result
                )
            
            return {
                "success": True,
                "product_id": product_id,
                **result
            }
            
        except Exception as e:
            logger.error(f"Error updating price: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def forecast_demand(
        self, 
        product_id: str, 
        days_ahead: int = 7
    ) -> Dict[str, Any]:
        """
        Forecast future demand for a product
        
        Args:
            product_id: Product identifier
            days_ahead: Number of days to forecast
            
        Returns:
            Demand forecast
        """
        try:
            # Get historical sales data
            orders_ref = self.firestore_client.collection(
                settings.firestore_orders_collection
            )
            
            cutoff_date = datetime.now() - timedelta(days=30)
            
            query = orders_ref.where("items", "array_contains", product_id)\
                            .where("created_at", ">=", cutoff_date)\
                            .order_by("created_at")
            
            sales_data = []
            for doc in query.stream():
                order = doc.to_dict()
                # Count quantity of this product in the order
                item_count = order.get("items", []).count(product_id)
                sales_data.append({
                    "date": order["created_at"],
                    "quantity": item_count
                })
            
            if not sales_data:
                return {
                    "forecast": [],
                    "confidence": "low",
                    "message": "Insufficient historical data"
                }
            
            # Group by day
            df = pd.DataFrame(sales_data)
            df['date'] = pd.to_datetime(df['date'])
            daily_sales = df.groupby(df['date'].dt.date)['quantity'].sum().reset_index()
            
            # Simple moving average forecast
            if len(daily_sales) >= 7:
                ma7 = daily_sales['quantity'].rolling(window=7).mean().iloc[-1]
            else:
                ma7 = daily_sales['quantity'].mean()
            
            # Generate forecast
            forecast = []
            for i in range(days_ahead):
                forecast_date = datetime.now().date() + timedelta(days=i+1)
                
                # Add some randomness for realistic forecast
                daily_forecast = max(0, int(ma7 + np.random.normal(0, ma7 * 0.2)))
                
                forecast.append({
                    "date": forecast_date.isoformat(),
                    "predicted_demand": daily_forecast,
                    "confidence_interval": {
                        "low": max(0, int(daily_forecast * 0.7)),
                        "high": int(daily_forecast * 1.3)
                    }
                })
            
            # Use Gemini for additional insights
            gemini_insights = await self._get_demand_insights(
                product_id, 
                daily_sales.to_dict('records'), 
                forecast
            )
            
            return {
                "product_id": product_id,
                "forecast": forecast,
                "total_predicted_demand": sum(f["predicted_demand"] for f in forecast),
                "average_daily_demand": ma7,
                "confidence": "high" if len(daily_sales) >= 14 else "medium",
                "insights": gemini_insights
            }
            
        except Exception as e:
            logger.error(f"Error forecasting demand: {str(e)}")
            return {"forecast": [], "error": str(e)}
    
    async def _get_demand_insights(
        self, 
        product_id: str, 
        historical_data: List[Dict], 
        forecast: List[Dict]
    ) -> Dict[str, Any]:
        """Get demand insights from Gemini"""
        try:
            prompt = f"""
            Analyze demand patterns for product {product_id}:
            
            Historical sales (last 30 days): {json.dumps(historical_data[-10:])}
            Forecast (next 7 days): {json.dumps(forecast)}
            
            Provide:
            1. Demand trend analysis
            2. Seasonality factors
            3. Risk factors that could affect demand
            4. Inventory recommendations
            5. Pricing strategy suggestions
            
            Format as JSON.
            """
            
            response = self.model.generate_content([prompt])
            
            try:
                insights = json.loads(response.text)
            except:
                insights = {"raw_text": response.text}
            
            return insights
            
        except Exception as e:
            logger.error(f"Error getting demand insights: {str(e)}")
            return {}
    
    async def calculate_reorder_point(
        self, 
        product_id: str,
        lead_time_days: int = 7,
        service_level: float = 0.95
    ) -> Dict[str, Any]:
        """
        Calculate optimal reorder point
        
        Args:
            product_id: Product identifier
            lead_time_days: Supplier lead time in days
            service_level: Desired service level (0-1)
            
        Returns:
            Reorder point calculation
        """
        try:
            # Get demand forecast
            forecast = await self.forecast_demand(product_id, days_ahead=lead_time_days)
            
            if not forecast.get("forecast"):
                return {"error": "Unable to calculate reorder point"}
            
            # Calculate average demand during lead time
            lead_time_demand = sum(
                f["predicted_demand"] for f in forecast["forecast"][:lead_time_days]
            )
            
            # Calculate safety stock (simplified)
            # Z-score for service level
            from scipy import stats
            z_score = stats.norm.ppf(service_level)
            
            # Standard deviation of demand (approximation)
            demand_std = forecast.get("average_daily_demand", 0) * 0.3
            safety_stock = z_score * demand_std * np.sqrt(lead_time_days)
            
            reorder_point = lead_time_demand + safety_stock
            
            # Get current stock
            stock_info = await self.check_stock_level(product_id)
            current_stock = stock_info.get("available_stock", 0)
            
            return {
                "product_id": product_id,
                "reorder_point": int(reorder_point),
                "current_stock": current_stock,
                "lead_time_demand": int(lead_time_demand),
                "safety_stock": int(safety_stock),
                "service_level": service_level * 100,
                "should_reorder": current_stock <= reorder_point,
                "days_until_stockout": int(current_stock / forecast.get("average_daily_demand", 1))
                    if forecast.get("average_daily_demand", 0) > 0 else 999
            }
            
        except Exception as e:
            logger.error(f"Error calculating reorder point: {str(e)}")
            return {"error": str(e)}
    
    async def get_supplier_info(self, product_id: str) -> Dict[str, Any]:
        """
        Get supplier information for a product
        
        Args:
            product_id: Product identifier
            
        Returns:
            Supplier information
        """
        try:
            # Get product
            product_ref = self.firestore_client.collection(
                settings.firestore_products_collection
            ).document(product_id)
            
            product_doc = product_ref.get()
            
            if not product_doc.exists:
                return {"error": "Product not found"}
            
            product = product_doc.to_dict()
            supplier_id = product.get("supplier_id")
            
            if not supplier_id:
                return {"error": "No supplier information available"}
            
            # Get supplier details
            supplier_ref = self.firestore_client.collection("suppliers").document(supplier_id)
            supplier_doc = supplier_ref.get()
            
            if supplier_doc.exists:
                supplier = supplier_doc.to_dict()
                return {
                    "supplier_id": supplier_id,
                    "name": supplier.get("name"),
                    "contact": supplier.get("contact"),
                    "lead_time_days": supplier.get("lead_time_days", 7),
                    "minimum_order_quantity": supplier.get("minimum_order_quantity", 1),
                    "reliability_score": supplier.get("reliability_score", 0.9),
                    "payment_terms": supplier.get("payment_terms", "Net 30"),
                    "last_order_date": supplier.get("last_order_date")
                }
            
            return {"error": "Supplier not found"}
            
        except Exception as e:
            logger.error(f"Error getting supplier info: {str(e)}")
            return {"error": str(e)}
    
    async def create_alert(
        self, 
        product_id: str, 
        alert_type: str, 
        details: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create inventory or price alert
        
        Args:
            product_id: Product identifier
            alert_type: Type of alert
            details: Alert details
            
        Returns:
            Alert creation result
        """
        try:
            alert_ref = self.firestore_client.collection("alerts").document()
            
            alert_data = {
                "product_id": product_id,
                "alert_type": alert_type,
                "details": details,
                "created_at": datetime.now(),
                "status": "active",
                "priority": self._determine_priority(alert_type, details)
            }
            
            alert_ref.set(alert_data)
            
            # Publish to Pub/Sub for real-time notifications
            if self.publisher:
                topic_path = self.publisher.topic_path(
                    settings.google_cloud_project, 
                    "inventory-alerts"
                )
                
                message = json.dumps(alert_data, default=str).encode()
                future = self.publisher.publish(topic_path, message)
                logger.info(f"Alert published: {future.result()}")
            
            return {
                "success": True,
                "alert_id": alert_ref.id,
                "alert_type": alert_type,
                "product_id": product_id
            }
            
        except Exception as e:
            logger.error(f"Error creating alert: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _determine_priority(self, alert_type: str, details: Dict[str, Any]) -> str:
        """Determine alert priority"""
        if alert_type == "out_of_stock":
            return "critical"
        elif alert_type == "low_stock":
            return "high"
        elif alert_type == "price_change":
            change = abs(details.get("change_percentage", 0))
            if change > 30:
                return "high"
            else:
                return "medium"
        else:
            return "low"
    
    async def monitor_inventory(
        self, 
        category: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Main method to monitor inventory status
        
        Args:
            category: Optional category filter
            
        Returns:
            Inventory monitoring report
        """
        try:
            logger.info("Starting inventory monitoring...")
            
            # Get all products or by category
            products_ref = self.firestore_client.collection(
                settings.firestore_products_collection
            )
            
            if category:
                query = products_ref.where("category", "==", category)
            else:
                query = products_ref
            
            alerts = []
            recommendations = []
            
            for doc in query.stream():
                product = doc.to_dict()
                product_id = doc.id
                
                # Check stock levels
                stock_info = await self.check_stock_level(product_id)
                if stock_info.get("alert"):
                    alerts.append({
                        "product_id": product_id,
                        "product_name": product.get("name"),
                        "alert_type": stock_info["alert"],
                        "current_stock": stock_info["available_stock"]
                    })
                
                # Calculate reorder point
                reorder_info = await self.calculate_reorder_point(product_id)
                if reorder_info.get("should_reorder"):
                    recommendations.append({
                        "product_id": product_id,
                        "product_name": product.get("name"),
                        "action": "reorder",
                        "reorder_quantity": reorder_info["reorder_point"] - stock_info["available_stock"],
                        "days_until_stockout": reorder_info["days_until_stockout"]
                    })
            
            # Create task for agent analysis
            task = Task(
                description=f"""
                Analyze inventory status and provide recommendations:
                
                Alerts: {json.dumps(alerts)}
                Reorder recommendations: {json.dumps(recommendations)}
                
                Provide:
                1. Priority actions needed
                2. Cost optimization opportunities
                3. Risk assessment
                4. Strategic recommendations
                """,
                agent=self.agent,
                expected_output="Detailed inventory analysis and recommendations"
            )
            
            # Execute task
            agent_analysis = await task.execute()
            
            return {
                "timestamp": datetime.now().isoformat(),
                "alerts": alerts,
                "recommendations": recommendations,
                "analysis": agent_analysis,
                "summary": {
                    "total_alerts": len(alerts),
                    "critical_alerts": len([a for a in alerts if a["alert_type"] == "OUT_OF_STOCK"]),
                    "products_to_reorder": len(recommendations)
                }
            }
            
        except Exception as e:
            logger.error(f"Error monitoring inventory: {str(e)}")
            raise