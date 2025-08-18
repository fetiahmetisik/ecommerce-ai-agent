"""
Multi-Agent Orchestrator
Coordinates multiple agents using CrewAI for complex e-commerce tasks
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import asyncio

from crewai import Crew, Task, Process
from langchain_google_vertexai import VertexAI
import vertexai
from vertexai.generative_models import GenerativeModel

from agents.visual_search_agent import VisualSearchAgent
from agents.recommendation_agent import RecommendationAgent
from agents.inventory_agent import InventoryAgent
from config import settings

logging.basicConfig(level=getattr(logging, settings.log_level))
logger = logging.getLogger(__name__)

class ECommerceOrchestrator:
    """Orchestrates multiple AI agents for e-commerce operations"""
    
    def __init__(self):
        """Initialize the orchestrator with all agents"""
        # Initialize Vertex AI
        vertexai.init(project=settings.google_cloud_project, location=settings.location)
        self.model = GenerativeModel(settings.vertex_ai_model)
        
        # Initialize master LLM
        self.master_llm = VertexAI(
            model_name=settings.vertex_ai_model,
            temperature=0.5,
            max_output_tokens=settings.vertex_ai_max_tokens,
            project=settings.google_cloud_project,
            location=settings.location
        )
        
        # Initialize specialized agents
        logger.info("Initializing specialized agents...")
        self.visual_agent = VisualSearchAgent()
        self.recommendation_agent = RecommendationAgent()
        self.inventory_agent = InventoryAgent()
        
        # Agent registry
        self.agents = {
            "visual_search": self.visual_agent,
            "recommendations": self.recommendation_agent,
            "inventory": self.inventory_agent
        }
        
        logger.info("Orchestrator initialized successfully")
    
    async def process_customer_query(
        self,
        user_id: str,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        image_data: Optional[bytes] = None
    ) -> Dict[str, Any]:
        """
        Process complex customer queries using multiple agents
        
        Args:
            user_id: User identifier
            query: Customer query
            context: Additional context
            image_data: Optional image for visual search
            
        Returns:
            Comprehensive response from multiple agents
        """
        try:
            logger.info(f"Processing customer query: {query[:100]}...")
            
            # Analyze query intent
            intent = await self._analyze_intent(query, image_data is not None)
            
            # Create crew based on intent
            crew = self._create_crew_for_intent(intent)
            
            # Define tasks
            tasks = []
            
            # Visual search task if image provided
            if image_data:
                visual_task = Task(
                    description=f"""
                    Analyze the provided image and find similar products.
                    User query: {query}
                    Extract visual features and match with inventory.
                    """,
                    agent=self.visual_agent.agent,
                    expected_output="Visual search results with similar products"
                )
                tasks.append(visual_task)
            
            # Recommendation task
            if intent.get("needs_recommendations", True):
                rec_task = Task(
                    description=f"""
                    Generate personalized recommendations for user {user_id}.
                    Consider: {query}
                    Context: {json.dumps(context or {})}
                    Provide tailored suggestions based on user history and preferences.
                    """,
                    agent=self.recommendation_agent.agent,
                    expected_output="Personalized product recommendations"
                )
                tasks.append(rec_task)
            
            # Inventory check task
            if intent.get("check_availability", False):
                inv_task = Task(
                    description=f"""
                    Check inventory availability for requested products.
                    Query: {query}
                    Provide stock status and alternative options if out of stock.
                    """,
                    agent=self.inventory_agent.agent,
                    expected_output="Inventory status and availability"
                )
                tasks.append(inv_task)
            
            # Execute crew tasks
            if tasks:
                crew = Crew(
                    agents=[task.agent for task in tasks],
                    tasks=tasks,
                    process=Process.sequential if intent.get("sequential", False) 
                           else Process.hierarchical,
                    verbose=settings.debug
                )
                
                # Prepare inputs
                inputs = {
                    "user_id": user_id,
                    "query": query,
                    "context": context or {}
                }
                
                if image_data:
                    # Process image separately with visual agent
                    visual_results = await self.visual_agent.process_search_request(
                        image_data, query
                    )
                    inputs["visual_results"] = visual_results
                
                # Execute crew
                crew_output = crew.kickoff(inputs=inputs)
                
                # Generate unified response
                unified_response = await self._generate_unified_response(
                    query, intent, crew_output, inputs
                )
                
                return {
                    "success": True,
                    "query": query,
                    "intent": intent,
                    "response": unified_response,
                    "timestamp": datetime.now().isoformat()
                }
            
            return {
                "success": False,
                "error": "No applicable tasks for the query"
            }
            
        except Exception as e:
            logger.error(f"Error processing customer query: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _analyze_intent(
        self, 
        query: str, 
        has_image: bool = False
    ) -> Dict[str, Any]:
        """
        Analyze user query to determine intent and required agents
        
        Args:
            query: User query
            has_image: Whether an image was provided
            
        Returns:
            Intent analysis
        """
        try:
            prompt = f"""
            Analyze this e-commerce query and determine the intent:
            
            Query: {query}
            Has Image: {has_image}
            
            Determine:
            1. Primary intent (search, browse, purchase, inquiry, comparison)
            2. Product categories mentioned
            3. Specific products mentioned
            4. Price sensitivity indicators
            5. Urgency level
            6. Whether recommendations are needed
            7. Whether inventory check is needed
            8. Whether visual search should be used
            
            Format as JSON.
            """
            
            response = self.model.generate_content([prompt])
            
            try:
                intent = json.loads(response.text)
            except:
                # Fallback intent
                intent = {
                    "primary_intent": "search",
                    "needs_recommendations": True,
                    "check_availability": "stock" in query.lower() or "available" in query.lower(),
                    "use_visual_search": has_image,
                    "categories": [],
                    "urgency": "normal"
                }
            
            return intent
            
        except Exception as e:
            logger.error(f"Error analyzing intent: {str(e)}")
            return {
                "primary_intent": "unknown",
                "needs_recommendations": True,
                "check_availability": False
            }
    
    def _create_crew_for_intent(self, intent: Dict[str, Any]) -> Crew:
        """
        Create a CrewAI crew based on intent
        
        Args:
            intent: Intent analysis
            
        Returns:
            Configured crew
        """
        agents = []
        
        if intent.get("use_visual_search"):
            agents.append(self.visual_agent.agent)
        
        if intent.get("needs_recommendations"):
            agents.append(self.recommendation_agent.agent)
        
        if intent.get("check_availability"):
            agents.append(self.inventory_agent.agent)
        
        # Default to recommendation agent if no specific agents selected
        if not agents:
            agents.append(self.recommendation_agent.agent)
        
        return Crew(
            agents=agents,
            process=Process.sequential,
            verbose=settings.debug
        )
    
    async def _generate_unified_response(
        self,
        query: str,
        intent: Dict[str, Any],
        crew_output: Any,
        inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate a unified response from multiple agent outputs
        
        Args:
            query: Original query
            intent: Intent analysis
            crew_output: Output from crew execution
            inputs: Input data including any visual results
            
        Returns:
            Unified response
        """
        try:
            prompt = f"""
            Create a unified, coherent response from multiple agent outputs:
            
            Customer Query: {query}
            Intent: {json.dumps(intent)}
            Crew Output: {str(crew_output)[:2000]}
            
            Create a natural, helpful response that:
            1. Directly addresses the customer's query
            2. Integrates insights from all agents seamlessly
            3. Provides clear recommendations
            4. Mentions any important availability or pricing information
            5. Suggests next steps
            
            Format as a structured response with sections.
            """
            
            response = self.model.generate_content([prompt])
            
            # Structure the response
            unified = {
                "summary": response.text[:500],
                "details": response.text,
                "recommendations": inputs.get("visual_results", {}).get("recommendations", []),
                "next_steps": self._generate_next_steps(intent)
            }
            
            return unified
            
        except Exception as e:
            logger.error(f"Error generating unified response: {str(e)}")
            return {
                "summary": "I've processed your request.",
                "details": str(crew_output),
                "next_steps": ["View recommendations", "Check availability"]
            }
    
    def _generate_next_steps(self, intent: Dict[str, Any]) -> List[str]:
        """Generate suggested next steps based on intent"""
        steps = []
        
        if intent.get("primary_intent") == "search":
            steps.append("Refine your search")
            steps.append("View similar products")
        elif intent.get("primary_intent") == "purchase":
            steps.append("Add to cart")
            steps.append("Check availability")
            steps.append("View shipping options")
        
        if intent.get("needs_recommendations"):
            steps.append("Explore personalized recommendations")
        
        return steps
    
    async def handle_bulk_operation(
        self,
        operation_type: str,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Handle bulk operations across multiple agents
        
        Args:
            operation_type: Type of bulk operation
            data: Operation data
            
        Returns:
            Operation results
        """
        try:
            logger.info(f"Handling bulk operation: {operation_type}")
            
            if operation_type == "inventory_update":
                # Bulk inventory update
                results = []
                for product in data.get("products", []):
                    result = await self.inventory_agent.update_stock(
                        product["id"],
                        product["quantity"],
                        operation=product.get("operation", "set"),
                        reason=data.get("reason", "Bulk update")
                    )
                    results.append(result)
                
                return {
                    "success": True,
                    "operation": operation_type,
                    "processed": len(results),
                    "results": results
                }
            
            elif operation_type == "price_update":
                # Bulk price update
                results = []
                for product in data.get("products", []):
                    result = await self.inventory_agent.update_price(
                        product["id"],
                        product["new_price"],
                        reason=data.get("reason", "Bulk price update")
                    )
                    results.append(result)
                
                return {
                    "success": True,
                    "operation": operation_type,
                    "processed": len(results),
                    "results": results
                }
            
            elif operation_type == "generate_recommendations":
                # Bulk recommendation generation
                results = []
                for user_id in data.get("user_ids", []):
                    recs = await self.recommendation_agent.generate_recommendations(
                        user_id,
                        context=data.get("context"),
                        limit=data.get("limit", 10)
                    )
                    results.append(recs)
                
                return {
                    "success": True,
                    "operation": operation_type,
                    "processed": len(results),
                    "results": results
                }
            
            else:
                return {
                    "success": False,
                    "error": f"Unknown operation type: {operation_type}"
                }
            
        except Exception as e:
            logger.error(f"Error in bulk operation: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def monitor_system_health(self) -> Dict[str, Any]:
        """
        Monitor health of all agents and system components
        
        Returns:
            System health report
        """
        try:
            health_report = {
                "timestamp": datetime.now().isoformat(),
                "agents": {},
                "system": {}
            }
            
            # Check each agent
            for agent_name, agent in self.agents.items():
                try:
                    # Simple health check - try to access agent
                    if hasattr(agent, 'agent') and agent.agent:
                        health_report["agents"][agent_name] = {
                            "status": "healthy",
                            "memory_enabled": agent.agent.memory if hasattr(agent.agent, 'memory') else False
                        }
                    else:
                        health_report["agents"][agent_name] = {
                            "status": "unknown"
                        }
                except Exception as e:
                    health_report["agents"][agent_name] = {
                        "status": "error",
                        "error": str(e)
                    }
            
            # Check Vertex AI connection
            try:
                test_response = self.model.generate_content(["Test"])
                health_report["system"]["vertex_ai"] = "healthy"
            except Exception as e:
                health_report["system"]["vertex_ai"] = f"error: {str(e)}"
            
            # Overall status
            all_healthy = all(
                agent.get("status") == "healthy" 
                for agent in health_report["agents"].values()
            )
            health_report["overall_status"] = "healthy" if all_healthy else "degraded"
            
            return health_report
            
        except Exception as e:
            logger.error(f"Error monitoring system health: {str(e)}")
            return {
                "overall_status": "error",
                "error": str(e)
            }
    
    async def execute_complex_workflow(
        self,
        workflow_name: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute predefined complex workflows
        
        Args:
            workflow_name: Name of the workflow
            parameters: Workflow parameters
            
        Returns:
            Workflow execution results
        """
        try:
            logger.info(f"Executing workflow: {workflow_name}")
            
            if workflow_name == "new_product_launch":
                # Workflow for launching a new product
                product_id = parameters.get("product_id")
                
                # Step 1: Set initial inventory
                inv_result = await self.inventory_agent.update_stock(
                    product_id,
                    parameters.get("initial_stock", 100),
                    operation="set",
                    reason="New product launch"
                )
                
                # Step 2: Set pricing
                price_result = await self.inventory_agent.update_price(
                    product_id,
                    parameters.get("launch_price"),
                    reason="Launch pricing"
                )
                
                # Step 3: Generate initial recommendations
                rec_task = Task(
                    description=f"""
                    Create launch campaign recommendations for product {product_id}.
                    Target audience: {parameters.get('target_audience', 'general')}
                    Generate marketing angles and cross-sell opportunities.
                    """,
                    agent=self.recommendation_agent.agent,
                    expected_output="Launch campaign recommendations"
                )
                
                crew = Crew(
                    agents=[self.recommendation_agent.agent],
                    tasks=[rec_task],
                    process=Process.sequential
                )
                
                campaign = crew.kickoff(inputs=parameters)
                
                return {
                    "success": True,
                    "workflow": workflow_name,
                    "results": {
                        "inventory_setup": inv_result,
                        "pricing_setup": price_result,
                        "campaign_recommendations": str(campaign)
                    }
                }
            
            elif workflow_name == "seasonal_campaign":
                # Workflow for seasonal campaigns
                season = parameters.get("season", "summer")
                
                # Get seasonal recommendations for multiple users
                user_ids = parameters.get("user_ids", [])
                recommendations = []
                
                for user_id in user_ids[:10]:  # Limit to 10 users for demo
                    recs = await self.recommendation_agent.get_seasonal_recommendations(
                        user_id, season
                    )
                    recommendations.append({
                        "user_id": user_id,
                        "recommendations": recs
                    })
                
                # Check inventory for recommended products
                product_ids = set()
                for rec in recommendations:
                    for product in rec["recommendations"][:5]:
                        product_ids.add(product.get("product_id"))
                
                inventory_status = []
                for product_id in product_ids:
                    if product_id:
                        status = await self.inventory_agent.check_stock_level(product_id)
                        inventory_status.append(status)
                
                return {
                    "success": True,
                    "workflow": workflow_name,
                    "season": season,
                    "users_processed": len(recommendations),
                    "products_checked": len(inventory_status),
                    "results": {
                        "recommendations": recommendations[:5],  # Sample
                        "inventory_status": inventory_status[:10]  # Sample
                    }
                }
            
            else:
                return {
                    "success": False,
                    "error": f"Unknown workflow: {workflow_name}"
                }
            
        except Exception as e:
            logger.error(f"Error executing workflow: {str(e)}")
            return {
                "success": False,
                "workflow": workflow_name,
                "error": str(e)
            }