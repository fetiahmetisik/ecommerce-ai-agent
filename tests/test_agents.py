"""
Tests for AI agents
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from agents.visual_search_agent import VisualSearchAgent
from agents.recommendation_agent import RecommendationAgent
from agents.inventory_agent import InventoryAgent
from agents.orchestrator import ECommerceOrchestrator

class TestVisualSearchAgent:
    """Tests for Visual Search Agent"""
    
    @pytest.fixture
    def visual_agent(self, mock_firestore_client, mock_vision_client, mock_storage_client, mock_vertex_ai):
        """Create visual search agent with mocked dependencies"""
        with patch('agents.visual_search_agent.vision.ImageAnnotatorClient', return_value=mock_vision_client), \
             patch('agents.visual_search_agent.storage.Client', return_value=mock_storage_client), \
             patch('agents.visual_search_agent.firestore.Client', return_value=mock_firestore_client), \
             patch('agents.visual_search_agent.vertexai.init'), \
             patch('agents.visual_search_agent.GenerativeModel', return_value=mock_vertex_ai):
            return VisualSearchAgent()
    
    @pytest.mark.asyncio
    async def test_analyze_image(self, visual_agent, mock_vision_response, sample_image_data):
        """Test image analysis functionality"""
        visual_agent.vision_client.annotate_image.return_value = mock_vision_response
        
        result = await visual_agent.analyze_image(sample_image_data)
        
        assert "labels" in result
        assert "objects" in result
        assert "colors" in result
        assert "confidence" in result
        assert result["labels"] == ["test_label"]
    
    @pytest.mark.asyncio
    async def test_find_similar_products(self, visual_agent, sample_product_data):
        """Test finding similar products"""
        # Mock Firestore query
        mock_query = Mock()
        mock_doc = Mock()
        mock_doc.to_dict.return_value = sample_product_data
        mock_doc.id = "test-product-id"
        mock_query.stream.return_value = [mock_doc]
        
        visual_agent.firestore_client.collection.return_value.where.return_value.limit.return_value = mock_query
        
        image_features = {
            "labels": ["Electronics"],
            "colors": [{"rgb": {"r": 255, "g": 0, "b": 0}}]
        }
        
        result = await visual_agent.find_similar_products(image_features)
        
        assert len(result) > 0
        assert result[0]["id"] == "test-product-id"
        assert "similarity_score" in result[0]
    
    @pytest.mark.asyncio
    async def test_process_search_request(self, visual_agent, sample_image_data):
        """Test complete search request processing"""
        # Mock analyze_image
        mock_features = {
            "labels": ["test_label"],
            "objects": ["test_object"],
            "colors": [],
            "confidence": 0.9
        }
        visual_agent.analyze_image = AsyncMock(return_value=mock_features)
        
        # Mock find_similar_products
        visual_agent.find_similar_products = AsyncMock(return_value=[])
        
        # Mock task execution
        with patch('agents.visual_search_agent.Task') as mock_task_class:
            mock_task = AsyncMock()
            mock_task.execute.return_value = "Test recommendations"
            mock_task_class.return_value = mock_task
            
            result = await visual_agent.process_search_request(sample_image_data, "test query")
            
            assert "image_analysis" in result
            assert "similar_products" in result
            assert "recommendations" in result
            assert result["image_analysis"] == mock_features

class TestRecommendationAgent:
    """Tests for Recommendation Agent"""
    
    @pytest.fixture
    def recommendation_agent(self, mock_firestore_client, mock_vertex_ai):
        """Create recommendation agent with mocked dependencies"""
        with patch('agents.recommendation_agent.firestore.Client', return_value=mock_firestore_client), \
             patch('agents.recommendation_agent.vertexai.init'), \
             patch('agents.recommendation_agent.GenerativeModel', return_value=mock_vertex_ai), \
             patch('agents.recommendation_agent.discoveryengine.SearchServiceClient'):
            return RecommendationAgent()
    
    @pytest.mark.asyncio
    async def test_get_user_profile(self, recommendation_agent, sample_user_data):
        """Test getting user profile"""
        # Mock Firestore document
        mock_doc = Mock()
        mock_doc.exists = True
        mock_doc.to_dict.return_value = sample_user_data
        recommendation_agent.firestore_client.collection.return_value.document.return_value.get.return_value = mock_doc
        
        # Mock compute preferences
        recommendation_agent._compute_preferences = AsyncMock(return_value={})
        
        result = await recommendation_agent.get_user_profile("test-user")
        
        assert result["email"] == sample_user_data["email"]
        assert "computed_preferences" in result
        assert "segments" in result
    
    @pytest.mark.asyncio
    async def test_get_purchase_history(self, recommendation_agent, sample_order_data):
        """Test getting purchase history"""
        # Mock Firestore query
        mock_query = Mock()
        mock_doc = Mock()
        mock_doc.to_dict.return_value = sample_order_data
        mock_doc.id = "order-id"
        mock_query.stream.return_value = [mock_doc]
        
        recommendation_agent.firestore_client.collection.return_value.where.return_value.order_by.return_value.limit.return_value = mock_query
        
        result = await recommendation_agent.get_purchase_history("test-user")
        
        assert len(result) > 0
        assert result[0]["order_id"] == "order-id"
    
    @pytest.mark.asyncio
    async def test_generate_recommendations(self, recommendation_agent, sample_user_data):
        """Test recommendation generation"""
        # Mock all required methods
        recommendation_agent.get_user_profile = AsyncMock(return_value=sample_user_data)
        recommendation_agent.get_purchase_history = AsyncMock(return_value=[])
        recommendation_agent.get_browsing_history = AsyncMock(return_value=[])
        recommendation_agent.get_trending_products = AsyncMock(return_value=[])
        recommendation_agent.get_seasonal_recommendations = AsyncMock(return_value=[])
        recommendation_agent.find_similar_users = AsyncMock(return_value=[])
        
        # Mock task execution
        with patch('agents.recommendation_agent.Task') as mock_task_class:
            mock_task = AsyncMock()
            mock_task.execute.return_value = "Personalized recommendations"
            mock_task_class.return_value = mock_task
            
            result = await recommendation_agent.generate_recommendations("test-user")
            
            assert result["user_id"] == "test-user"
            assert "recommendations" in result
            assert "user_segments" in result

class TestInventoryAgent:
    """Tests for Inventory Agent"""
    
    @pytest.fixture
    def inventory_agent(self, mock_firestore_client, mock_vertex_ai):
        """Create inventory agent with mocked dependencies"""
        with patch('agents.inventory_agent.firestore.Client', return_value=mock_firestore_client), \
             patch('agents.inventory_agent.pubsub_v1.PublisherClient'), \
             patch('agents.inventory_agent.vertexai.init'), \
             patch('agents.inventory_agent.GenerativeModel', return_value=mock_vertex_ai):
            return InventoryAgent()
    
    @pytest.mark.asyncio
    async def test_check_stock_level(self, inventory_agent, sample_product_data):
        """Test checking stock levels"""
        # Mock Firestore document
        mock_doc = Mock()
        mock_doc.exists = True
        mock_doc.to_dict.return_value = sample_product_data
        inventory_agent.firestore_client.collection.return_value.document.return_value.get.return_value = mock_doc
        
        result = await inventory_agent.check_stock_level("test-product")
        
        assert result["product_id"] == "test-product"
        assert "current_stock" in result
        assert "available_stock" in result
        assert "status" in result
    
    @pytest.mark.asyncio
    async def test_update_stock(self, inventory_agent):
        """Test stock update functionality"""
        # Mock transaction
        mock_transaction = Mock()
        mock_snapshot = Mock()
        mock_snapshot.exists = True
        mock_snapshot.get.return_value = 100  # current stock
        
        with patch.object(inventory_agent.firestore_client, 'transaction', return_value=mock_transaction):
            # Mock the transactional function
            def mock_update_function(transaction, product_ref):
                return {
                    "previous_stock": 100,
                    "new_stock": 150,
                    "change": 50
                }
            
            with patch('agents.inventory_agent.firestore.transactional') as mock_transactional:
                mock_transactional.side_effect = lambda func: func
                
                # Mock product reference
                mock_product_ref = Mock()
                inventory_agent.firestore_client.collection.return_value.document.return_value = mock_product_ref
                mock_product_ref.get.return_value = mock_snapshot
                
                inventory_agent.create_alert = AsyncMock()
                
                result = await inventory_agent.update_stock("test-product", 50, "add", "Restock")
                
                # The actual update logic will be mocked, so we just verify structure
                assert "success" in result
                assert "product_id" in result
    
    @pytest.mark.asyncio
    async def test_forecast_demand(self, inventory_agent, sample_order_data):
        """Test demand forecasting"""
        # Mock Firestore query for orders
        mock_query = Mock()
        mock_doc = Mock()
        mock_doc.to_dict.return_value = sample_order_data
        mock_query.stream.return_value = [mock_doc]
        
        inventory_agent.firestore_client.collection.return_value.where.return_value.order_by.return_value = mock_query
        
        # Mock Gemini insights
        inventory_agent._get_demand_insights = AsyncMock(return_value={"trend": "increasing"})
        
        result = await inventory_agent.forecast_demand("test-product")
        
        assert "forecast" in result
        assert "product_id" in result
        assert result["product_id"] == "test-product"

class TestOrchestrator:
    """Tests for E-Commerce Orchestrator"""
    
    @pytest.fixture
    def orchestrator(self, mock_vertex_ai):
        """Create orchestrator with mocked dependencies"""
        with patch('agents.orchestrator.vertexai.init'), \
             patch('agents.orchestrator.GenerativeModel', return_value=mock_vertex_ai), \
             patch('agents.orchestrator.VisualSearchAgent'), \
             patch('agents.orchestrator.RecommendationAgent'), \
             patch('agents.orchestrator.InventoryAgent'):
            return ECommerceOrchestrator()
    
    @pytest.mark.asyncio
    async def test_analyze_intent(self, orchestrator):
        """Test intent analysis"""
        query = "I want to buy a red dress for summer"
        
        result = await orchestrator._analyze_intent(query, has_image=False)
        
        assert "primary_intent" in result
        assert "needs_recommendations" in result
        assert "check_availability" in result
    
    @pytest.mark.asyncio
    async def test_process_customer_query(self, orchestrator):
        """Test customer query processing"""
        # Mock intent analysis
        orchestrator._analyze_intent = AsyncMock(return_value={
            "primary_intent": "search",
            "needs_recommendations": True,
            "check_availability": False
        })
        
        # Mock crew creation and execution
        with patch('agents.orchestrator.Crew') as mock_crew_class, \
             patch('agents.orchestrator.Task') as mock_task_class:
            
            mock_crew = Mock()
            mock_crew.kickoff.return_value = "Test crew result"
            mock_crew_class.return_value = mock_crew
            
            mock_task = Mock()
            mock_task_class.return_value = mock_task
            
            # Mock unified response generation
            orchestrator._generate_unified_response = AsyncMock(return_value={
                "summary": "Test response",
                "details": "Detailed response"
            })
            
            result = await orchestrator.process_customer_query(
                user_id="test-user",
                query="test query",
                context={"page": "home"}
            )
            
            assert result["success"] is True
            assert result["query"] == "test query"
            assert "intent" in result
            assert "response" in result
    
    @pytest.mark.asyncio
    async def test_monitor_system_health(self, orchestrator):
        """Test system health monitoring"""
        # Mock agent health checks
        for agent_name, agent in orchestrator.agents.items():
            agent.agent = Mock()
        
        result = await orchestrator.monitor_system_health()
        
        assert "timestamp" in result
        assert "agents" in result
        assert "system" in result
        assert "overall_status" in result
    
    @pytest.mark.asyncio
    async def test_handle_bulk_operation(self, orchestrator):
        """Test bulk operation handling"""
        # Mock inventory agent methods
        orchestrator.inventory_agent.update_stock = AsyncMock(return_value={"success": True})
        
        operation_data = {
            "products": [
                {"id": "prod1", "quantity": 100, "operation": "set"},
                {"id": "prod2", "quantity": 50, "operation": "add"}
            ],
            "reason": "Bulk update test"
        }
        
        result = await orchestrator.handle_bulk_operation("inventory_update", operation_data)
        
        assert result["success"] is True
        assert result["operation"] == "inventory_update"
        assert "processed" in result
        assert "results" in result