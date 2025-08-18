"""
Tests for main FastAPI application
"""

import pytest
import json
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient

def test_health_check(client: TestClient):
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "timestamp" in data

def test_health_check_with_orchestrator(client: TestClient):
    """Test health check with orchestrator initialized"""
    with patch('main.orchestrator') as mock_orchestrator:
        mock_health_report = {
            "overall_status": "healthy",
            "agents": {
                "visual_search": {"status": "healthy"},
                "recommendations": {"status": "healthy"},
                "inventory": {"status": "healthy"}
            }
        }
        mock_orchestrator.monitor_system_health.return_value = mock_health_report
        
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "agents" in data

def test_unauthorized_request(client: TestClient):
    """Test request without authentication"""
    response = client.post("/api/v1/query", json={
        "user_id": "test-user",
        "query": "test query"
    })
    assert response.status_code == 403  # Forbidden due to missing auth

def test_invalid_api_key(client: TestClient):
    """Test request with invalid API key"""
    headers = {"Authorization": "Bearer invalid-key"}
    response = client.post("/api/v1/query", json={
        "user_id": "test-user",
        "query": "test query"
    }, headers=headers)
    assert response.status_code == 401  # Unauthorized

def test_process_customer_query(client: TestClient, auth_headers: dict):
    """Test customer query processing"""
    with patch('main.orchestrator') as mock_orchestrator:
        mock_result = {
            "success": True,
            "query": "test query",
            "response": {"summary": "Test response"}
        }
        mock_orchestrator.process_customer_query.return_value = mock_result
        
        response = client.post("/api/v1/query", json={
            "user_id": "test-user",
            "query": "test query",
            "context": {"page": "home"}
        }, headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["query"] == "test query"

def test_process_customer_query_missing_orchestrator(client: TestClient, auth_headers: dict):
    """Test customer query when orchestrator is not initialized"""
    with patch('main.orchestrator', None):
        response = client.post("/api/v1/query", json={
            "user_id": "test-user",
            "query": "test query"
        }, headers=auth_headers)
        
        assert response.status_code == 503
        assert "Orchestrator not initialized" in response.json()["detail"]

def test_visual_search(client: TestClient, auth_headers: dict, sample_image_data: bytes):
    """Test visual search endpoint"""
    with patch('main.visual_agent') as mock_visual_agent:
        mock_result = {
            "image_analysis": {"labels": ["test_label"]},
            "similar_products": [],
            "recommendations": "Test recommendations"
        }
        mock_visual_agent.process_search_request.return_value = mock_result
        
        files = {"image": ("test.png", sample_image_data, "image/png")}
        data = {"user_id": "test-user", "query": "test query"}
        
        response = client.post("/api/v1/visual-search", 
                             files=files, 
                             data=data, 
                             headers=auth_headers)
        
        assert response.status_code == 200
        result = response.json()
        assert "image_analysis" in result
        assert "similar_products" in result

def test_visual_search_invalid_image(client: TestClient, auth_headers: dict):
    """Test visual search with invalid image format"""
    files = {"image": ("test.txt", b"not an image", "text/plain")}
    data = {"user_id": "test-user"}
    
    response = client.post("/api/v1/visual-search", 
                         files=files, 
                         data=data, 
                         headers=auth_headers)
    
    assert response.status_code == 400
    assert "Invalid image format" in response.json()["detail"]

def test_get_recommendations(client: TestClient, auth_headers: dict):
    """Test recommendations endpoint"""
    with patch('main.recommendation_agent') as mock_rec_agent:
        mock_result = {
            "user_id": "test-user",
            "recommendations": {"personalized": []},
            "user_segments": ["test_segment"]
        }
        mock_rec_agent.generate_recommendations.return_value = mock_result
        
        response = client.post("/api/v1/recommendations", json={
            "user_id": "test-user",
            "context": {"page": "home"},
            "limit": 10
        }, headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == "test-user"
        assert "recommendations" in data

def test_get_inventory_status(client: TestClient, auth_headers: dict):
    """Test inventory status endpoint"""
    with patch('main.inventory_agent') as mock_inv_agent:
        mock_result = {
            "product_id": "test-product",
            "current_stock": 100,
            "available_stock": 90,
            "status": "normal"
        }
        mock_inv_agent.check_stock_level.return_value = mock_result
        
        response = client.get("/api/v1/inventory/test-product", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["product_id"] == "test-product"
        assert data["current_stock"] == 100

def test_get_inventory_status_product_not_found(client: TestClient, auth_headers: dict):
    """Test inventory status for non-existent product"""
    with patch('main.inventory_agent') as mock_inv_agent:
        mock_inv_agent.check_stock_level.return_value = {"error": "Product not found"}
        
        response = client.get("/api/v1/inventory/non-existent", headers=auth_headers)
        
        assert response.status_code == 404

def test_update_inventory(client: TestClient, auth_headers: dict):
    """Test inventory update endpoint"""
    with patch('main.inventory_agent') as mock_inv_agent:
        mock_result = {
            "success": True,
            "product_id": "test-product",
            "previous_stock": 100,
            "new_stock": 150
        }
        mock_inv_agent.update_stock.return_value = mock_result
        
        response = client.put("/api/v1/inventory", json={
            "product_id": "test-product",
            "quantity": 50,
            "operation": "add",
            "reason": "Restock"
        }, headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["new_stock"] == 150

def test_get_pricing_info(client: TestClient, auth_headers: dict):
    """Test pricing info endpoint"""
    with patch('main.inventory_agent') as mock_inv_agent:
        mock_result = {
            "product_id": "test-product",
            "current_price": 99.99,
            "currency": "TRY"
        }
        mock_inv_agent.check_price.return_value = mock_result
        
        response = client.get("/api/v1/pricing/test-product", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["current_price"] == 99.99

def test_update_pricing(client: TestClient, auth_headers: dict):
    """Test pricing update endpoint"""
    with patch('main.inventory_agent') as mock_inv_agent:
        mock_result = {
            "success": True,
            "product_id": "test-product",
            "previous_price": 99.99,
            "new_price": 89.99
        }
        mock_inv_agent.update_price.return_value = mock_result
        
        response = client.put("/api/v1/pricing", json={
            "product_id": "test-product",
            "new_price": 89.99,
            "reason": "Sale promotion"
        }, headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["new_price"] == 89.99

def test_bulk_operation(client: TestClient, auth_headers: dict):
    """Test bulk operation endpoint"""
    with patch('main.orchestrator') as mock_orchestrator:
        response = client.post("/api/v1/bulk", json={
            "operation_type": "inventory_update",
            "data": {
                "products": [
                    {"id": "prod1", "quantity": 100},
                    {"id": "prod2", "quantity": 50}
                ]
            }
        }, headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "Bulk operation started" in data["message"]

def test_execute_workflow(client: TestClient, auth_headers: dict):
    """Test workflow execution endpoint"""
    with patch('main.orchestrator') as mock_orchestrator:
        mock_result = {
            "success": True,
            "workflow": "new_product_launch",
            "results": {"inventory_setup": {"success": True}}
        }
        mock_orchestrator.execute_complex_workflow.return_value = mock_result
        
        response = client.post("/api/v1/workflows", json={
            "workflow_name": "new_product_launch",
            "parameters": {
                "product_id": "new-product",
                "initial_stock": 100
            }
        }, headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["workflow"] == "new_product_launch"

def test_get_user_profile(client: TestClient, auth_headers: dict):
    """Test user profile endpoint"""
    with patch('main.recommendation_agent') as mock_rec_agent:
        mock_profile = {
            "user_id": "test-user",
            "email": "test@example.com",
            "preferences": {"favorite_categories": ["Electronics"]}
        }
        mock_rec_agent.get_user_profile.return_value = mock_profile
        
        response = client.get("/api/v1/users/test-user/profile", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == "test-user"
        assert "preferences" in data

def test_get_user_history(client: TestClient, auth_headers: dict):
    """Test user history endpoint"""
    with patch('main.recommendation_agent') as mock_rec_agent:
        mock_history = [
            {"product_id": "prod1", "quantity": 1},
            {"product_id": "prod2", "quantity": 2}
        ]
        mock_rec_agent.get_purchase_history.return_value = mock_history
        
        response = client.get("/api/v1/users/test-user/history?history_type=purchase", 
                            headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == "test-user"
        assert data["history_type"] == "purchase"
        assert len(data["data"]) == 2

def test_monitor_inventory(client: TestClient, auth_headers: dict):
    """Test inventory monitoring endpoint"""
    with patch('main.inventory_agent') as mock_inv_agent:
        mock_report = {
            "timestamp": "2024-01-01T00:00:00",
            "alerts": [],
            "recommendations": [],
            "summary": {"total_alerts": 0}
        }
        mock_inv_agent.monitor_inventory.return_value = mock_report
        
        response = client.get("/api/v1/monitoring/inventory", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "alerts" in data
        assert "recommendations" in data

def test_monitor_system(client: TestClient, auth_headers: dict):
    """Test system monitoring endpoint"""
    with patch('main.orchestrator') as mock_orchestrator:
        mock_health = {
            "overall_status": "healthy",
            "agents": {"visual_search": {"status": "healthy"}}
        }
        mock_orchestrator.monitor_system_health.return_value = mock_health
        
        response = client.get("/api/v1/monitoring/system", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["overall_status"] == "healthy"
        assert "agents" in data

def test_error_handling(client: TestClient, auth_headers: dict):
    """Test general error handling"""
    with patch('main.orchestrator') as mock_orchestrator:
        mock_orchestrator.process_customer_query.side_effect = Exception("Test error")
        
        response = client.post("/api/v1/query", json={
            "user_id": "test-user",
            "query": "test query"
        }, headers=auth_headers)
        
        assert response.status_code == 500
        data = response.json()
        assert "error" in data
        assert "timestamp" in data

def test_query_with_image(client: TestClient, auth_headers: dict, sample_image_data: bytes):
    """Test combined query with image endpoint"""
    with patch('main.orchestrator') as mock_orchestrator:
        mock_result = {
            "success": True,
            "query": "test query",
            "response": {"summary": "Test response with image"}
        }
        mock_orchestrator.process_customer_query.return_value = mock_result
        
        files = {"image": ("test.png", sample_image_data, "image/png")}
        data = {
            "user_id": "test-user",
            "query": "test query",
            "context": '{"page": "search"}'
        }
        
        response = client.post("/api/v1/query-with-image", 
                             files=files, 
                             data=data, 
                             headers=auth_headers)
        
        assert response.status_code == 200
        result = response.json()
        assert result["success"] is True
        assert result["query"] == "test query"