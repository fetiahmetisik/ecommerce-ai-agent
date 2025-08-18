"""
Personalized Recommendation Agent
Provides AI-powered product recommendations based on user behavior and preferences
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import json

from google.cloud import firestore
from google.cloud import discoveryengine_v1beta as discoveryengine
import vertexai
from vertexai.generative_models import GenerativeModel
from crewai import Agent, Task
from langchain.tools import Tool
from langchain_google_vertexai import VertexAI
from langchain.memory import ConversationBufferMemory
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from config import settings

logging.basicConfig(level=getattr(logging, settings.log_level))
logger = logging.getLogger(__name__)

class RecommendationAgent:
    """Agent for personalized product recommendations"""
    
    def __init__(self):
        """Initialize Recommendation Agent"""
        self.firestore_client = firestore.Client(project=settings.google_cloud_project)
        
        # Initialize Vertex AI
        vertexai.init(project=settings.google_cloud_project, location=settings.location)
        self.model = GenerativeModel(settings.vertex_ai_model)
        
        # Initialize Retail API client for recommendations
        self.retail_client = discoveryengine.SearchServiceClient()
        
        # Initialize LLM for agent
        self.llm = VertexAI(
            model_name=settings.vertex_ai_model,
            temperature=0.8,  # Higher temperature for creative recommendations
            max_output_tokens=settings.vertex_ai_max_tokens,
            project=settings.google_cloud_project,
            location=settings.location
        )
        
        # Memory for conversation context
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Create the agent
        self.agent = self._create_agent()
    
    def _create_agent(self) -> Agent:
        """Create and configure the recommendation agent"""
        return Agent(
            role="Personal Shopping Assistant & Recommendation Expert",
            goal="Provide highly personalized product recommendations that delight customers",
            backstory="""You are an expert personal shopper with deep knowledge of 
            fashion trends, customer psychology, and product matching. You understand 
            individual preferences, seasonal trends, and can predict what customers 
            will love based on their history and behavior. You excel at cross-selling 
            and upselling by finding complementary products.""",
            tools=self._create_tools(),
            llm=self.llm,
            verbose=settings.debug,
            max_iter=settings.agent_max_iterations,
            memory=settings.enable_memory
        )
    
    def _create_tools(self) -> List[Tool]:
        """Create tools for the recommendation agent"""
        return [
            Tool(
                name="get_user_profile",
                func=self.get_user_profile,
                description="Retrieve user profile and preferences"
            ),
            Tool(
                name="get_purchase_history",
                func=self.get_purchase_history,
                description="Get user's purchase history"
            ),
            Tool(
                name="get_browsing_history",
                func=self.get_browsing_history,
                description="Get user's recent browsing history"
            ),
            Tool(
                name="find_similar_users",
                func=self.find_similar_users,
                description="Find users with similar preferences"
            ),
            Tool(
                name="get_trending_products",
                func=self.get_trending_products,
                description="Get currently trending products"
            ),
            Tool(
                name="get_complementary_products",
                func=self.get_complementary_products,
                description="Find products that complement a given product"
            ),
            Tool(
                name="calculate_affinity_score",
                func=self.calculate_affinity_score,
                description="Calculate user-product affinity score"
            ),
            Tool(
                name="get_seasonal_recommendations",
                func=self.get_seasonal_recommendations,
                description="Get season-appropriate product recommendations"
            )
        ]
    
    async def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """
        Retrieve user profile and preferences from Firestore
        
        Args:
            user_id: User identifier
            
        Returns:
            User profile data
        """
        try:
            user_ref = self.firestore_client.collection(
                settings.firestore_users_collection
            ).document(user_id)
            
            user_doc = user_ref.get()
            
            if user_doc.exists:
                profile = user_doc.to_dict()
                
                # Enrich with computed preferences
                profile["computed_preferences"] = await self._compute_preferences(user_id)
                profile["segments"] = self._determine_user_segments(profile)
                
                return profile
            else:
                # Return default profile for new users
                return {
                    "user_id": user_id,
                    "preferences": {},
                    "segments": ["new_user"],
                    "created_at": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error getting user profile: {str(e)}")
            return {}
    
    async def _compute_preferences(self, user_id: str) -> Dict[str, Any]:
        """
        Compute user preferences based on behavior
        
        Args:
            user_id: User identifier
            
        Returns:
            Computed preferences
        """
        try:
            # Get user's interaction history
            history = await self.get_purchase_history(user_id)
            browsing = await self.get_browsing_history(user_id)
            
            preferences = {
                "favorite_categories": [],
                "price_range": {"min": 0, "max": 0},
                "favorite_brands": [],
                "style_preferences": [],
                "color_preferences": []
            }
            
            if history:
                # Analyze purchase patterns
                categories = {}
                brands = {}
                prices = []
                
                for item in history:
                    # Count categories
                    cat = item.get("category", "unknown")
                    categories[cat] = categories.get(cat, 0) + 1
                    
                    # Count brands
                    brand = item.get("brand", "unknown")
                    brands[brand] = brands.get(brand, 0) + 1
                    
                    # Collect prices
                    if item.get("price"):
                        prices.append(item["price"])
                
                # Determine favorites
                if categories:
                    sorted_cats = sorted(categories.items(), key=lambda x: x[1], reverse=True)
                    preferences["favorite_categories"] = [cat for cat, _ in sorted_cats[:3]]
                
                if brands:
                    sorted_brands = sorted(brands.items(), key=lambda x: x[1], reverse=True)
                    preferences["favorite_brands"] = [brand for brand, _ in sorted_brands[:3]]
                
                if prices:
                    preferences["price_range"] = {
                        "min": min(prices),
                        "max": max(prices),
                        "average": sum(prices) / len(prices)
                    }
            
            # Use Gemini to analyze patterns
            if history or browsing:
                gemini_analysis = await self._analyze_with_gemini(history, browsing)
                preferences["ai_insights"] = gemini_analysis
            
            return preferences
            
        except Exception as e:
            logger.error(f"Error computing preferences: {str(e)}")
            return {}
    
    async def _analyze_with_gemini(
        self, 
        purchase_history: List[Dict], 
        browsing_history: List[Dict]
    ) -> Dict[str, Any]:
        """Use Gemini to analyze user behavior patterns"""
        try:
            prompt = f"""
            Analyze this user's shopping behavior and provide insights:
            
            Purchase History: {json.dumps(purchase_history[:10])}
            Browsing History: {json.dumps(browsing_history[:10])}
            
            Provide:
            1. Shopping personality type (e.g., bargain hunter, brand loyalist, trendsetter)
            2. Key motivations for purchases
            3. Predicted interests for next purchase
            4. Recommended product categories to explore
            5. Price sensitivity level (low/medium/high)
            
            Format as JSON.
            """
            
            response = self.model.generate_content([prompt])
            
            try:
                insights = json.loads(response.text)
            except:
                insights = {"raw_text": response.text}
            
            return insights
            
        except Exception as e:
            logger.error(f"Gemini analysis error: {str(e)}")
            return {}
    
    def _determine_user_segments(self, profile: Dict[str, Any]) -> List[str]:
        """
        Determine user segments for targeting
        
        Args:
            profile: User profile data
            
        Returns:
            List of user segments
        """
        segments = []
        
        # Age-based segments
        age = profile.get("age")
        if age:
            if age < 25:
                segments.append("gen_z")
            elif age < 40:
                segments.append("millennial")
            elif age < 55:
                segments.append("gen_x")
            else:
                segments.append("boomer")
        
        # Spending-based segments
        if profile.get("computed_preferences", {}).get("price_range", {}).get("average", 0) > 500:
            segments.append("high_spender")
        elif profile.get("computed_preferences", {}).get("price_range", {}).get("average", 0) < 50:
            segments.append("budget_conscious")
        
        # Behavior-based segments
        if profile.get("total_purchases", 0) > 10:
            segments.append("loyal_customer")
        elif profile.get("total_purchases", 0) == 0:
            segments.append("new_customer")
        
        return segments
    
    async def get_purchase_history(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get user's purchase history
        
        Args:
            user_id: User identifier
            limit: Maximum number of purchases to retrieve
            
        Returns:
            List of purchase records
        """
        try:
            orders_ref = self.firestore_client.collection(
                settings.firestore_orders_collection
            )
            
            query = orders_ref.where("user_id", "==", user_id)\
                            .order_by("created_at", direction=firestore.Query.DESCENDING)\
                            .limit(limit)
            
            purchases = []
            for doc in query.stream():
                purchase = doc.to_dict()
                purchase["order_id"] = doc.id
                purchases.append(purchase)
            
            return purchases
            
        except Exception as e:
            logger.error(f"Error getting purchase history: {str(e)}")
            return []
    
    async def get_browsing_history(self, user_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get user's recent browsing history
        
        Args:
            user_id: User identifier
            limit: Maximum number of events to retrieve
            
        Returns:
            List of browsing events
        """
        try:
            # Get from events collection
            events_ref = self.firestore_client.collection("user_events")
            
            # Get events from last 30 days
            cutoff_date = datetime.now() - timedelta(days=30)
            
            query = events_ref.where("user_id", "==", user_id)\
                            .where("event_type", "in", ["view", "click", "add_to_cart"])\
                            .where("timestamp", ">=", cutoff_date)\
                            .order_by("timestamp", direction=firestore.Query.DESCENDING)\
                            .limit(limit)
            
            events = []
            for doc in query.stream():
                event = doc.to_dict()
                events.append(event)
            
            return events
            
        except Exception as e:
            logger.error(f"Error getting browsing history: {str(e)}")
            return []
    
    async def find_similar_users(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Find users with similar preferences using collaborative filtering
        
        Args:
            user_id: User identifier
            limit: Maximum number of similar users
            
        Returns:
            List of similar users with similarity scores
        """
        try:
            # Get current user profile
            user_profile = await self.get_user_profile(user_id)
            
            if not user_profile:
                return []
            
            # Get all users (in production, use more efficient methods)
            users_ref = self.firestore_client.collection(
                settings.firestore_users_collection
            )
            
            similar_users = []
            user_vector = self._profile_to_vector(user_profile)
            
            for doc in users_ref.stream():
                if doc.id == user_id:
                    continue
                
                other_profile = doc.to_dict()
                other_vector = self._profile_to_vector(other_profile)
                
                # Calculate cosine similarity
                similarity = cosine_similarity([user_vector], [other_vector])[0][0]
                
                if similarity > 0.5:  # Threshold for similarity
                    similar_users.append({
                        "user_id": doc.id,
                        "similarity_score": float(similarity),
                        "common_interests": self._find_common_interests(
                            user_profile, other_profile
                        )
                    })
            
            # Sort by similarity
            similar_users.sort(key=lambda x: x["similarity_score"], reverse=True)
            
            return similar_users[:limit]
            
        except Exception as e:
            logger.error(f"Error finding similar users: {str(e)}")
            return []
    
    def _profile_to_vector(self, profile: Dict[str, Any]) -> np.ndarray:
        """Convert user profile to numerical vector for similarity calculation"""
        vector = []
        
        # Encode categorical preferences
        categories = ["electronics", "fashion", "home", "sports", "books", "toys"]
        fav_cats = profile.get("computed_preferences", {}).get("favorite_categories", [])
        for cat in categories:
            vector.append(1.0 if cat in fav_cats else 0.0)
        
        # Encode price range (normalized)
        price_avg = profile.get("computed_preferences", {}).get("price_range", {}).get("average", 100)
        vector.append(min(price_avg / 1000, 1.0))  # Normalize to 0-1
        
        # Encode activity level
        total_purchases = profile.get("total_purchases", 0)
        vector.append(min(total_purchases / 100, 1.0))  # Normalize to 0-1
        
        return np.array(vector)
    
    def _find_common_interests(
        self, 
        profile1: Dict[str, Any], 
        profile2: Dict[str, Any]
    ) -> List[str]:
        """Find common interests between two user profiles"""
        interests = []
        
        # Common categories
        cats1 = set(profile1.get("computed_preferences", {}).get("favorite_categories", []))
        cats2 = set(profile2.get("computed_preferences", {}).get("favorite_categories", []))
        common_cats = cats1 & cats2
        if common_cats:
            interests.extend(list(common_cats))
        
        # Common brands
        brands1 = set(profile1.get("computed_preferences", {}).get("favorite_brands", []))
        brands2 = set(profile2.get("computed_preferences", {}).get("favorite_brands", []))
        common_brands = brands1 & brands2
        if common_brands:
            interests.extend(list(common_brands))
        
        return interests
    
    async def get_trending_products(
        self, 
        category: Optional[str] = None, 
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get currently trending products
        
        Args:
            category: Optional category filter
            limit: Maximum number of products
            
        Returns:
            List of trending products
        """
        try:
            products_ref = self.firestore_client.collection(
                settings.firestore_products_collection
            )
            
            # Base query
            query = products_ref
            
            if category:
                query = query.where("category", "==", category)
            
            # Order by trending score (views, purchases, etc.)
            query = query.where("trending_score", ">", 0)\
                        .order_by("trending_score", direction=firestore.Query.DESCENDING)\
                        .limit(limit)
            
            trending = []
            for doc in query.stream():
                product = doc.to_dict()
                product["product_id"] = doc.id
                trending.append(product)
            
            return trending
            
        except Exception as e:
            logger.error(f"Error getting trending products: {str(e)}")
            # Fallback to popular products
            return await self._get_popular_products(category, limit)
    
    async def _get_popular_products(
        self, 
        category: Optional[str] = None, 
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Fallback method to get popular products"""
        try:
            products_ref = self.firestore_client.collection(
                settings.firestore_products_collection
            )
            
            query = products_ref
            if category:
                query = query.where("category", "==", category)
            
            query = query.order_by("sales_count", direction=firestore.Query.DESCENDING)\
                        .limit(limit)
            
            products = []
            for doc in query.stream():
                product = doc.to_dict()
                product["product_id"] = doc.id
                products.append(product)
            
            return products
            
        except Exception as e:
            logger.error(f"Error getting popular products: {str(e)}")
            return []
    
    async def get_complementary_products(
        self, 
        product_id: str, 
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find products that complement a given product
        
        Args:
            product_id: Product identifier
            limit: Maximum number of complementary products
            
        Returns:
            List of complementary products
        """
        try:
            # Get the main product
            product_ref = self.firestore_client.collection(
                settings.firestore_products_collection
            ).document(product_id)
            
            product_doc = product_ref.get()
            if not product_doc.exists:
                return []
            
            product = product_doc.to_dict()
            
            # Find frequently bought together
            orders_ref = self.firestore_client.collection(
                settings.firestore_orders_collection
            )
            
            # Find orders containing this product
            orders_with_product = orders_ref.where(
                "items", "array_contains", product_id
            ).limit(100)
            
            # Count co-occurrences
            co_occurrences = {}
            for doc in orders_with_product.stream():
                order = doc.to_dict()
                for item_id in order.get("items", []):
                    if item_id != product_id:
                        co_occurrences[item_id] = co_occurrences.get(item_id, 0) + 1
            
            # Sort by frequency
            sorted_items = sorted(
                co_occurrences.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:limit]
            
            # Get product details
            complementary = []
            for item_id, count in sorted_items:
                item_ref = self.firestore_client.collection(
                    settings.firestore_products_collection
                ).document(item_id)
                
                item_doc = item_ref.get()
                if item_doc.exists:
                    item_data = item_doc.to_dict()
                    item_data["product_id"] = item_id
                    item_data["co_purchase_count"] = count
                    complementary.append(item_data)
            
            # If not enough co-purchases, use category-based recommendations
            if len(complementary) < limit:
                category_recs = await self._get_category_complements(
                    product, 
                    limit - len(complementary)
                )
                complementary.extend(category_recs)
            
            return complementary
            
        except Exception as e:
            logger.error(f"Error getting complementary products: {str(e)}")
            return []
    
    async def _get_category_complements(
        self, 
        product: Dict[str, Any], 
        limit: int
    ) -> List[Dict[str, Any]]:
        """Get complementary products based on category rules"""
        complements_map = {
            "clothing": ["accessories", "shoes", "bags"],
            "electronics": ["accessories", "cables", "cases"],
            "furniture": ["decor", "lighting", "rugs"],
            "books": ["stationery", "reading_accessories"],
        }
        
        category = product.get("category", "").lower()
        complement_categories = complements_map.get(category, [])
        
        if not complement_categories:
            return []
        
        products = []
        for comp_cat in complement_categories:
            cat_products = await self.get_trending_products(comp_cat, limit=2)
            products.extend(cat_products)
        
        return products[:limit]
    
    async def calculate_affinity_score(
        self, 
        user_id: str, 
        product_id: str
    ) -> float:
        """
        Calculate affinity score between user and product
        
        Args:
            user_id: User identifier
            product_id: Product identifier
            
        Returns:
            Affinity score (0-1)
        """
        try:
            # Get user profile and product
            user_profile = await self.get_user_profile(user_id)
            
            product_ref = self.firestore_client.collection(
                settings.firestore_products_collection
            ).document(product_id)
            product_doc = product_ref.get()
            
            if not product_doc.exists:
                return 0.0
            
            product = product_doc.to_dict()
            
            score = 0.0
            weights = {
                "category_match": 0.3,
                "brand_match": 0.2,
                "price_match": 0.2,
                "style_match": 0.15,
                "trending": 0.15
            }
            
            # Category match
            user_cats = user_profile.get("computed_preferences", {}).get("favorite_categories", [])
            if product.get("category") in user_cats:
                score += weights["category_match"]
            
            # Brand match
            user_brands = user_profile.get("computed_preferences", {}).get("favorite_brands", [])
            if product.get("brand") in user_brands:
                score += weights["brand_match"]
            
            # Price match
            price_range = user_profile.get("computed_preferences", {}).get("price_range", {})
            if price_range:
                product_price = product.get("price", 0)
                if price_range.get("min", 0) <= product_price <= price_range.get("max", float('inf')):
                    score += weights["price_match"]
            
            # Style match (simplified)
            user_style = user_profile.get("computed_preferences", {}).get("style_preferences", [])
            product_tags = product.get("tags", [])
            if any(style in product_tags for style in user_style):
                score += weights["style_match"]
            
            # Trending bonus
            if product.get("trending_score", 0) > 50:
                score += weights["trending"]
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating affinity score: {str(e)}")
            return 0.0
    
    async def get_seasonal_recommendations(
        self, 
        user_id: str, 
        season: Optional[str] = None, 
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get season-appropriate product recommendations
        
        Args:
            user_id: User identifier
            season: Season (auto-detected if not provided)
            limit: Maximum number of recommendations
            
        Returns:
            List of seasonal product recommendations
        """
        try:
            # Auto-detect season if not provided
            if not season:
                month = datetime.now().month
                if month in [12, 1, 2]:
                    season = "winter"
                elif month in [3, 4, 5]:
                    season = "spring"
                elif month in [6, 7, 8]:
                    season = "summer"
                else:
                    season = "fall"
            
            # Get user profile for personalization
            user_profile = await self.get_user_profile(user_id)
            
            # Query seasonal products
            products_ref = self.firestore_client.collection(
                settings.firestore_products_collection
            )
            
            query = products_ref.where("seasons", "array_contains", season)\
                              .limit(limit * 2)  # Get extra for filtering
            
            seasonal_products = []
            for doc in query.stream():
                product = doc.to_dict()
                product["product_id"] = doc.id
                
                # Calculate personalized score
                affinity = await self.calculate_affinity_score(user_id, doc.id)
                product["personalization_score"] = affinity
                
                seasonal_products.append(product)
            
            # Sort by personalization score
            seasonal_products.sort(
                key=lambda x: x["personalization_score"], 
                reverse=True
            )
            
            return seasonal_products[:limit]
            
        except Exception as e:
            logger.error(f"Error getting seasonal recommendations: {str(e)}")
            return []
    
    async def generate_recommendations(
        self,
        user_id: str,
        context: Optional[Dict[str, Any]] = None,
        limit: int = 10
    ) -> Dict[str, Any]:
        """
        Main method to generate personalized recommendations
        
        Args:
            user_id: User identifier
            context: Additional context (current page, search query, etc.)
            limit: Maximum number of recommendations
            
        Returns:
            Personalized recommendations with explanations
        """
        try:
            logger.info(f"Generating recommendations for user {user_id}")
            
            # Get user data
            user_profile = await self.get_user_profile(user_id)
            purchase_history = await self.get_purchase_history(user_id, limit=20)
            browsing_history = await self.get_browsing_history(user_id, limit=50)
            
            # Get various recommendation types
            trending = await self.get_trending_products(limit=5)
            seasonal = await self.get_seasonal_recommendations(user_id, limit=5)
            
            # Get collaborative filtering recommendations
            similar_users = await self.find_similar_users(user_id, limit=5)
            collab_recs = []
            for similar_user in similar_users[:3]:
                their_purchases = await self.get_purchase_history(
                    similar_user["user_id"], 
                    limit=5
                )
                collab_recs.extend(their_purchases)
            
            # Create task for the agent
            task = Task(
                description=f"""
                Generate personalized product recommendations for user.
                
                User Profile: {json.dumps(user_profile)}
                Recent Purchases: {len(purchase_history)} items
                Browsing History: {len(browsing_history)} events
                Context: {json.dumps(context or {})}
                
                Available recommendations:
                - Trending: {len(trending)} products
                - Seasonal: {len(seasonal)} products
                - From similar users: {len(collab_recs)} products
                
                Provide:
                1. Top {limit} personalized recommendations
                2. Explanation for each recommendation
                3. Cross-sell/upsell opportunities
                4. Personalized message for the user
                """,
                agent=self.agent,
                expected_output="Detailed personalized recommendations with explanations"
            )
            
            # Execute task
            agent_result = await task.execute()
            
            return {
                "user_id": user_id,
                "recommendations": {
                    "personalized": agent_result,
                    "trending": trending[:3],
                    "seasonal": seasonal[:3],
                    "based_on_history": collab_recs[:3]
                },
                "user_segments": user_profile.get("segments", []),
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            raise