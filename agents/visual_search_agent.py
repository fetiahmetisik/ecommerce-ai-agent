"""
Visual Product Search Agent
Handles image-based product searches using Google Vision API and Vertex AI
"""

import base64
from typing import List, Dict, Any, Optional
from io import BytesIO
import logging

from google.cloud import vision
from google.cloud import storage
from google.cloud import firestore
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from PIL import Image
import numpy as np

from crewai import Agent, Task
from langchain.tools import Tool
from langchain_google_vertexai import VertexAI

from config import settings

logging.basicConfig(level=getattr(logging, settings.log_level))
logger = logging.getLogger(__name__)

class VisualSearchAgent:
    """Agent for visual product search and analysis"""
    
    def __init__(self):
        """Initialize Visual Search Agent with necessary clients"""
        self.vision_client = vision.ImageAnnotatorClient()
        self.storage_client = storage.Client(project=settings.google_cloud_project)
        self.firestore_client = firestore.Client(project=settings.google_cloud_project)
        self.bucket = self.storage_client.bucket(settings.gcs_bucket_name)
        
        # Initialize Vertex AI
        vertexai.init(project=settings.google_cloud_project, location=settings.location)
        self.model = GenerativeModel(settings.vertex_ai_model)
        
        # Initialize LLM for agent
        self.llm = VertexAI(
            model_name=settings.vertex_ai_model,
            temperature=settings.vertex_ai_temperature,
            max_output_tokens=settings.vertex_ai_max_tokens,
            project=settings.google_cloud_project,
            location=settings.location
        )
        
        # Create the agent
        self.agent = self._create_agent()
        
    def _create_agent(self) -> Agent:
        """Create and configure the visual search agent"""
        return Agent(
            role="Visual Product Search Specialist",
            goal="Find and recommend products based on visual similarity and features",
            backstory="""You are an expert in visual recognition and product matching. 
            You can analyze images to identify products, extract features, and find 
            similar items in the inventory. You understand fashion, electronics, 
            home goods, and various product categories.""",
            tools=self._create_tools(),
            llm=self.llm,
            verbose=settings.debug,
            max_iter=settings.agent_max_iterations,
            memory=settings.enable_memory
        )
    
    def _create_tools(self) -> List[Tool]:
        """Create tools for the visual search agent"""
        return [
            Tool(
                name="analyze_image",
                func=self.analyze_image,
                description="Analyze an image to extract product features and attributes"
            ),
            Tool(
                name="find_similar_products",
                func=self.find_similar_products,
                description="Find similar products based on visual features"
            ),
            Tool(
                name="detect_objects",
                func=self.detect_objects,
                description="Detect and identify objects in an image"
            ),
            Tool(
                name="extract_text",
                func=self.extract_text_from_image,
                description="Extract text from product images (brands, labels, etc.)"
            ),
            Tool(
                name="compare_images",
                func=self.compare_images,
                description="Compare two product images for similarity"
            )
        ]
    
    async def analyze_image(self, image_data: bytes) -> Dict[str, Any]:
        """
        Analyze product image using Vision API and Gemini
        
        Args:
            image_data: Image bytes
            
        Returns:
            Analysis results including features, colors, labels
        """
        try:
            # Vision API analysis
            image = vision.Image(content=image_data)
            
            # Perform multiple detections
            features = [
                vision.Feature(type_=vision.Feature.Type.LABEL_DETECTION, max_results=10),
                vision.Feature(type_=vision.Feature.Type.OBJECT_LOCALIZATION, max_results=10),
                vision.Feature(type_=vision.Feature.Type.IMAGE_PROPERTIES),
                vision.Feature(type_=vision.Feature.Type.WEB_DETECTION, max_results=5),
                vision.Feature(type_=vision.Feature.Type.TEXT_DETECTION)
            ]
            
            request = vision.AnnotateImageRequest(image=image, features=features)
            response = self.vision_client.annotate_image(request=request)
            
            # Extract results
            labels = [label.description for label in response.label_annotations]
            objects = [obj.name for obj in response.localized_object_annotations]
            
            # Extract dominant colors
            colors = []
            if response.image_properties_annotation.dominant_colors:
                for color in response.image_properties_annotation.dominant_colors.colors[:5]:
                    colors.append({
                        "rgb": {
                            "r": color.color.red,
                            "g": color.color.green,
                            "b": color.color.blue
                        },
                        "score": color.score,
                        "pixel_fraction": color.pixel_fraction
                    })
            
            # Web entities for better product understanding
            web_entities = []
            if response.web_detection:
                web_entities = [entity.description for entity in response.web_detection.web_entities[:5]]
            
            # Use Gemini for detailed analysis
            gemini_analysis = await self._analyze_with_gemini(image_data)
            
            return {
                "labels": labels,
                "objects": objects,
                "colors": colors,
                "web_entities": web_entities,
                "gemini_analysis": gemini_analysis,
                "confidence": self._calculate_confidence(response)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing image: {str(e)}")
            raise
    
    async def _analyze_with_gemini(self, image_data: bytes) -> Dict[str, Any]:
        """
        Use Gemini for detailed product analysis
        
        Args:
            image_data: Image bytes
            
        Returns:
            Detailed analysis from Gemini
        """
        try:
            # Convert image to base64 for Gemini
            image_part = Part.from_data(data=image_data, mime_type="image/jpeg")
            
            prompt = """
            Analyze this product image and provide:
            1. Product category and type
            2. Brand (if visible)
            3. Key features and attributes
            4. Target demographic
            5. Price range estimate (budget/mid-range/premium)
            6. Style/design characteristics
            7. Recommended use cases
            
            Format as JSON.
            """
            
            response = self.model.generate_content([prompt, image_part])
            
            # Parse JSON response
            import json
            try:
                analysis = json.loads(response.text)
            except:
                analysis = {"raw_text": response.text}
            
            return analysis
            
        except Exception as e:
            logger.error(f"Gemini analysis error: {str(e)}")
            return {"error": str(e)}
    
    async def find_similar_products(
        self, 
        image_features: Dict[str, Any], 
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Find similar products in Firestore based on image features
        
        Args:
            image_features: Extracted image features
            limit: Maximum number of results
            
        Returns:
            List of similar products
        """
        try:
            # Query Firestore for products
            products_ref = self.firestore_client.collection(
                settings.firestore_products_collection
            )
            
            # Build query based on features
            similar_products = []
            
            # Search by labels/categories
            if image_features.get("labels"):
                for label in image_features["labels"][:3]:
                    query = products_ref.where("category", "==", label.lower()).limit(limit)
                    docs = query.stream()
                    
                    for doc in docs:
                        product = doc.to_dict()
                        product["id"] = doc.id
                        product["similarity_score"] = self._calculate_similarity(
                            image_features, product
                        )
                        similar_products.append(product)
            
            # Search by colors
            if image_features.get("colors"):
                dominant_color = image_features["colors"][0]["rgb"]
                # Color-based search logic here
            
            # Sort by similarity score
            similar_products.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
            
            return similar_products[:limit]
            
        except Exception as e:
            logger.error(f"Error finding similar products: {str(e)}")
            return []
    
    def detect_objects(self, image_data: bytes) -> List[Dict[str, Any]]:
        """
        Detect objects in an image with bounding boxes
        
        Args:
            image_data: Image bytes
            
        Returns:
            List of detected objects with locations
        """
        try:
            image = vision.Image(content=image_data)
            response = self.vision_client.object_localization(image=image)
            
            objects = []
            for obj in response.localized_object_annotations:
                vertices = []
                for vertex in obj.bounding_poly.normalized_vertices:
                    vertices.append({"x": vertex.x, "y": vertex.y})
                
                objects.append({
                    "name": obj.name,
                    "confidence": obj.score,
                    "bounding_box": vertices
                })
            
            return objects
            
        except Exception as e:
            logger.error(f"Error detecting objects: {str(e)}")
            return []
    
    def extract_text_from_image(self, image_data: bytes) -> Dict[str, Any]:
        """
        Extract text from product images
        
        Args:
            image_data: Image bytes
            
        Returns:
            Extracted text and metadata
        """
        try:
            image = vision.Image(content=image_data)
            response = self.vision_client.text_detection(image=image)
            
            texts = response.text_annotations
            if texts:
                return {
                    "full_text": texts[0].description,
                    "words": [text.description for text in texts[1:]],
                    "language": texts[0].locale if hasattr(texts[0], 'locale') else "unknown"
                }
            
            return {"full_text": "", "words": [], "language": "unknown"}
            
        except Exception as e:
            logger.error(f"Error extracting text: {str(e)}")
            return {"full_text": "", "words": [], "language": "unknown"}
    
    async def compare_images(
        self, 
        image1_data: bytes, 
        image2_data: bytes
    ) -> Dict[str, Any]:
        """
        Compare two product images for similarity
        
        Args:
            image1_data: First image bytes
            image2_data: Second image bytes
            
        Returns:
            Similarity analysis
        """
        try:
            # Analyze both images
            features1 = await self.analyze_image(image1_data)
            features2 = await self.analyze_image(image2_data)
            
            # Calculate similarity metrics
            label_similarity = self._calculate_label_similarity(
                features1.get("labels", []), 
                features2.get("labels", [])
            )
            
            color_similarity = self._calculate_color_similarity(
                features1.get("colors", []), 
                features2.get("colors", [])
            )
            
            # Use Gemini for detailed comparison
            comparison = await self._compare_with_gemini(image1_data, image2_data)
            
            return {
                "label_similarity": label_similarity,
                "color_similarity": color_similarity,
                "overall_similarity": (label_similarity + color_similarity) / 2,
                "gemini_comparison": comparison,
                "is_similar": (label_similarity + color_similarity) / 2 > 0.7
            }
            
        except Exception as e:
            logger.error(f"Error comparing images: {str(e)}")
            raise
    
    async def _compare_with_gemini(
        self, 
        image1_data: bytes, 
        image2_data: bytes
    ) -> Dict[str, Any]:
        """Use Gemini to compare two product images"""
        try:
            image1_part = Part.from_data(data=image1_data, mime_type="image/jpeg")
            image2_part = Part.from_data(data=image2_data, mime_type="image/jpeg")
            
            prompt = """
            Compare these two product images and provide:
            1. Are they the same product? (yes/no)
            2. Similarity percentage (0-100)
            3. Common features
            4. Key differences
            5. Category match
            
            Format as JSON.
            """
            
            response = self.model.generate_content([prompt, image1_part, image2_part])
            
            import json
            try:
                comparison = json.loads(response.text)
            except:
                comparison = {"raw_text": response.text}
            
            return comparison
            
        except Exception as e:
            logger.error(f"Gemini comparison error: {str(e)}")
            return {"error": str(e)}
    
    def _calculate_confidence(self, vision_response) -> float:
        """Calculate overall confidence score from Vision API response"""
        scores = []
        
        if vision_response.label_annotations:
            scores.extend([label.score for label in vision_response.label_annotations[:5]])
        
        if vision_response.localized_object_annotations:
            scores.extend([obj.score for obj in vision_response.localized_object_annotations[:5]])
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def _calculate_similarity(
        self, 
        image_features: Dict[str, Any], 
        product: Dict[str, Any]
    ) -> float:
        """Calculate similarity score between image features and product"""
        score = 0.0
        
        # Category/label matching
        if image_features.get("labels") and product.get("category"):
            for label in image_features["labels"]:
                if label.lower() in product["category"].lower():
                    score += 0.3
                    break
        
        # Brand matching
        if image_features.get("gemini_analysis", {}).get("brand"):
            if product.get("brand", "").lower() == image_features["gemini_analysis"]["brand"].lower():
                score += 0.3
        
        # Color matching (simplified)
        if image_features.get("colors") and product.get("colors"):
            score += 0.2
        
        # Web entity matching
        if image_features.get("web_entities") and product.get("tags"):
            common_tags = set(image_features["web_entities"]) & set(product.get("tags", []))
            if common_tags:
                score += 0.2
        
        return min(score, 1.0)
    
    def _calculate_label_similarity(self, labels1: List[str], labels2: List[str]) -> float:
        """Calculate similarity between two sets of labels"""
        if not labels1 or not labels2:
            return 0.0
        
        set1 = set(label.lower() for label in labels1)
        set2 = set(label.lower() for label in labels2)
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_color_similarity(
        self, 
        colors1: List[Dict], 
        colors2: List[Dict]
    ) -> float:
        """Calculate color similarity between two color sets"""
        if not colors1 or not colors2:
            return 0.0
        
        # Get dominant colors
        dom_color1 = colors1[0]["rgb"]
        dom_color2 = colors2[0]["rgb"]
        
        # Calculate Euclidean distance in RGB space
        distance = np.sqrt(
            (dom_color1["r"] - dom_color2["r"]) ** 2 +
            (dom_color1["g"] - dom_color2["g"]) ** 2 +
            (dom_color1["b"] - dom_color2["b"]) ** 2
        )
        
        # Normalize (max distance in RGB is ~441.67)
        max_distance = np.sqrt(3 * (255 ** 2))
        similarity = 1 - (distance / max_distance)
        
        return similarity
    
    async def process_search_request(
        self, 
        image_data: bytes, 
        user_query: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Main method to process visual search requests
        
        Args:
            image_data: Image bytes
            user_query: Optional text query from user
            
        Returns:
            Search results and recommendations
        """
        try:
            # Analyze the image
            logger.info("Analyzing uploaded image...")
            image_features = await self.analyze_image(image_data)
            
            # Find similar products
            logger.info("Searching for similar products...")
            similar_products = await self.find_similar_products(image_features)
            
            # Create task for the agent
            task = Task(
                description=f"""
                Based on the image analysis results and user query: {user_query or 'No specific query'},
                provide personalized product recommendations.
                
                Image features: {image_features}
                Similar products found: {len(similar_products)}
                
                Provide:
                1. Top 5 product recommendations with explanations
                2. Alternative options if exact match not found
                3. Styling or usage tips
                """,
                agent=self.agent,
                expected_output="Detailed product recommendations with explanations"
            )
            
            # Execute task
            result = await task.execute()
            
            return {
                "image_analysis": image_features,
                "similar_products": similar_products[:5],
                "recommendations": result,
                "total_matches": len(similar_products)
            }
            
        except Exception as e:
            logger.error(f"Error processing search request: {str(e)}")
            raise