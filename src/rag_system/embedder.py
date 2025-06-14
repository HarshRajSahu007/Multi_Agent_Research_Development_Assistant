from sentence_transformers import SentenceTransformer
import torch
from typing import List, Dict, Any, Union
import numpy as np
import logging
from PIL import Image
import clip


class MultiModalEmbedder:
    """Handles embedding generation for text and visual content."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize text embedding model
        self.text_model = SentenceTransformer(config["models"]["embedding_model"])
        
        # Initialize CLIP for image embeddings
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        except:
            self.logger.warning("CLIP model not available, image embeddings disabled")
            self.clip_model = None
    
    def embed_text(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings for text content."""
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.text_model.encode(texts)
        return embeddings
    
    def embed_image(self, image: Image.Image) -> np.ndarray:
        """Generate embeddings for image content using CLIP."""
        if self.clip_model is None:
            return np.zeros(512)  # Return zero vector if CLIP not available
        
        try:
            image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            return image_features.cpu().numpy()[0]
        
        except Exception as e:
            self.logger.error(f"Image embedding failed: {e}")
            return np.zeros(512)
    
    def embed_multimodal(self, text: str, image: Image.Image = None) -> np.ndarray:
        """Generate combined embeddings for text and image."""
        text_emb = self.embed_text(text)[0]
        
        if image is not None:
            image_emb = self.embed_image(image)
            # Simple concatenation - could be improved with learned fusion
            combined_emb = np.concatenate([text_emb, image_emb])
        else:
            # Pad with zeros if no image
            combined_emb = np.concatenate([text_emb, np.zeros(512)])
        
        return combined_emb
