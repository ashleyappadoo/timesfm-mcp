import logging
import json
import gc
import os
from typing import Dict, List, Optional, Any
from datetime import datetime
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("timesfm-mcp")

class TimesFMServer:
    def __init__(self):
        self.model = None
        self.model_loaded = False
        
    def initialize_model(self):
        """Initialize TimesFM 1.0 PyTorch model with memory optimization"""
        try:
            logger.info("Loading TimesFM 1.0 PyTorch model (200M - Memory Optimized)...")
            
            # Force garbage collection before loading
            gc.collect()
            
            # Import timesfm here to handle import errors gracefully
            import timesfm
            
            # OPTIMIZED: Smaller batch size and optimized settings for 512MB
            self.model = timesfm.TimesFm(
                hparams=timesfm.TimesFmHparams(
                    backend="cpu",
                    per_core_batch_size=8,     # REDUCED from 32 to 8
                    horizon_len=64,            # REDUCED from 128 to 64
                    input_patch_len=16,        # REDUCED from 32 to 16  
                    output_patch_len=64,       # REDUCED from 128 to 64
                    num_layers=20,             # TimesFM 1.0 (vs 50 for 2.0)
                    model_dims=1280,           # Keep standard
                    use_positional_embedding=False,
                ),
                checkpoint=timesfm.TimesFmCheckpoint(
                    huggingface_repo_id="google/timesfm-1.0-200m-pytorch"  # TimesFM 1.0 (lighter)
                )
            )
            
            # Force garbage collection after loading
            gc.collect()
            
            self.model_loaded = True
            logger.info("TimesFM 1.0 PyTorch model loaded successfully with memory optimization!")
            
        except Exception as e:
            logger.error(f"Failed to load TimesFM 1.0 model: {e}")
            # Try even more aggressive memory cleanup
            gc.collect()
            raise

    def forecast_time_series(
        self, 
        data: List[float], 
        horizon: int = 30,
        confidence_intervals: bool = False  # DEFAULT TO FALSE to save memory
    ) -> Dict[str, Any]:
        """Generate forecasts using TimesFM 1.0 with memory optimization"""
        
        if not self.model_loaded:
            raise ValueError("TimesFM 1.0 model not loaded")
            
        try:
            # Prepare data
            input_data = np.array(data, dtype=np.float32)
            
            # Ensure minimum length
            if len(input_data) < 10:
                raise ValueError("Need at least 10 data points for forecasting")
            
            # MEMORY OPTIMIZATION: Limit horizon for small memory
            horizon = min(horizon, 32)  # VERY conservative limit for 512MB
            
            # Clean memory before forecast
            gc.collect()
            
            logger.info(f"Starting forecast for {len(data)} data points, horizon {horizon}")
            
            # Generate forecast using TimesFM 1.0 API (simplified)
            try:
                point_forecast, quantile_forecast = self.model.forecast(
                    inputs=[input_data],
                    freq=[0]  # Default to high frequency
                )
            except Exception as forecast_error:
                logger.warning(f"Forecast with quantiles failed: {forecast_error}")
                # Fallback: try without quantiles
                point_forecast, _ = self.model.forecast(
                    inputs=[input_data],
                    freq=[0]
                )
                quantile_forecast = None
            
            # Clean memory after forecast
            gc.collect()
            
            logger.info("Forecast completed successfully")
            
            # Process results (minimal memory usage)
            result = {
                "point_forecast": point_forecast[0].tolist() if len(point_forecast) > 0 else [],
                "horizon": len(point_forecast[0]) if len(point_forecast) > 0 else horizon,
                "input_length": len(data),
                "timestamp": datetime.now().isoformat(),
                "model_version": "TimesFM-1.0-200M",
                "memory_optimized": True,
                "platform": "Render-512MB"
            }
            
            # Only add confidence intervals if explicitly requested AND available
            if confidence_intervals and quantile_forecast is not None and len(quantile_forecast) > 0:
                result["confidence_intervals"] = {
                    "quantiles": quantile_forecast[0].tolist(),
                    "note": "Limited quantiles for memory optimization"
                }
            else:
                result["confidence_intervals"] = {
                    "note": "Disabled for memory optimization on Render"
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Forecasting error: {e}")
            # Emergency memory cleanup
            gc.collect()
            raise

# Lightweight HTTP server
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse

class TimesFMHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, timesfm_server=None, **kwargs):
        self.timesfm_server = timesfm_server
        super().__init__(*args, **kwargs)
        
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            # Memory status
            import psutil
            memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
            
            response = {
                "status": "healthy", 
                "model_loaded": self.timesfm_server.model_loaded,
                "timesfm_version": "1.0-200M",
                "memory_optimized": True,
                "current_memory_mb": round(memory_mb, 2),
                "platform": "Render (512MB limit)",
                "backend": "PyTorch CPU"
            }
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.end_headers()
            
    def do_POST(self):
        if self.path == '/forecast':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode('utf-8'))
                
                result = self.timesfm_server.forecast_time_series(
                    data=data.get('data', []),
                    horizon=min(data.get('horizon', 30), 32),  # Force limit
                    confidence_intervals=data.get('confidence_intervals', False)  # Default False
                )
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(result).encode())
                
            except Exception as e:
                logger.error(f"Request error: {e}")
                # Emergency cleanup
                gc.collect()
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                error_response = {
                    "error": str(e),
                    "note": "Memory-constrained environment (Render 512MB)"
                }
                self.wfile.write(json.dumps(error_response).encode())

def make_handler(timesfm_server):
    def handler(*args, **kwargs):
        return TimesFMHandler(*args, timesfm_server=timesfm_server, **kwargs)
    return handler

def main():
    """Memory-optimized main function for Render"""
    logger.info("Starting TimesFM 1.0 server with memory optimization for Render...")
    
    # Force initial memory cleanup
    gc.collect()
    
    # Initialize TimesFM server
    timesfm_server = TimesFMServer()
    
    # Load model (with memory optimization)
    timesfm_server.initialize_model()
    
    # Start HTTP server
    handler = make_handler(timesfm_server)
    
    # Use port from environment (Render requirement)
    port = int(os.environ.get('PORT', 8080))
    httpd = HTTPServer(('0.0.0.0', port), handler)
    
    logger.info(f"TimesFM 1.0 Memory-Optimized server starting on port {port}...")
    logger.info("Optimized for Render 512MB environment")
    
    httpd.serve_forever()

if __name__ == "__main__":
    main()
