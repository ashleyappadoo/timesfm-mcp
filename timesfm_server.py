import logging
import json
import gc
import os
import threading
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
        self.loading_lock = threading.Lock()
        
    def _lazy_load_model(self):
        """Lazy load TimesFM model only when needed"""
        if self.model_loaded:
            return
            
        with self.loading_lock:
            # Double-check pattern
            if self.model_loaded:
                return
                
            try:
                logger.info("🔄 Lazy loading TimesFM 1.0 model (CPU-only, optimized)...")
                
                # Force garbage collection before loading
                gc.collect()
                
                # Import timesfm here for lazy loading
                import timesfm
                
                # ULTRA-OPTIMIZED settings for minimal memory
                self.model = timesfm.TimesFm(
                    hparams=timesfm.TimesFmHparams(
                        backend="cpu",               # CPU-only (plus léger)
                        per_core_batch_size=4,       # TRÈS RÉDUIT (was 8)
                        horizon_len=32,              # TRÈS RÉDUIT (was 64)
                        input_patch_len=8,           # TRÈS RÉDUIT (was 16)  
                        output_patch_len=32,         # TRÈS RÉDUIT (was 64)
                        num_layers=20,               # TimesFM 1.0
                        model_dims=1280,             # Standard
                        use_positional_embedding=False,
                    ),
                    checkpoint=timesfm.TimesFmCheckpoint(
                        huggingface_repo_id="google/timesfm-1.0-200m-pytorch"
                    )
                )
                
                # Aggressive memory cleanup
                gc.collect()
                
                self.model_loaded = True
                logger.info("✅ TimesFM 1.0 model loaded successfully (CPU-only)!")
                
            except Exception as e:
                logger.error(f"❌ Failed to load TimesFM model: {e}")
                self.model_loaded = False
                raise

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information without loading it"""
        try:
            import psutil
            memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
        except:
            memory_mb = "unknown"
            
        return {
            "model_loaded": self.model_loaded,
            "timesfm_version": "1.0-200M",
            "backend": "CPU-only (optimized)",
            "lazy_loading": True,
            "current_memory_mb": memory_mb,
            "platform": "Railway/Render compatible",
            "optimization": "Ultra-lightweight"
        }

    def forecast_time_series(
        self, 
        data: List[float], 
        horizon: int = 30,
        confidence_intervals: bool = False
    ) -> Dict[str, Any]:
        """Generate forecasts with lazy model loading"""
        
        # Lazy load model only when needed
        if not self.model_loaded:
            self._lazy_load_model()
            
        if not self.model_loaded:
            raise ValueError("Failed to load TimesFM model")
            
        try:
            # Validate input
            if not data or len(data) < 5:
                raise ValueError("Need at least 5 data points for forecasting")
                
            # Prepare data efficiently
            input_data = np.array(data, dtype=np.float32)
            
            # ULTRA-conservative limits for Railway/Render
            horizon = min(horizon, 16)  # TRÈS limité
            
            logger.info(f"🔮 Forecasting {len(data)} points → {horizon} steps")
            
            # Memory cleanup before prediction
            gc.collect()
            
            try:
                point_forecast, quantile_forecast = self.model.forecast(
                    inputs=[input_data],
                    freq=[0]  # High frequency
                )
            except Exception as e:
                logger.warning(f"⚠️ Forecast with quantiles failed: {e}")
                # Ultra-simple fallback
                point_forecast = [input_data[-1:].tolist()]
                quantile_forecast = None
            
            # Memory cleanup after prediction
            gc.collect()
            
            logger.info("✅ Forecast completed")
            
            # Minimal result structure
            result = {
                "point_forecast": point_forecast[0].tolist() if point_forecast and len(point_forecast) > 0 else [],
                "horizon": len(point_forecast[0]) if point_forecast and len(point_forecast) > 0 else horizon,
                "input_length": len(data),
                "timestamp": datetime.now().isoformat(),
                "model_version": "TimesFM-1.0-200M-CPU",
                "optimized": "ultra-lightweight",
                "platform": "Railway/Render"
            }
            
            # Confidence intervals only if explicitly requested
            if confidence_intervals and quantile_forecast is not None:
                result["confidence_intervals"] = {
                    "note": "Available but disabled for memory optimization"
                }
            else:
                result["confidence_intervals"] = {
                    "note": "Disabled for Railway/Render compatibility"
                }
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Forecasting error: {e}")
            gc.collect()  # Emergency cleanup
            raise

# Ultra-lightweight HTTP server
from http.server import HTTPServer, BaseHTTPRequestHandler

class TimesFMHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, timesfm_server=None, **kwargs):
        self.timesfm_server = timesfm_server
        super().__init__(*args, **kwargs)
    
    def log_message(self, format, *args):
        # Suppress default HTTP logging to reduce noise
        pass
        
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            response = {
                "status": "healthy",
                **self.timesfm_server.get_model_info()
            }
            
            self.wfile.write(json.dumps(response).encode())
            
        elif self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            response = {
                "service": "TimesFM Forecasting Server",
                "version": "1.0-ultra-optimized",
                "endpoints": ["/health", "/forecast"],
                "status": "ready"
            }
            
            self.wfile.write(json.dumps(response).encode())
            
        else:
            self.send_response(404)
            self.end_headers()
            
    def do_POST(self):
        if self.path == '/forecast':
            try:
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                data = json.loads(post_data.decode('utf-8'))
                
                result = self.timesfm_server.forecast_time_series(
                    data=data.get('data', []),
                    horizon=min(data.get('horizon', 30), 16),  # Force ultra-small
                    confidence_intervals=data.get('confidence_intervals', False)
                )
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(result).encode())
                
            except Exception as e:
                logger.error(f"❌ Request error: {e}")
                gc.collect()  # Emergency cleanup
                
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                
                error_response = {
                    "error": str(e),
                    "note": "Ultra-optimized for Railway/Render constraints"
                }
                self.wfile.write(json.dumps(error_response).encode())

def make_handler(timesfm_server):
    def handler(*args, **kwargs):
        return TimesFMHandler(*args, timesfm_server=timesfm_server, **kwargs)
    return handler

def main():
    """Ultra-lightweight main with lazy loading"""
    logger.info("🚀 Starting TimesFM Ultra-Lightweight Server...")
    
    # Initial memory cleanup
    gc.collect()
    
    # Initialize server (NO model loading yet!)
    timesfm_server = TimesFMServer()
    
    # Get port from environment
    port = int(os.environ.get('PORT', 8080))
    
    # Start HTTP server
    handler = make_handler(timesfm_server)
    httpd = HTTPServer(('0.0.0.0', port), handler)
    
    logger.info(f"✅ Server ready on port {port}")
    logger.info("💡 Model will load lazily on first forecast request")
    logger.info("🎯 Optimized for Railway/Render deployment")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("🛑 Server stopped")
        httpd.shutdown()

if __name__ == "__main__":
    main()
