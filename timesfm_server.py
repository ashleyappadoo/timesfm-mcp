import logging
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
import numpy as np
import pandas as pd

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
        """Initialize TimesFM 2.0 PyTorch model - SYNCHRONOUS"""
        try:
            logger.info("Loading TimesFM 2.0 PyTorch model...")
            
            # Import timesfm here to handle import errors gracefully
            import timesfm
            
            # For TimesFM 2.0 PyTorch - using the CORRECT checkpoint
            self.model = timesfm.TimesFm(
                hparams=timesfm.TimesFmHparams(
                    backend="cpu",  # or "gpu" if available
                    per_core_batch_size=32,
                    horizon_len=128,
                    input_patch_len=32,
                    output_patch_len=128,
                    #num_layers=50,  # 50 for 2.0 version (vs 20 for 1.0)
                    num_layers=20,  # 20 pour 1.0 (au lieu de 50)
                    model_dims=1280,
                    use_positional_embedding=False,
                ),
                checkpoint=timesfm.TimesFmCheckpoint(
                    huggingface_repo_id="google/timesfm-1.0-200m-pytorch"  # PYTORCH version!
                )
            )
            
            self.model_loaded = True
            logger.info("TimesFM 2.0 PyTorch model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Failed to load TimesFM 2.0 model: {e}")
            raise

    def forecast_time_series(
        self, 
        data: List[float], 
        horizon: int = 30,
        confidence_intervals: bool = True
    ) -> Dict[str, Any]:
        """Generate forecasts using TimesFM 2.0 - SYNCHRONOUS"""
        
        if not self.model_loaded:
            raise ValueError("TimesFM 2.0 model not loaded")
            
        try:
            # Prepare data
            input_data = np.array(data, dtype=np.float32)
            
            # Ensure minimum length
            if len(input_data) < 10:
                raise ValueError("Need at least 10 data points for forecasting")
            
            # TimesFM 2.0 can handle larger horizons (up to 2048 context)
            horizon = min(horizon, 256)  # Conservative limit
            
            # For TimesFM 2.0, frequency categories:
            # 0: high frequency (daily), 1: medium (weekly/monthly), 2: low (quarterly/yearly)
            frequency = 0  # Default to high frequency
            
            logger.info(f"Starting forecast for {len(data)} data points, horizon {horizon}")
            
            # Generate forecast using TimesFM 2.0 API (synchronous)
            point_forecast, quantile_forecast = self.model.forecast(
                inputs=[input_data],
                freq=[frequency]
            )
            
            logger.info("Forecast completed successfully")
            
            # Process results
            result = {
                "point_forecast": point_forecast[0].tolist() if len(point_forecast) > 0 else [],
                "horizon": horizon,
                "input_length": len(data),
                "timestamp": datetime.now().isoformat(),
                "model_version": "TimesFM-2.0-500M",
                "api_version": "1.3.0+"
            }
            
            # Add quantile forecasts if available (TimesFM 2.0 has better quantile support)
            if confidence_intervals and quantile_forecast is not None and len(quantile_forecast) > 0:
                result["confidence_intervals"] = {
                    "quantiles": quantile_forecast[0].tolist(),
                    "quantile_levels": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                    "note": "Quantile forecasts from TimesFM 2.0 (experimental)"
                }
            else:
                result["confidence_intervals"] = {
                    "note": "Quantile forecasts not available in this prediction"
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Forecasting error: {e}")
            raise

# Simple HTTP server for testing
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
            response = {
                "status": "healthy", 
                "model_loaded": self.timesfm_server.model_loaded,
                "timesfm_version": "2.0-500M",
                "api_version": "1.3.0+",
                "backend": "PyTorch",
                "note": "Real TimesFM model from Google Research"
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
                
                # SYNCHRONOUS call - no asyncio!
                result = self.timesfm_server.forecast_time_series(
                    data=data.get('data', []),
                    horizon=data.get('horizon', 30),
                    confidence_intervals=data.get('confidence_intervals', True)
                )
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(result).encode())
                
            except Exception as e:
                logger.error(f"Request error: {e}")
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                error_response = {"error": str(e)}
                self.wfile.write(json.dumps(error_response).encode())

def make_handler(timesfm_server):
    def handler(*args, **kwargs):
        return TimesFMHandler(*args, timesfm_server=timesfm_server, **kwargs)
    return handler

def main():
    """SYNCHRONOUS main function"""
    # Initialize TimesFM server
    timesfm_server = TimesFMServer()
    
    # Load model (synchronous)
    timesfm_server.initialize_model()
    
    # Start HTTP server
    handler = make_handler(timesfm_server)
    httpd = HTTPServer(('0.0.0.0', 8080), handler)
    
    logger.info("TimesFM 2.0 server starting on port 8080...")
    httpd.serve_forever()

if __name__ == "__main__":
    main()  # NO asyncio.run!
