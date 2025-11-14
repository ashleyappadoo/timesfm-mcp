#!/usr/bin/env python3
"""
TimesFM Railway Server: Pure Python (no numpy)
Ultra-lightweight forecasting server
"""

import json
import logging
import os
import urllib.parse
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("timesfm-railway")

class TimesFMRailwayServer:
    def __init__(self):
        self.model_id = "timesfm-optimized-railway"
        self.is_ready = True

    def get_status(self) -> Dict[str, Any]:
        """Get server status"""
        return {
            "status": "healthy",
            "service": "TimesFM Forecasting Server",
            "version": "railway-optimized-pure-python",
            "model_id": self.model_id,
            "algorithm": "Trend-based (pure Python)",
            "memory_usage": "~25MB",
            "ready": self.is_ready,
            "timestamp": datetime.now().isoformat(),
            "platform": "Railway"
        }

    def forecast(self, data: List[float], horizon: int = 5) -> Dict[str, Any]:
        """Generate forecast using pure Python"""
        if not data or len(data) < 3:
            raise ValueError("Need at least 3 data points")
        
        horizon = min(horizon, 32)
        
        try:
            # Convert to float list (pure Python)
            values = [float(x) for x in data]
            
            # Calculate trend using simple linear regression
            n = len(values)
            
            # Simple trend calculation
            if n >= 3:
                # Calculate trend from last 3 points for stability
                recent_values = values[-3:]
                x_coords = list(range(len(recent_values)))
                
                # Simple linear regression: y = ax + b
                sum_x = sum(x_coords)
                sum_y = sum(recent_values)
                sum_xy = sum(x * y for x, y in zip(x_coords, recent_values))
                sum_x2 = sum(x * x for x in x_coords)
                
                # Calculate slope (trend)
                if len(recent_values) * sum_x2 - sum_x * sum_x != 0:
                    trend = (len(recent_values) * sum_xy - sum_x * sum_y) / (len(recent_values) * sum_x2 - sum_x * sum_x)
                else:
                    trend = 0
            else:
                # Fallback for small datasets
                trend = values[-1] - values[-2] if len(values) >= 2 else 0
            
            # Generate forecast with trend dampening
            forecast = []
            last_value = values[-1]
            
            for i in range(horizon):
                # Apply trend with dampening over time
                dampening_factor = max(0.95 ** i, 0.3)  # Gradually reduce trend impact
                predicted_value = last_value + (trend * (i + 1) * dampening_factor)
                
                # Add slight randomness for realism (optional)
                forecast.append(round(predicted_value, 2))
            
            # Calculate some basic statistics
            mean_value = sum(values) / len(values)
            
            result = {
                "success": True,
                "point_forecast": forecast,
                "horizon": horizon,
                "input_length": len(data),
                "input_data": values[-5:],  # Last 5 points
                "input_mean": round(mean_value, 2),
                "detected_trend": round(trend, 4),
                "model_version": "TimesFM-Railway-Pure-Python",
                "algorithm": "Linear trend with dampening",
                "timestamp": datetime.now().isoformat(),
                "platform": "Railway (zero dependencies)"
            }
            
            logger.info(f"✅ Forecast: {len(data)} → {horizon} points (trend: {trend:.3f})")
            return result
            
        except Exception as e:
            logger.error(f"❌ Forecast error: {e}")
            raise

# HTTP Handler
class TimesFMHTTPHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, server_instance=None, **kwargs):
        self.server_instance = server_instance
        super().__init__(*args, **kwargs)
    
    def log_message(self, format, *args):
        pass  # Suppress HTTP logs
        
    def do_GET(self):
        if self.path == '/health':
            # Railway healthcheck
            status = self.server_instance.get_status()
            self._send_json_response(200, status)
            
        elif self.path == '/' or self.path == '/status':
            self._send_json_response(200, {
                "service": "TimesFM Forecasting Server",
                "version": "Railway Pure Python",
                "endpoints": ["/health", "/forecast (GET/POST)", "/status"],
                "dependencies": "Zero external dependencies",
                "algorithm": "Pure Python trend forecasting",
                "status": "ready"
            })
            
        elif self.path.startswith('/forecast'):
            # NEW: GET support for forecast
            try:
                # Parse URL parameters
                from urllib.parse import urlparse, parse_qs
                parsed_url = urlparse(self.path)
                params = parse_qs(parsed_url.query)
                
                # Extract parameters
                if 'ts' not in params:
                    self._send_json_response(400, {
                        "error": "Missing 'ts' parameter",
                        "usage": "/forecast?ts=[timeseries_json]&h=6&f=monthly"
                    })
                    return
                
                # Decode timeseries data
                timeseries_json = params['ts'][0]
                timeseries_data = json.loads(timeseries_json)
                
                # Extract other parameters with defaults
                horizon = int(params.get('h', ['6'])[0])
                frequency = params.get('f', ['monthly'])[0]
                
                logger.info(f"📊 GET Forecast: {len(timeseries_data)} points → {horizon} steps ({frequency})")
                
                # Process timeseries data - extract values only
                if isinstance(timeseries_data[0], list):
                    # Format: [[timestamp, value], [timestamp, value], ...]
                    data_values = [point[1] for point in timeseries_data]
                else:
                    # Format: [value1, value2, value3, ...]
                    data_values = timeseries_data
                
                # Validate
                if len(data_values) < 3:
                    self._send_json_response(400, {"error": "Need at least 3 data points"})
                    return
                
                # Generate forecast using existing logic
                result = self.server_instance.forecast(data_values, horizon)
                
                # Add metadata about the request
                result.update({
                    "request_method": "GET",
                    "frequency": frequency,
                    "timeseries_length": len(timeseries_data)
                })
                
                self._send_json_response(200, result)
                
            except json.JSONDecodeError:
                self._send_json_response(400, {"error": "Invalid JSON in 'ts' parameter"})
            except ValueError as e:
                self._send_json_response(400, {"error": str(e)})
            except Exception as e:
                logger.error(f"❌ GET forecast error: {e}")
                self._send_json_response(500, {"error": "Internal server error"})
            
        else:
            self._send_json_response(404, {"error": "Endpoint not found"})
    
    def do_POST(self):
        if self.path == '/forecast':
            try:
                # Read request
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                data = json.loads(post_data.decode('utf-8'))
                
                # Extract parameters
                input_data = data.get('data', [])
                horizon = data.get('horizon', 5)
                
                logger.info(f"📊 Forecast request: {len(input_data)} points → {horizon} steps")
                
                # Validate
                if not input_data:
                    self._send_json_response(400, {"error": "No data provided"})
                    return
                    
                if len(input_data) < 3:
                    self._send_json_response(400, {"error": "Need at least 3 data points"})
                    return
                
                # Generate forecast
                result = self.server_instance.forecast(input_data, horizon)
                self._send_json_response(200, result)
                
            except json.JSONDecodeError:
                self._send_json_response(400, {"error": "Invalid JSON format"})
            except ValueError as e:
                self._send_json_response(400, {"error": str(e)})
            except Exception as e:
                logger.error(f"❌ Request error: {e}")
                self._send_json_response(500, {"error": "Internal server error"})
        else:
            self._send_json_response(404, {"error": "Endpoint not found"})
    
    def _send_json_response(self, status_code: int, data: Dict):
        self.send_response(status_code)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode())

def make_handler(server_instance):
    def handler(*args, **kwargs):
        return TimesFMHTTPHandler(*args, server_instance=server_instance, **kwargs)
    return handler

def main():
    """Main entry point"""
    logger.info("🚀 Starting TimesFM Railway Server (Pure Python)")
    
    # Initialize server
    server = TimesFMRailwayServer()
    
    # Get port
    port = int(os.environ.get('PORT', 8080))
    
    # Start HTTP server
    handler = make_handler(server)
    httpd = HTTPServer(('0.0.0.0', port), handler)
    
    logger.info(f"✅ Server ready on port {port}")
    logger.info("🔗 Endpoints: /health, /forecast, /status")
    logger.info("🪶 Pure Python - zero external dependencies")
    logger.info("💾 Ultra-lightweight (~25MB)")
    logger.info("🎯 Railway healthchecks enabled!")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("🛑 Server stopped")
        httpd.shutdown()

if __name__ == "__main__":
    main()
