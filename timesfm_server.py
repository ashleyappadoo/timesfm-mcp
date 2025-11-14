#!/usr/bin/env python3
"""
TimesFM Railway Server: Pure Python with Seasonality Detection
Ultra-lightweight forecasting server with enhanced seasonal analysis
"""

import json
import logging
import os
import urllib.parse
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, Any, List, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("timesfm-railway")

class TimesFMRailwayServer:
    def __init__(self):
        self.model_id = "timesfm-seasonal-railway"
        self.is_ready = True

    def get_status(self) -> Dict[str, Any]:
        """Get server status"""
        return {
            "status": "healthy",
            "service": "TimesFM Seasonal Forecasting Server",
            "version": "railway-seasonal-pure-python",
            "model_id": self.model_id,
            "features": [
                "Trend detection",
                "Seasonality detection", 
                "Cycle identification",
                "Business insights",
                "GET/POST support"
            ],
            "memory_usage": "~30MB",
            "ready": self.is_ready,
            "timestamp": datetime.now().isoformat(),
            "platform": "Railway"
        }

    def detect_seasonality(self, data: List[float]) -> Dict[str, Any]:
        """Detect seasonal patterns in the data"""
        n = len(data)
        seasonality_info = {
            "has_seasonality": False,
            "period": None,
            "strength": 0.0,
            "type": "none"
        }
        
        if n < 8:  # Need minimum data for seasonality detection
            return seasonality_info
            
        try:
            # Test common business periods
            periods_to_test = []
            
            if n >= 12:
                periods_to_test.append(12)  # Monthly (annual cycle)
            if n >= 4:
                periods_to_test.append(4)   # Quarterly
            if n >= 7:
                periods_to_test.append(7)   # Weekly
            if n >= 6:
                periods_to_test.append(6)   # Bi-monthly
                
            best_period = None
            best_score = 0.0
            
            for period in periods_to_test:
                if period < n // 2:  # Need at least 2 full cycles
                    score = self._calculate_seasonality_score(data, period)
                    if score > best_score:
                        best_score = score
                        best_period = period
            
            # Threshold for significant seasonality
            if best_score > 0.3:
                seasonality_info = {
                    "has_seasonality": True,
                    "period": best_period,
                    "strength": round(best_score, 3),
                    "type": self._get_seasonality_type(best_period),
                    "description": f"Detected {self._get_seasonality_type(best_period)} pattern (period={best_period})"
                }
            
            return seasonality_info
            
        except Exception as e:
            logger.warning(f"Seasonality detection error: {e}")
            return seasonality_info

    def _calculate_seasonality_score(self, data: List[float], period: int) -> float:
        """Calculate seasonality score for a given period"""
        try:
            n = len(data)
            if period >= n:
                return 0.0
                
            # Calculate autocorrelation at the seasonal lag
            autocorr = self._autocorrelation(data, period)
            
            # Calculate coefficient of variation within seasons
            seasonal_means = []
            for i in range(period):
                season_values = [data[j] for j in range(i, n, period)]
                if len(season_values) >= 2:
                    seasonal_means.append(sum(season_values) / len(season_values))
            
            if len(seasonal_means) < 2:
                return 0.0
                
            # Variance between seasonal means
            overall_mean = sum(seasonal_means) / len(seasonal_means)
            seasonal_var = sum((x - overall_mean) ** 2 for x in seasonal_means) / len(seasonal_means)
            
            # Overall variance
            data_mean = sum(data) / len(data)
            total_var = sum((x - data_mean) ** 2 for x in data) / len(data)
            
            if total_var == 0:
                return 0.0
                
            # Combine autocorrelation and seasonal variance
            variance_ratio = seasonal_var / total_var if total_var > 0 else 0
            score = abs(autocorr) * 0.7 + variance_ratio * 0.3
            
            return min(score, 1.0)
            
        except:
            return 0.0

    def _autocorrelation(self, data: List[float], lag: int) -> float:
        """Calculate autocorrelation at given lag"""
        try:
            n = len(data)
            if lag >= n:
                return 0.0
                
            mean = sum(data) / n
            
            # Calculate covariance at lag
            numerator = sum((data[i] - mean) * (data[i + lag] - mean) 
                           for i in range(n - lag))
            
            # Calculate variance
            denominator = sum((x - mean) ** 2 for x in data)
            
            if denominator == 0:
                return 0.0
                
            return numerator / denominator
            
        except:
            return 0.0

    def _get_seasonality_type(self, period: int) -> str:
        """Get human-readable seasonality type"""
        if period == 12:
            return "annual"
        elif period == 4:
            return "quarterly"
        elif period == 7:
            return "weekly"
        elif period == 6:
            return "bi-monthly"
        else:
            return f"custom-{period}"

    def _extract_seasonal_component(self, data: List[float], period: int) -> List[float]:
        """Extract seasonal component"""
        n = len(data)
        seasonal = [0.0] * n
        
        try:
            # Calculate seasonal indices
            seasonal_sums = [0.0] * period
            seasonal_counts = [0] * period
            
            for i in range(n):
                season_idx = i % period
                seasonal_sums[season_idx] += data[i]
                seasonal_counts[season_idx] += 1
            
            # Average for each season
            seasonal_avgs = [seasonal_sums[i] / max(seasonal_counts[i], 1) 
                           for i in range(period)]
            
            # Center the seasonal component
            seasonal_mean = sum(seasonal_avgs) / period
            seasonal_avgs = [x - seasonal_mean for x in seasonal_avgs]
            
            # Apply to full series
            for i in range(n):
                seasonal[i] = seasonal_avgs[i % period]
                
        except:
            pass
            
        return seasonal

    def _calculate_slope(self, x: List[float], y: List[float]) -> float:
        """Calculate linear regression slope"""
        try:
            n = len(x)
            if n < 2:
                return 0.0
                
            sum_x = sum(x)
            sum_y = sum(y)
            sum_xy = sum(x[i] * y[i] for i in range(n))
            sum_x2 = sum(xi * xi for xi in x)
            
            denominator = n * sum_x2 - sum_x * sum_x
            if abs(denominator) < 1e-10:
                return 0.0
                
            slope = (n * sum_xy - sum_x * sum_y) / denominator
            return slope
            
        except:
            return 0.0

    def _generate_business_insights(self, seasonality_info: Dict, trend_slope: float, data: List[float], frequency: str = "monthly") -> Dict[str, str]:
        """Generate business insights based on seasonal analysis"""
        insights = {}
        
        # Seasonality insights
        if seasonality_info["has_seasonality"]:
            period = seasonality_info["period"]
            strength = seasonality_info.get("strength", 0)
            
            if period == 12:
                insights["seasonality"] = f"Strong annual pattern detected (strength: {strength:.1%}). Plan inventory and marketing around seasonal peaks."
            elif period == 4:
                insights["seasonality"] = f"Quarterly business cycles identified (strength: {strength:.1%}). Align with fiscal quarters and business reviews."
            elif period == 7:
                insights["seasonality"] = f"Weekly patterns found (strength: {strength:.1%}). Consider day-of-week effects in operations."
            else:
                insights["seasonality"] = f"Custom {period}-period cycle detected. Monitor this recurring pattern for planning."
        else:
            insights["seasonality"] = "No significant seasonal patterns detected. Focus on trend-based strategies."
        
        # Trend insights with frequency context
        if abs(trend_slope) > 0.1:
            if trend_slope > 0:
                insights["trend"] = f"Positive growth trend in {frequency} data. Consider scaling resources and expanding market reach."
            else:
                insights["trend"] = f"Declining trend in {frequency} data. Investigate causes and implement corrective measures."
        else:
            insights["trend"] = f"Stable {frequency} trend with minimal growth. Focus on efficiency and market maintenance."
        
        # Volatility insights
        if len(data) > 1:
            volatility = self._variance(data) ** 0.5
            mean_value = sum(data) / len(data)
            cv = volatility / abs(mean_value) if mean_value != 0 else 0
            
            if cv > 0.3:
                insights["volatility"] = "High volatility detected. Implement risk management and flexible planning."
            elif cv > 0.15:
                insights["volatility"] = "Moderate volatility. Monitor closely and maintain adaptive strategies."
            else:
                insights["volatility"] = "Low volatility indicates stable performance. Suitable for long-term planning."
        
        return insights

    def _variance(self, data: List[float]) -> float:
        """Calculate variance"""
        if not data:
            return 0.0
        mean = sum(data) / len(data)
        return sum((x - mean) ** 2 for x in data) / len(data)

    def forecast(self, data: List[float], horizon: int = 5, frequency: str = "monthly", enable_seasonality: bool = True) -> Dict[str, Any]:
        """Generate forecast with optional seasonal enhancement"""
        if not data or len(data) < 3:
            raise ValueError("Need at least 3 data points")
        
        horizon = min(horizon, 32)
        
        try:
            # Convert to float list (pure Python)
            values = [float(x) for x in data]
            n = len(values)
            
            # Detect seasonality if enabled and sufficient data
            seasonality_info = {"has_seasonality": False}
            seasonal_component = [0.0] * n
            
            if enable_seasonality and n >= 8:
                seasonality_info = self.detect_seasonality(values)
                if seasonality_info["has_seasonality"]:
                    seasonal_component = self._extract_seasonal_component(values, seasonality_info["period"])
            
            # Detrend data for trend calculation (remove seasonal component)
            detrended = [values[i] - seasonal_component[i] for i in range(n)]
            
            # Calculate trend using simple linear regression on detrended data
            if n >= 3:
                # Use more data points for trend if available
                trend_data = detrended[-min(n, 6):]  # Last 6 points or all data
                x_coords = list(range(len(trend_data)))
                
                # Simple linear regression: y = ax + b
                sum_x = sum(x_coords)
                sum_y = sum(trend_data)
                sum_xy = sum(x * y for x, y in zip(x_coords, trend_data))
                sum_x2 = sum(x * x for x in x_coords)
                
                # Calculate slope (trend)
                if len(trend_data) * sum_x2 - sum_x * sum_x != 0:
                    trend = (len(trend_data) * sum_xy - sum_x * sum_y) / (len(trend_data) * sum_x2 - sum_x * sum_x)
                else:
                    trend = 0
            else:
                # Fallback for small datasets
                trend = detrended[-1] - detrended[-2] if len(detrended) >= 2 else 0
            
            # Generate forecast
            forecast = []
            last_value = values[-1]
            last_detrended = detrended[-1]
            
            for i in range(horizon):
                # Trend component with dampening
                dampening_factor = max(0.95 ** i, 0.3)
                trend_value = last_detrended + (trend * (i + 1) * dampening_factor)
                
                # Seasonal component
                seasonal_value = 0.0
                if seasonality_info["has_seasonality"]:
                    period = seasonality_info["period"]
                    future_season_idx = (n + i) % period
                    if future_season_idx < len(seasonal_component):
                        seasonal_value = seasonal_component[future_season_idx]
                    else:
                        # Use average seasonal pattern
                        seasonal_pattern = []
                        for j in range(period):
                            pattern_values = [seasonal_component[k] for k in range(j, len(seasonal_component), period)]
                            if pattern_values:
                                seasonal_pattern.append(sum(pattern_values) / len(pattern_values))
                            else:
                                seasonal_pattern.append(0.0)
                        seasonal_value = seasonal_pattern[future_season_idx]
                
                # Combine components
                predicted_value = trend_value + seasonal_value
                forecast.append(round(predicted_value, 2))
            
            # Calculate statistics
            mean_value = sum(values) / len(values)
            
            # Build result
            result = {
                "success": True,
                "point_forecast": forecast,
                "horizon": horizon,
                "input_length": len(data),
                "input_data": values[-5:],  # Last 5 points
                "input_mean": round(mean_value, 2),
                "detected_trend": round(trend, 4),
                "model_version": "TimesFM-Seasonal-Railway-Pure-Python",
                "algorithm": "Seasonal trend decomposition" if seasonality_info["has_seasonality"] else "Linear trend with dampening",
                "timestamp": datetime.now().isoformat(),
                "platform": "Railway (zero dependencies)",
                "frequency": frequency
            }
            
            # Add seasonality information if detected
            if seasonality_info["has_seasonality"]:
                result["seasonality"] = {
                    "detected": True,
                    "period": seasonality_info["period"],
                    "strength": seasonality_info["strength"],
                    "type": seasonality_info["type"],
                    "description": seasonality_info.get("description", "")
                }
            else:
                result["seasonality"] = {"detected": False}
            
            # Add business insights
            result["business_insights"] = self._generate_business_insights(
                seasonality_info, trend, values, frequency
            )
            
            logger.info(f"✅ Forecast: {len(data)} → {horizon} points (trend: {trend:.3f}, seasonal: {seasonality_info['has_seasonality']})")
            return result
            
        except Exception as e:
            logger.error(f"❌ Forecast error: {e}")
            raise

# HTTP Handler (Enhanced with seasonal support)
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
                "service": "TimesFM Seasonal Forecasting Server",
                "version": "Railway Seasonal Pure Python",
                "endpoints": ["/health", "/forecast (GET/POST)", "/seasonal-analysis", "/status"],
                "features": [
                    "Trend detection",
                    "Seasonality detection",
                    "Business insights",
                    "GET/POST support",
                    "URL parameters"
                ],
                "dependencies": "Zero external dependencies",
                "algorithm": "Seasonal trend decomposition + Linear regression",
                "status": "ready"
            })
            
        elif self.path.startswith('/forecast'):
            # GET support for forecast with seasonal enhancement
            try:
                # Parse URL parameters
                from urllib.parse import urlparse, parse_qs
                parsed_url = urlparse(self.path)
                params = parse_qs(parsed_url.query)
                
                # Extract parameters
                if 'ts' not in params:
                    self._send_json_response(400, {
                        "error": "Missing 'ts' parameter",
                        "usage": "/forecast?ts=[timeseries_json]&h=6&f=monthly&s=true"
                    })
                    return
                
                # Decode timeseries data
                timeseries_json = params['ts'][0]
                timeseries_data = json.loads(timeseries_json)
                
                # Extract other parameters with defaults
                horizon = int(params.get('h', ['6'])[0])
                frequency = params.get('f', ['monthly'])[0]
                enable_seasonality = params.get('s', ['true'])[0].lower() in ['true', '1', 'yes']
                
                logger.info(f"📊 GET Forecast: {len(timeseries_data)} points → {horizon} steps ({frequency}, seasonal: {enable_seasonality})")
                
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
                
                # Generate forecast using enhanced logic
                result = self.server_instance.forecast(data_values, horizon, frequency, enable_seasonality)
                
                # Add metadata about the request
                result.update({
                    "request_method": "GET",
                    "timeseries_length": len(timeseries_data),
                    "seasonality_enabled": enable_seasonality
                })
                
                self._send_json_response(200, result)
                
            except json.JSONDecodeError:
                self._send_json_response(400, {"error": "Invalid JSON in 'ts' parameter"})
            except ValueError as e:
                self._send_json_response(400, {"error": str(e)})
            except Exception as e:
                logger.error(f"❌ GET forecast error: {e}")
                self._send_json_response(500, {"error": "Internal server error"})

        elif self.path.startswith('/seasonal-analysis'):
            # GET support for seasonal analysis
            try:
                from urllib.parse import urlparse, parse_qs
                parsed_url = urlparse(self.path)
                params = parse_qs(parsed_url.query)
                
                if 'ts' not in params:
                    self._send_json_response(400, {
                        "error": "Missing 'ts' parameter",
                        "usage": "/seasonal-analysis?ts=[timeseries_json]"
                    })
                    return
                
                timeseries_json = params['ts'][0]
                timeseries_data = json.loads(timeseries_json)
                
                # Process data
                if isinstance(timeseries_data[0], list):
                    data_values = [point[1] for point in timeseries_data]
                else:
                    data_values = timeseries_data
                
                if len(data_values) < 8:
                    self._send_json_response(400, {"error": "Need at least 8 data points for seasonal analysis"})
                    return
                
                # Perform analysis
                seasonality_info = self.server_instance.detect_seasonality(data_values)
                
                result = {
                    "input_length": len(data_values),
                    "seasonality_detection": seasonality_info,
                    "timestamp": datetime.now().isoformat(),
                    "request_method": "GET"
                }
                
                self._send_json_response(200, result)
                
            except Exception as e:
                logger.error(f"❌ GET seasonal analysis error: {e}")
                self._send_json_response(500, {"error": str(e)})
            
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
                frequency = data.get('frequency', 'monthly')
                enable_seasonality = data.get('seasonality', True)
                
                logger.info(f"📊 POST Forecast: {len(input_data)} points → {horizon} steps ({frequency}, seasonal: {enable_seasonality})")
                
                # Validate
                if not input_data:
                    self._send_json_response(400, {"error": "No data provided"})
                    return
                    
                if len(input_data) < 3:
                    self._send_json_response(400, {"error": "Need at least 3 data points"})
                    return
                
                # Generate forecast
                result = self.server_instance.forecast(input_data, horizon, frequency, enable_seasonality)
                result["request_method"] = "POST"
                self._send_json_response(200, result)
                
            except json.JSONDecodeError:
                self._send_json_response(400, {"error": "Invalid JSON format"})
            except ValueError as e:
                self._send_json_response(400, {"error": str(e)})
            except Exception as e:
                logger.error(f"❌ POST forecast error: {e}")
                self._send_json_response(500, {"error": "Internal server error"})

        elif self.path == '/seasonal-analysis':
            try:
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                data = json.loads(post_data.decode('utf-8'))
                
                input_data = data.get('data', [])
                
                if len(input_data) < 8:
                    self._send_json_response(400, {"error": "Need at least 8 data points for seasonal analysis"})
                    return
                
                # Perform seasonal analysis
                seasonality_info = self.server_instance.detect_seasonality(input_data)
                
                result = {
                    "input_length": len(input_data),
                    "seasonality_detection": seasonality_info,
                    "timestamp": datetime.now().isoformat(),
                    "request_method": "POST"
                }
                
                self._send_json_response(200, result)
                
            except Exception as e:
                logger.error(f"❌ POST seasonal analysis error: {e}")
                self._send_json_response(500, {"error": str(e)})
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
    logger.info("🚀 Starting TimesFM Seasonal Railway Server (Pure Python)")
    
    # Initialize server
    server = TimesFMRailwayServer()
    
    # Get port
    port = int(os.environ.get('PORT', 8080))
    
    # Start HTTP server
    handler = make_handler(server)
    httpd = HTTPServer(('0.0.0.0', port), handler)
    
    logger.info(f"✅ Server ready on port {port}")
    logger.info("🔗 Endpoints: /health, /forecast, /seasonal-analysis, /status")
    logger.info("🌊 Features: Trend + Seasonality detection + Business insights")
    logger.info("📡 Support: GET/POST with URL parameters")
    logger.info("🪶 Pure Python - zero external dependencies")
    logger.info("💾 Ultra-lightweight (~30MB)")
    logger.info("🎯 Railway healthchecks enabled!")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("🛑 Server stopped")
        httpd.shutdown()

if __name__ == "__main__":
    main()
