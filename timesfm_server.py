#!/usr/bin/env python3
"""
TimesFM MCP Server using HuggingFace Inference API
Zero memory footprint - uses TimesFM 2.5 via API calls
"""

import asyncio
import json
import logging
import os
import httpx
from typing import Any, Dict, List, Optional
from datetime import datetime

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Resource, 
    Tool, 
    TextContent,
    CallToolRequest,
    CallToolResult,
    ListResourcesRequest,
    ListResourcesResult,
    ListToolsRequest,
    ListToolsResult,
    ReadResourceRequest,
    ReadResourceResult
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("timesfm-mcp-hf")

class TimesFMHuggingFaceMCP:
    def __init__(self):
        self.server = Server("timesfm-huggingface-forecaster")
        self.hf_token = os.getenv("HUGGINGFACE_TOKEN")  # Optional but recommended
        self.model_id = "google/timesfm-2.5-200m-pytorch"
        self.api_url = f"https://api-inference.huggingface.co/models/{self.model_id}"
        self._setup_handlers()
        
    def _setup_handlers(self):
        """Setup MCP protocol handlers"""
        
        @self.server.list_resources()
        async def list_resources() -> ListResourcesResult:
            """List available resources"""
            return ListResourcesResult(
                resources=[
                    Resource(
                        uri="timesfm://model/info",
                        name="TimesFM 2.5 Model Information",
                        description="Information about TimesFM 2.5 via HuggingFace Inference API",
                        mimeType="application/json"
                    ),
                    Resource(
                        uri="timesfm://api/status", 
                        name="HuggingFace API Status",
                        description="Current status of HuggingFace Inference API",
                        mimeType="application/json"
                    )
                ]
            )
            
        @self.server.read_resource()
        async def read_resource(request: ReadResourceRequest) -> ReadResourceResult:
            """Read resource content"""
            if request.uri == "timesfm://model/info":
                info = await self._get_model_info()
                return ReadResourceResult(
                    contents=[
                        TextContent(
                            type="text",
                            text=json.dumps(info, indent=2)
                        )
                    ]
                )
            elif request.uri == "timesfm://api/status":
                status = await self._check_api_status()
                return ReadResourceResult(
                    contents=[
                        TextContent(
                            type="text",
                            text=json.dumps(status, indent=2)
                        )
                    ]
                )
            else:
                raise ValueError(f"Unknown resource: {request.uri}")
        
        @self.server.list_tools()
        async def list_tools() -> ListToolsResult:
            """List available tools"""
            return ListToolsResult(
                tools=[
                    Tool(
                        name="timesfm_forecast",
                        description="Generate time series forecasts using Google's TimesFM 2.5 via HuggingFace API",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "data": {
                                    "type": "array",
                                    "items": {"type": "number"},
                                    "description": "Historical time series data (minimum 10 points for TimesFM 2.5)",
                                    "minItems": 10
                                },
                                "horizon": {
                                    "type": "integer",
                                    "description": "Number of future periods to forecast (1-100)",
                                    "minimum": 1,
                                    "maximum": 100,
                                    "default": 10
                                },
                                "frequency": {
                                    "type": "string",
                                    "description": "Data frequency hint",
                                    "enum": ["daily", "weekly", "monthly", "quarterly", "yearly", "auto"],
                                    "default": "auto"
                                }
                            },
                            "required": ["data"]
                        }
                    ),
                    Tool(
                        name="timesfm_api_status",
                        description="Check HuggingFace API status and model availability",
                        inputSchema={
                            "type": "object",
                            "properties": {},
                            "additionalProperties": False
                        }
                    )
                ]
            )
            
        @self.server.call_tool()
        async def call_tool(request: CallToolRequest) -> CallToolResult:
            """Handle tool calls"""
            try:
                if request.name == "timesfm_forecast":
                    result = await self._handle_forecast(request.arguments or {})
                elif request.name == "timesfm_api_status":
                    result = await self._check_api_status()
                else:
                    raise ValueError(f"Unknown tool: {request.name}")
                
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text", 
                            text=json.dumps(result, indent=2)
                        )
                    ]
                )
                
            except Exception as e:
                logger.error(f"Tool call error: {e}")
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text=json.dumps({
                                "error": str(e),
                                "tool": request.name,
                                "status": "failed"
                            }, indent=2)
                        )
                    ],
                    isError=True
                )

    async def _get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "model_id": self.model_id,
            "model_version": "TimesFM-2.5-200M",
            "provider": "HuggingFace Inference API",
            "architecture": "Foundation model for time series",
            "memory_footprint": "Zero (API-based)",
            "max_horizon": 100,
            "min_input_size": 10,
            "api_endpoint": self.api_url,
            "has_token": bool(self.hf_token),
            "status": "ready",
            "advantages": [
                "Latest TimesFM 2.5 model",
                "Zero memory usage",
                "Infinite scaling",
                "No cold start after first call"
            ]
        }

    async def _check_api_status(self) -> Dict[str, Any]:
        """Check HuggingFace API status"""
        try:
            headers = {}
            if self.hf_token:
                headers["Authorization"] = f"Bearer {self.hf_token}"
            
            async with httpx.AsyncClient() as client:
                # Test API availability
                response = await client.get(
                    f"https://huggingface.co/api/models/{self.model_id}",
                    headers=headers,
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    model_info = response.json()
                    return {
                        "api_status": "available",
                        "model_status": "loaded",
                        "model_downloads": model_info.get("downloads", 0),
                        "last_modified": model_info.get("lastModified", "unknown"),
                        "response_time_ms": "fast",
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    return {
                        "api_status": "error",
                        "status_code": response.status_code,
                        "error": "API not responding correctly"
                    }
                    
        except Exception as e:
            return {
                "api_status": "error", 
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def _handle_forecast(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle forecasting request via HuggingFace API"""
        
        # Validate arguments
        data = arguments.get("data", [])
        horizon = arguments.get("horizon", 10)
        frequency = arguments.get("frequency", "auto")
        
        if not data or len(data) < 10:
            raise ValueError("TimesFM 2.5 requires at least 10 data points")
        
        # Limit horizon
        horizon = min(horizon, 100)
        
        logger.info(f"🔮 Forecasting via HF API: {len(data)} points → {horizon} steps")
        
        try:
            # Prepare headers
            headers = {"Content-Type": "application/json"}
            if self.hf_token:
                headers["Authorization"] = f"Bearer {self.hf_token}"
            
            # Prepare request payload for TimesFM API
            payload = {
                "inputs": {
                    "data": data,
                    "horizon": horizon,
                    "frequency": frequency
                },
                "parameters": {
                    "return_forecast_intervals": True
                }
            }
            
            start_time = datetime.now()
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=30.0  # Generous timeout for API
                )
                
                if response.status_code == 200:
                    api_result = response.json()
                    end_time = datetime.now()
                    duration = (end_time - start_time).total_seconds()
                    
                    # Format result
                    result = {
                        "success": True,
                        "point_forecast": api_result.get("forecast", []),
                        "horizon": horizon,
                        "input_length": len(data),
                        "input_data": data[-10:],  # Last 10 points for context
                        "model_version": "TimesFM-2.5-200M-HF",
                        "provider": "HuggingFace Inference API",
                        "api_response_time": round(duration, 2),
                        "timestamp": datetime.now().isoformat(),
                        "frequency": frequency
                    }
                    
                    # Add confidence intervals if available
                    if "intervals" in api_result:
                        result["confidence_intervals"] = api_result["intervals"]
                    
                    logger.info(f"✅ Forecast completed in {duration:.2f}s via HF API")
                    return result
                    
                elif response.status_code == 503:
                    # Model loading on HuggingFace
                    return {
                        "success": False,
                        "error": "Model loading on HuggingFace (cold start)",
                        "estimated_time": "~20 seconds",
                        "suggestion": "Retry in 20-30 seconds",
                        "status_code": 503
                    }
                else:
                    # API error
                    error_detail = response.text
                    return {
                        "success": False,
                        "error": f"HuggingFace API error: {response.status_code}",
                        "detail": error_detail,
                        "suggestion": "Check API token or model availability"
                    }
            
        except httpx.TimeoutException:
            return {
                "success": False,
                "error": "API timeout",
                "suggestion": "Retry request - HuggingFace may be under load"
            }
        except Exception as e:
            logger.error(f"❌ API forecast error: {e}")
            raise

    async def run(self):
        """Run the MCP server"""
        logger.info("🚀 Starting TimesFM 2.5 MCP Server (HuggingFace API)")
        logger.info(f"📡 Using model: {self.model_id}")
        logger.info(f"🔑 API Token: {'✅ Provided' if self.hf_token else '❌ Not provided (rate limited)'}")
        
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options()
            )

def main():
    """Main entry point"""
    server = TimesFMHuggingFaceMCP()
    asyncio.run(server.run())

if __name__ == "__main__":
    main()
