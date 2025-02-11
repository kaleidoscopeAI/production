import numpy as np
import networkx as nx
from typing import Dict, List, Optional
import plotly.graph_objects as go
from dataclasses import dataclass
import torch
import asyncio
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import json
import time

@dataclass
class CubeState:
    dimensions: List[str]
    insights: Dict[str, torch.Tensor]
    connections: nx.Graph
    current_focus: str

class CubeVisualizer:
    def __init__(self, dimensions: List[str]):
        self.state = CubeState(
            dimensions=dimensions,
            insights={},
            connections=nx.Graph(),
            current_focus=""
        )
        self.layout = nx.spring_layout(self.state.connections, dim=3)
        
    def update_insight(self, key: str, insight: torch.Tensor):
        """Update insight in cube space"""
        self.state.insights[key] = insight
        self._update_connections(key)
        
    def _update_connections(self, new_key: str):
        """Update graph connections based on insight similarities"""
        new_insight = self.state.insights[new_key]
        
        for key, insight in self.state.insights.items():
            if key != new_key:
                similarity = torch.cosine_similarity(
                    new_insight.unsqueeze(0),
                    insight.unsqueeze(0)
                )
                if similarity > 0.7:
                    self.state.connections.add_edge(new_key, key, weight=float(similarity))
                    
        self.layout = nx.spring_layout(
            self.state.connections, 
            dim=3,
            pos=self.layout,
            iterations=50
        )
        
    def generate_visualization(self) -> Dict:
        """Generate 3D visualization data"""
        node_trace = go.Scatter3d(
            x=[pos[0] for pos in self.layout.values()],
            y=[pos[1] for pos in self.layout.values()],
            z=[pos[2] for pos in self.layout.values()],
            mode='markers',
            marker=dict(
                size=10,
                color=[len(self.state.connections[node]) for node in self.state.connections.nodes()],
                colorscale='Viridis'
            ),
            text=list(self.state.connections.nodes()),
            hoverinfo='text'
        )
        
        edge_x = []
        edge_y = []
        edge_z = []
        for edge in self.state.connections.edges():
            x0, y0, z0 = self.layout[edge[0]]
            x1, y1, z1 = self.layout[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_z.extend([z0, z1, None])
            
        edge_trace = go.Scatter3d(
            x=edge_x,
            y=edge_y,
            z=edge_z,
            mode='lines',
            line=dict(color='rgba(136, 136, 136, 0.5)', width=1),
            hoverinfo='none'
        )
        
        return {
            'data': [edge_trace, node_trace],
            'layout': go.Layout(
                showlegend=False,
                scene=dict(
                    xaxis=dict(showticklabels=False),
                    yaxis=dict(showticklabels=False),
                    zaxis=dict(showticklabels=False)
                ),
                margin=dict(l=0, r=0, t=0, b=0)
            )
        }

class SystemMonitor:
    def __init__(self):
        self.metrics = {}
        self.alerts = []
        self.performance_history = []
        
    async def update_metrics(self, component: str, metrics: Dict):
        """Update system metrics"""
        self.metrics[component] = {
            **metrics,
            'timestamp': time.time()
        }
        
        await self._check_alerts(component, metrics)
        self._update_history(component, metrics)
        
    async def _check_alerts(self, component: str, metrics: Dict):
        """Check for system alerts"""
        if 'memory_usage' in metrics and metrics['memory_usage'] > 0.9:
            self.alerts.append({
                'component': component,
                'type': 'high_memory',
                'value': metrics['memory_usage'],
                'timestamp': time.time()
            })
            
        if 'processing_time' in metrics and metrics['processing_time'] > 5.0:
            self.alerts.append({
                'component': component,
                'type': 'high_latency',
                'value': metrics['processing_time'],
                'timestamp': time.time()
            })
            
    def _update_history(self, component: str, metrics: Dict):
        """Update performance history"""
        self.performance_history.append({
            'component': component,
            'metrics': metrics,
            'timestamp': time.time()
        })
        
        # Keep last 1000 entries
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
            
    def get_system_state(self) -> Dict:
        """Get current system state"""
        return {
            'metrics': self.metrics,
            'alerts': self.alerts[-10:],  # Last 10 alerts
            'performance': {
                component: [
                    entry['metrics'] 
                    for entry in self.performance_history 
                    if entry['component'] == component
                ][-100:]  # Last 100 entries per component
                for component in set(e['component'] for e in self.performance_history)
            }
        }

# FastAPI Server for Real-time Monitoring
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

monitor = SystemMonitor()
cube = CubeVisualizer(['complexity', 'importance', 'novelty'])

@app.websocket("/ws/monitor")
async def websocket_monitor(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            system_state = monitor.get_system_state()
            await websocket.send_json(system_state)
            await asyncio.sleep(1)  # Update every second
    except Exception as e:
        print(f"WebSocket error: {e}")

@app.websocket("/ws/visualization")
async def websocket_visualization(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            viz_data = cube.generate_visualization()
            await websocket.send_json(viz_data)
            await asyncio.sleep(0.1)  # Update 10 times per second
    except Exception as e:
        print(f"WebSocket error: {e}")

@app.post("/api/metrics/{component}")
async def update_metrics(component: str, metrics: Dict):
    await monitor.update_metrics(component, metrics)
    return {"status": "success"}

@app.post("/api/insight")
async def update_insight(key: str, insight: Dict):
    tensor_insight = torch.tensor(insight['data'])
    cube.update_insight(key, tensor_insight)
    return {"status": "success"}

# React Dashboard Component
dashboard_component = """
import React, { useEffect, useState } from 'react';
import Plot from 'react-plotly.js';

export const SystemDashboard = () => {
    const [systemState, setSystemState] = useState(null);
    const [visualization, setVisualization] = useState(null);
    
    useEffect(() => {
        const monitorWs = new WebSocket('ws://localhost:8000/ws/monitor');
        const vizWs = new WebSocket('ws://localhost:8000/ws/visualization');
        
        monitorWs.onmessage = (event) => {
            setSystemState(JSON.parse(event.data));
        };
        
        vizWs.onmessage = (event) => {
            setVisualization(JSON.parse(event.data));
        };
        
        return () => {
            monitorWs.close();
            vizWs.close();
        };
    }, []);
    
    if (!systemState || !visualization) return <div>Loading...</div>;
    
    return (
        <div className="grid grid-cols-2 gap-4 p-4">
            <div className="col-span-2">
                <Plot
                    data={visualization.data}
                    layout={visualization.layout}
                    style={{ width: '100%', height: '600px' }}
                />
            </div>
            
            <div className="bg-white rounded-lg shadow p-4">
                <h2 className="text-xl font-bold mb-4">System Metrics</h2>
                {Object.entries(systemState.metrics).map(([component, metrics]) => (
                    <div key={component} className="mb-4">
                        <h3 className="font-semibold">{component}</h3>
                        <div className="grid grid-cols-2 gap-2">
                            {Object.entries(metrics).map(([key, value]) => (
                                <div key={key} className="flex justify-between">
                                    <span>{key}:</span>
                                    <span>{value}</span>
                                </div>
                            ))}
                        </div>
                    </div>
                ))}
            </div>
            
            <div className="bg-white rounded-lg shadow p-4">
                <h2 className="text-xl font-bold mb-4">Alerts</h2>
                {systemState.alerts.map((alert, i) => (
                    <div key={i} className="mb-2 p-2 bg-red-100 rounded">
                        <div className="font-semibold">{alert.type}</div>
                        <div>{alert.component}: {alert.value}</div>
                        <div className="text-sm text-gray-600">
                            {new Date(alert.timestamp * 1000).toLocaleString()}
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
};
"""

# Advanced Performance Monitoring
class PerformanceMonitor:
    def __init__(self):
        self.metrics_buffer = []
        self.anomaly_detector = IsolationForest(contamination=0.1)
        
    async def track_performance(self, component: str, start_time: float):
        duration = time.time() - start_time
        self.metrics_buffer.append({
            'component': component,
            'duration': duration,
            'timestamp': time.time()
        })
        
        if len(self.metrics_buffer) >= 100:
            await self._analyze_performance()
            
    async def _analyze_performance(self):
        durations = np.array([m['duration'] for m in self.metrics_buffer])
        anomalies = self.anomaly_detector.fit_predict(durations.reshape(-1, 1))
        
        if -1 in anomalies:  # Anomaly detected
            await monitor.update_metrics('performance', {
                'anomalies_detected': True,
                'anomaly_score': float(np.mean(anomalies == -1))
            })
        
        self.metrics_buffer = []

if __name__ == "__main__":
    import uvicorn
    perf_monitor = PerformanceMonitor()
    uvicorn.run(app, host="0.0.0.0", port=8000)