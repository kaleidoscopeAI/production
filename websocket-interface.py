from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
from typing import List, Set, Dict
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.system_status = {
            'processing': False,
            'last_update': None,
            'active_nodes': 0
        }
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        await self.broadcast_status()
    
    async def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        await self.broadcast_status()
    
    async def broadcast_status(self):
        self.system_status['last_update'] = datetime.now().isoformat()
        message = json.dumps({
            'type': 'status_update',
            'data': self.system_status
        })
        await self.broadcast(message)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Failed to send message: {e}")
    
    def update_status(self, status_update: Dict):
        self.system_status.update(status_update)

app = FastAPI()
manager = ConnectionManager()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message['type'] == 'query':
                response = await system.chat_response(message['data'])
                await websocket.send_text(json.dumps({
                    'type': 'response',
                    'data': response
                }))
            
            elif message['type'] == 'process_data':
                manager.update_status({'processing': True})
                await manager.broadcast_status()
                
                try:
                    results = await system.process_input(message['data'])
                    await websocket.send_text(json.dumps({
                        'type': 'process_complete',
                        'data': results
                    }))
                except Exception as e:
                    await websocket.send_text(json.dumps({
                        'type': 'error',
                        'data': str(e)
                    }))
                finally:
                    manager.update_status({'processing': False})
                    await manager.broadcast_status()
    
    except WebSocketDisconnect:
        await manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await manager.disconnect(websocket)

@app.get("/status")
async def get_status():
    return manager.system_status

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_path = f"uploads/{file.filename}"
    try:
        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)
        
        return {"filename": file.filename, "status": "success"}
    except Exception as e:
        return {"filename": file.filename, "status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
