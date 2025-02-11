
        asyncio.create_task(self._monitor_memory())

    async def _monitor_memory(self):
        """Monitor memory usage and trigger dumps if needed"""
        while True:
            try:
                current_usage = sum(x.numel() * x.element_size() for x in self.memory_buffer)
                if current_usage > self.config.memory_threshold:
                    await self._dump_to_engines()
                    self.memory_buffer.clear()
                await asyncio.sleep(1)
            except Exception as e:
                self.logger.error(f"Memory monitoring error: {e}")
                self.health_status = 'unhealthy'

    def get_metrics(self) -> Dict:
        """Get node metrics"""
        return {
            'health_status': self.health_status,
            'load': self.current_load,
            'processed_data_size': self.processed_data_size,
            'memory_usage': sum(x.numel() * x.element_size() for x in self.memory_buffer),
            'dna_generation': self.dna.generation
        }

    def get_available_memory(self) -> float:
        """Get available memory capacity"""
        return self.config.memory_threshold - sum(x.numel() * x.element_size() for x in self.memory_buffer)

    async def reset(self):
        """Reset node state"""
        self.memory_buffer.clear()
        self.current_load = 0
        self.health_status = 'healthy'
        await self.initialize()

    async def get_current_data(self) -> torch.Tensor:
        """Get current data in memory buffer"""
        return torch.stack(self.memory_buffer) if self.memory_buffer else torch.tensor([])

    async def get_excess_data(self, target_load: float) -> torch.Tensor:
        """Get excess data based on target load"""
        current_size = len(self.memory_buffer)
        target_size = int(target_load * self.config.memory_threshold)
        if current_size <= target_size:
            return torch.tensor([])
        
        excess_data = self.memory_buffer[target_size:]
        self.memory_buffer = self.memory_buffer[:target_size]
        return torch.stack(excess_data)


class TopologyProcessor:
    def __init__(self, max_dimension: int = 3):
        self.max_dimension = max_dimension
        
    def compute_persistence(self, data: torch.Tensor) -> Dict:
        """Compute persistence diagrams and Betti numbers"""
        data_np = data.cpu().numpy()
        if data_np.ndim > 2:
            data_np = data_np.reshape(-1, data_np.shape[-1])
            
        # Create Rips complex
        rips = RipsComplex(points=data_np)
        simplex_tree = rips.create_simplex_tree(max_dimension=self.max_dimension)
        
        # Compute persistence
        persistence = simplex_tree.persistence()
        diagrams = [[] for _ in range(self.max_dimension + 1)]
        
        for dim, (birth, death) in persistence:
            if death != float('inf'):
                diagrams[dim].append([birth, death])
                
        # Calculate Betti numbers
        betti_numbers = [len(diagram) for diagram in diagrams]
        
        return {
            "persistence_diagrams": np.array(diagrams, dtype=object),
            "betti_numbers": np.array(betti_numbers)
        }
