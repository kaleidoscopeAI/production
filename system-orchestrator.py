import asyncio
import torch
import numpy as np
from typing import Dict, List, Set
from dataclasses import dataclass
import networkx as nx
from datetime import datetime, timedelta
import logging
import boto3
from scipy.optimize import linear_sum_assignment
from heapq import heappush, heappop

@dataclass
class Task:
    id: str
    priority: float
    dependencies: Set[str]
    resource_requirements: Dict[str, float]
    deadline: datetime
    status: str = 'pending'
    assigned_to: str = None
    progress: float = 0.0

class SystemOrchestrator:
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.membrane = None
        self.nodes = {}
        self.engines = {}
        self.supernodes = {}
        self.cube_router = None
        self.chatbot = None
        self.task_graph = nx.DiGraph()
        self.resource_monitor = ResourceMonitor()
        self.task_queue = []
        self.sqs = boto3.client('sqs')
        self.dynamodb = boto3.resource('dynamodb')
        self.logger = logging.getLogger("Orchestrator")
        self.performance_metrics = {}

    async def initialize_system(self):
        await self._initialize_components()
        await self._establish_connections()
        await self._start_monitoring()
        self.logger.info("System initialization complete")

    async def _initialize_components(self):
        components = {
            'membrane': Membrane,
            'kaleidoscope_engine': KaleidoscopeEngine,
            'mirror_engine': MirrorEngine,
            'cube_router': HypercubeRouter,
            'chatbot': ChatBot
        }

        for name, component_class in components.items():
            try:
                component = component_class(self.config[name])
                await component.initialize()
                setattr(self, name, component)
            except Exception as e:
                self.logger.error(f"Failed to initialize {name}: {e}")
                raise

    async def schedule_task(self, task: Task):
        self._validate_task(task)
        dependencies_met = await self._check_dependencies(task)
        
        if not dependencies_met:
            heappush(self.task_queue, (task.priority, task))
            return

        optimal_component = await self._find_optimal_component(task)
        if optimal_component:
            await self._assign_task(task, optimal_component)
        else:
            heappush(self.task_queue, (task.priority, task))

    def _validate_task(self, task: Task):
        required_fields = {'id', 'priority', 'dependencies', 'resource_requirements', 'deadline'}
        missing_fields = required_fields - set(task.__dict__.keys())
        if missing_fields:
            raise ValueError(f"Missing required task fields: {missing_fields}")

    async def _check_dependencies(self, task: Task) -> bool:
        for dep_id in task.dependencies:
            dep_task = self.task_graph.nodes.get(dep_id)
            if not dep_task or dep_task['status'] != 'completed':
                return False
        return True

    async def _find_optimal_component(self, task: Task) -> str:
        components = self._get_available_components()
        if not components:
            return None

        scores = []
        for comp_id, component in components.items():
            score = await self._calculate_component_score(component, task)
            scores.append((score, comp_id))

        if not scores:
            return None

        return max(scores, key=lambda x: x[0])[1]

    async def _calculate_component_score(self, component: any, task: Task) -> float:
        metrics = await component.get_metrics()
        
        # Calculate base score from component health and capacity
        base_score = metrics['health_status'] * (1 - metrics['load'])
        
        # Calculate resource match score
        resource_match = sum(
            min(metrics['resources'].get(r, 0), req)
            for r, req in task.resource_requirements.items()
        ) / sum(task.resource_requirements.values())
        
        # Calculate specialization match if applicable
        spec_match = 0.0
        if hasattr(component, 'dna') and hasattr(component.dna, 'specialization'):
            spec_match = self._calculate_specialization_match(
                component.dna.specialization,
                task.resource_requirements
            )
        
        # Combine scores with weights
        return 0.4 * base_score + 0.4 * resource_match + 0.2 * spec_match

    async def _assign_task(self, task: Task, component_id: str):
        try:
            component = self._get_component(component_id)
            await component.assign_task(task)
            
            task.status = 'assigned'
            task.assigned_to = component_id
            
            self.task_graph.add_node(
                task.id,
                task=task,
                start_time=datetime.utcnow()
            )
            
            await self._update_task_status(task)
            
        except Exception as e:
            self.logger.error(f"Failed to assign task {task.id}: {e}")
            heappush(self.task_queue, (task.priority, task))

    async def monitor_system(self):
        while True:
            try:
                await self._check_component_health()
                await self._process_task_queue()
                await self._update_performance_metrics()
                await self._optimize_resource_allocation()
                await asyncio.sleep(self.config['monitoring_interval'])
            except Exception as e:
                self.logger.error(f"System monitoring error: {e}")

    async def _check_component_health(self):
        unhealthy_components = []
        for component_id, component in self._get_all_components().items():
            metrics = await component.get_metrics()
            if metrics['health_status'] < 0.5:
                unhealthy_components.append(component_id)
                await self._handle_unhealthy_component(component_id)

    async def _handle_unhealthy_component(self, component_id: str):
        component = self._get_component(component_id)
        active_tasks = self._get_component_tasks(component_id)
        
        # Attempt recovery
        recovered = await self._attempt_recovery(component)
        if not recovered:
            # Reassign tasks
            for task in active_tasks:
                task.status = 'pending'
                task.assigned_to = None
                await self.schedule_task(task)
            
            # Initialize replacement if needed
            await self._initialize_replacement(component_id)

    async def _attempt_recovery(self, component: any) -> bool:
        try:
            await component.reset()
            metrics = await component.get_metrics()
            return metrics['health_status'] >= 0.8
        except Exception as e:
            self.logger.error(f"Recovery attempt failed: {e}")
            return False

    async def _process_task_queue(self):
        while self.task_queue:
            priority, task = self.task_queue[0]
            
            if await self._check_dependencies(task):
                optimal_component = await self._find_optimal_component(task)
                if optimal_component:
                    heappop(self.task_queue)
                    await self._assign_task(task, optimal_component)
                else:
                    break
            else:
                break

    async def _update_performance_metrics(self):
        metrics = {
            'system_health': await self._calculate_system_health(),
            'resource_utilization': self.resource_monitor.get_utilization(),
            'task_completion_rate': self._calculate_completion_rate(),
            'component_performance': await self._get_component_performance(),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        self.performance_metrics = metrics
        await self._store_metrics(metrics)

    async def _calculate_system_health(self) -> float:
        component_health = []
        for component in self._get_all_components().values():
            metrics = await component.get_metrics()
            component_health.append(metrics['health_status'])
        
        return np.mean(component_health) if component_health else 0.0

    def _calculate_completion_rate(self) -> float:
        completed_tasks = len([
            task for task in self.task_graph.nodes()
            if self.task_graph.nodes[task]['task'].status == 'completed'
        ])
        total_tasks = self.task_graph.number_of_nodes()
        
        return completed_tasks / total_tasks if total_tasks > 0 else 1.0

    async def _optimize_resource_allocation(self):
        current_allocation = self.resource_monitor.get_allocation()
        optimal_allocation = await self._calculate_optimal_allocation()
        
        if self._should_rebalance(current_allocation, optimal_allocation):
            await self._rebalance_resources(optimal_allocation)

    async def _calculate_optimal_allocation(self) -> Dict:
        components = self._get_all_components()
        resources = self.resource_monitor.get_available_resources()
        
        # Create cost matrix for assignment
        cost_matrix = np.zeros((len(components), len(resources)))
        for i, component in enumerate(components.values()):
            for j, resource in enumerate(resources):
                cost_matrix[i, j] = await self._calculate_resource_cost(
                    component, resource
                )
        
        # Solve assignment problem
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Convert to allocation dict
        allocation = {}
        for i, j in zip(row_ind, col_ind):
            component_id = list(components.keys())[i]
            resource_id = list(resources.keys())[j]
            allocation[component_id] = resource_id
            
        return allocation

    def _should_rebalance(
        self,
        current: Dict,
        optimal: Dict
    ) -> bool:
        if not current:
            return True
            
        differences = sum(1 for k, v in current.items() if optimal.get(k) != v)
        return differences > len(current) * 0.2  # Rebalance if >20% different

    async def _rebalance_resources(self, new_allocation: Dict):
        for component_id, resource_id in new_allocation.items():
            component = self._get_component(component_id)
            await component.update_resources(resource_id)
            
        self.resource_monitor.update_allocation(new_allocation)

    def get_system_status(self) -> Dict:
        return {
            'health': self.performance_metrics.get('system_health', 0.0),
            'active_tasks': len(self.task_queue),
            'resource_utilization': self.performance_metrics.get('resource_utilization', {}),
            'component_status': {
                comp_id: comp.get_status()
                for comp_id, comp in self._get_all_components().items()
            },
            'timestamp': datetime.utcnow().isoformat()
        }

class ResourceMonitor:
    def __init__(self):
        self.resources = {}
        self.allocations = {}
        self.utilization_history = []
        
    def get_utilization(self) -> Dict:
        if not self.utilization_history:
            return {}
        
        recent = self.utilization_history[-10:]
        return {
            resource_id: np.mean([h[resource_id] for h in recent])
            for resource_id in self.resources
        }
        
    def update_allocation(self, new_allocation: Dict):
        self.allocations = new_allocation
        self._record_utilization()
        
    def _record_utilization(self):
        utilization = {
            r_id: self._calculate_resource_utilization(r_id)
            for r_id in self.resources
        }
        self.utilization_history.append(utilization)
        
        if len(self.utilization_history) > 100:
            self.utilization_history.pop(0)
            
    def _calculate_resource_utilization(self, resource_id: str) -> float:
        # Calculate actual resource usage
        return len([1 for v in self.allocations.values() if v == resource_id]) / len(self.allocations)
