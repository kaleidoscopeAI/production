import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Activity, Zap, Database } from 'lucide-react';

const NodeMatrix = ({ selectedNode }) => {
  const [nodeData, setNodeData] = useState([]);
  const [matrixView, setMatrixView] = useState('activity'); // 'activity' | 'connections' | 'performance'

  useEffect(() => {
    // Simulate fetching node data
    const generateNodeData = () => {
      return Array.from({ length: 16 }, (_, i) => ({
        id: `node${i}`,
        activity: Math.random(),
        connections: Math.floor(Math.random() * 20) + 5,
        performance: Math.random() * 100,
        status: Math.random() > 0.9 ? 'warning' : 'normal',
        type: Math.random() > 0.7 ? 'supernode' : 'node'
      }));
    };

    setNodeData(generateNodeData());
    const interval = setInterval(() => setNodeData(generateNodeData()), 5000);
    return () => clearInterval(interval);
  }, []);

  const getNodeColor = (node) => {
    if (matrixView === 'activity') {
      return `rgba(139, 92, 246, ${node.activity})`;
    } else if (matrixView === 'connections') {
      const intensity = node.connections / 25;
      return `rgba(236, 72, 153, ${intensity})`;
    } else {
      const intensity = node.performance / 100;
      return `rgba(34, 197, 94, ${intensity})`;
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex justify-between items-center">
        <div className="flex space-x-4">
          <button
            onClick={() => setMatrixView('activity')}
            className={`px-4 py-2 rounded-lg flex items-center space-x-2 ${
              matrixView === 'activity' 
                ? 'bg-purple-500/20 text-purple-300' 
                : 'bg-black/20 text-gray-400'
            }`}
          >
            <Activity className="h-4 w-4" />
            <span>Activity</span>
          </button>
          <button
            onClick={() => setMatrixView('connections')}
            className={`px-4 py-2 rounded-lg flex items-center space-x-2 ${
              matrixView === 'connections' 
                ? 'bg-pink-500/20 text-pink-300' 
                : 'bg-black/20 text-gray-400'
            }`}
          >
            <Database className="h-4 w-4" />
            <span>Connections</span>
          </button>
          <button
            onClick={() => setMatrixView('performance')}
            className={`px-4 py-2 rounded-lg flex items-center space-x-2 ${
              matrixView === 'performance' 
                ? 'bg-green-500/20 text-green-300' 
                : 'bg-black/20 text-gray-400'
            }`}
          >
            <Zap className="h-4 w-4" />
            <span>Performance</span>
          </button>
        </div>
      </div>

      <div className="grid grid-cols-4 gap-4">
        {nodeData.map((node) => (
          <motion.div
            key={node.id}
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ 
              scale: node.id === selectedNode ? 1.05 : 1,
              opacity: 1 
            }}
            transition={{ duration: 0.2 }}
            className={`relative p-4 rounded-lg border ${
              node.id === selectedNode 
                ? 'border-purple-500/50' 
                : 'border-white/10'
            }`}
            style={{
              background: `linear-gradient(45deg, ${getNodeColor(node)}, transparent)`,
              backdropFilter: 'blur(12px)'
            }}
          >
            <div className="flex justify-between items-start">
              <div>
                <h4 className="text-white font-medium">{node.type === 'supernode' ? 'SuperNode' : 'Node'}</h4>
                <p className="text-sm text-gray-400">{node.id}</p>
              </div>
              {node.status === 'warning' && (
                <span className="h-2 w-2 rounded-full bg-yellow-400 animate-pulse" />
              )}
            </div>

            <div className="mt-4 space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-gray-400">Activity</span>
                <span className="text-white">{(node.activity * 100).toFixed(1)}%</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-gray-400">Connections</span>
                <span className="text-white">{node.connections}</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-gray-400">Performance</span>
                <span className="text-white">{node.performance.toFixed(1)}%</span>
              </div>
            </div>

            <motion.div
              className="absolute inset-0 rounded-lg"
              animate={{
                boxShadow: node.id === selectedNode 
                  ? '0 0 20px rgba(139, 92, 246, 0.3)' 
                  : 'none'
              }}
            />
          </motion.div>
        ))}
      </div>
    </div>
  );
};

export default NodeMatrix;