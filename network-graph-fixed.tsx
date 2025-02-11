import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';

const NetworkGraph = ({ nodes, onNodeSelect }) => {
  const svgRef = useRef(null);
  const width = 800;
  const height = 600;

  useEffect(() => {
    if (!nodes) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const simulation = d3.forceSimulation()
      .force("link", d3.forceLink().id(d => d.id).distance(50))
      .force("charge", d3.forceManyBody().strength(-100))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collision", d3.forceCollide().radius(30));

    const generateNetworkData = () => {
      const nodeData = Array.from({ length: nodes }, (_, i) => ({
        id: `node${i}`,
        type: Math.random() > 0.7 ? 'supernode' : 'node',
        activity: Math.random(),
        connections: Math.floor(Math.random() * 5) + 1
      }));

      const links = [];
      nodeData.forEach(node => {
        for (let i = 0; i < node.connections; i++) {
          const target = nodeData[Math.floor(Math.random() * nodeData.length)];
          if (target.id !== node.id) {
            links.push({
              source: node.id,
              target: target.id,
              strength: Math.random()
            });
          }
        }
      });

      return { nodes: nodeData, links };
    };

    const data = generateNetworkData();

    const link = svg.append("g")
      .selectAll("line")
      .data(data.links)
      .enter().append("line")
      .attr("stroke", d => `rgba(139, 92, 246, ${d.strength})`)
      .attr("stroke-width", d => d.strength * 2);

    const nodeElements = svg.append("g")
      .selectAll("g")
      .data(data.nodes)
      .enter().append("g");

    const circles = nodeElements
      .append("circle")
      .attr("r", d => d.type === 'supernode' ? 15 : 8)
      .attr("fill", d => d.type === 'supernode' ? '#ec4899' : '#8b5cf6')
      .attr("stroke", "#fff")
      .attr("stroke-width", 2)
      .style("cursor", "pointer")
      .on("click", (event, d) => onNodeSelect(d.id));

    // Add pulsing effect
    const pulseAnimation = () => {
      nodeElements.selectAll("circle")
        .transition()
        .duration(2000)
        .attr("r", d => d.type === 'supernode' ? 20 : 12)
        .transition()
        .duration(2000)
        .attr("r", d => d.type === 'supernode' ? 15 : 8)
        .on("end", pulseAnimation);
    };

    pulseAnimation();

    // Add node activity indicators
    nodeElements
      .append("circle")
      .attr("r", 3)
      .attr("fill", d => d.activity > 0.7 ? '#22c55e' : '#gray')
      .attr("cx", 8)
      .attr("cy", 8);

    // Add node labels
    nodeElements
      .append("text")
      .text(d => d.type === 'supernode' ? 'S' : '')
      .attr("fill", "white")
      .attr("text-anchor", "middle")
      .attr("dy", ".35em");

    // Handle simulation tick events
    simulation.nodes(data.nodes).on("tick", () => {
      link
        .attr("x1", d => d.source.x)
        .attr("y1", d => d.source.y)
        .attr("x2", d => d.target.x)
        .attr("y2", d => d.target.y);

      nodeElements
        .attr("transform", d => `translate(${d.x},${d.y})`);
    });

    simulation.force("link").links(data.links);

    // Add zoom behavior
    const zoom = d3.zoom()
      .scaleExtent([0.5, 2])
      .on("zoom", (event) => {
        svg.selectAll("g").attr("transform", event.transform);
      });

    svg.call(zoom);

    // Cleanup
    return () => {
      simulation.stop();
    };
  }, [nodes, onNodeSelect]);

  return (
    <svg
      ref={svgRef}
      width="100%"
      height="100%"
      viewBox={`0 0 ${width} ${height}`}
      className="bg-black/20 rounded-lg"
    />
  );
};

export default NetworkGraph;