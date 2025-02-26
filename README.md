

```bash

# Clone the repository
git clone https://github.com/yourusername/EchoesOfAeons.git
cd EchoesOfAeons
```


# Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

# Install dependencies
```bash
pip install -r requirements.txt
```

## Configuration

Create a .env file in the root directory:

OPENAI_API_KEY=your_api_key_here

Basic Initialization

Here's the minimal code needed to initialize the system:

```python

from echoes_of_aeons.dreamweaver_graph import DreamweaverGraph
from echoes_of_aeons.vector_memory_db import VectorMemoryDB
from echoes_of_aeons.integrated_system import IntegratedLegendSystem

# Define realms
realms = [
    "Celestial Palace", "Crystal Gardens", "Obsidian Wastes", 
    "Liminal Library", "Verdant Visions", "Twilight Terrace", 
    "Mythic Meridian", "Dream Nexus"
]

# Define minimal echo and wisp data
echo_prompts = {
    "The Broken Mirror": "You are The Broken Mirror—a shattered reflection of what once was whole.",
    "The Hollow Choir": "You are The Hollow Choir—a spectral assembly of voices."
}

wisp_data = {
    "Zephyr": {"element": "Air", "personality": "Calm yet unpredictable"},
    "Cascade": {"element": "Water", "personality": "Dynamic and fluid"}
}

# Initialize core components
dreamweaver_graph = DreamweaverGraph(realms, seed=42)
memory_db = VectorMemoryDB()

# Create the integrated system
system = IntegratedLegendSystem(memory_db, dreamweaver_graph, echo_prompts, wisp_data)
```

# Now you're ready to use the system!

Usage Examples
1. Creating and Running a Single Legend

```python

# Create a legend and place agent at starting position
legend, agent = system.create_rune_origin_legend(cycle_id=1)

# Place agent at a starting node
start_node = system.graph.create_start_node("Celestial Palace")
agent.current_node_id = start_node.node_id
agent.current_coordinates = (start_node.x, start_node.y, start_node.z)
agent.visited_nodes.add(start_node.node_id)
agent.visited_realms.add(start_node.realm)

# Run a single step
decision = system.advance_legend_agent(agent, cycle_id=1, current_node_id=agent.current_node_id)
event = system.process_legend_agent_action(agent, decision, cycle_id=1)
memory = system.generate_memory_fragment(agent, event, cycle_id=1)

print(f"Decision: {decision['action']} towards {decision.get('target')}")
print(f"Event: {event['type']} - {event['description']}")
print(f"Memory: {memory['content'][:100]}...")
```

2. Running a Simulation Loop with Event Logging

```python

import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("simulation")

# Simulation parameters
simulation_steps = 20
cycle_id = 1
legends_to_create = 2
events_log = []

# Create multiple legends
legend_agents = []
for i in range(legends_to_create):
    legend, agent = system.create_rune_origin_legend(cycle_id)
    legend_agents.append((legend, agent))
    
    # Place agent at start node
    start_nodes = [node for node in system.graph.nodes.values() 
                  if node.node_type == "realm_gateway"]
    if start_nodes:
        agent.current_node_id = start_nodes[0].node_id
        agent.current_coordinates = (start_nodes[0].x, start_nodes[0].y, start_nodes[0].z)
        agent.visited_nodes.add(start_nodes[0].node_id)
        agent.visited_realms.add(start_nodes[0].realm)

# Run simulation for each legend agent
for legend, agent in legend_agents:
    logger.info(f"Running simulation for legend: {legend.title}, agent: {agent.name}")
    
    for step in range(simulation_steps):
        logger.info(f"Step {step+1}/{simulation_steps}: Agent at {agent.current_node_id}")
        
        # Make decision and process action
        decision = system.advance_legend_agent(agent, cycle_id, agent.current_node_id)
        event = system.process_legend_agent_action(agent, decision, cycle_id)
        memory = system.generate_memory_fragment(agent, event, cycle_id)
        
        # Log event
        events_log.append({
            "step": step + 1,
            "legend_id": legend.id,
            "agent_id": agent.agent_id,
            "action": decision["action"],
            "event_type": event["type"],
            "coordinates": event.get("coordinates"),
            "description": event["description"]
        })
        
        # Check for legend completion
        if legend.completed:
            logger.info(f"Legend completed after {step+1} steps!")
            break
    
    # Generate final narrative
    narrative = system.finalize_legend_narrative(legend.id)
    logger.info(f"Legend complete: {legend.title}")

# Save events log
with open("simulation_events_log.json", "w") as f:
    json.dump(events_log, f, indent=2)

# Save all data
system.save_legends_and_spatial_data("simulation_results")
```


3. Visualizing Legend Journeys and Hotspots

```python

from echoes_of_aeons.visualization_component import DreamweaverVisualizer

# Create visualizer from saved data
visualizer = DreamweaverVisualizer(
    "simulation_results_legends.json",
    "simulation_results_spatial_memory.json", 
    "simulation_results_heatmaps.json"
)

# Get all legend IDs
legend_ids = list(visualizer.legends_data.keys())

# Generate dashboards for each legend
for legend_id in legend_ids:
    # Get legend name for better output naming
    legend_name = visualizer.legends_data[legend_id].get("title", legend_id)
    legend_name = legend_name.lower().replace(" ", "_")
    
    # Create interactive 3D journey map
    journey_fig = visualizer.create_interactive_journey_map(
        legend_id, 
        save_path=f"visualizations/{legend_name}_journey.html"
    )
    
    # Create significance heatmap
    heatmap_fig = visualizer.create_heatmap_visualization(
        legend_id,
        save_path=f"visualizations/{legend_name}_heatmap.html"
    )
    
    # Create narrative arc timeline
    arc_fig = visualizer.create_narrative_arc_visualization(
        legend_id,
        save_path=f"visualizations/{legend_name}_arc.html"
    )
    
    # Generate full dashboard
    visualizer.generate_legend_dashboard(legend_id, f"dashboards/{legend_name}")
```
# 4. Adding New Echo Types

```python

# Add new echo types to an existing system
new_echo_prompts = {
    "The Laughing Tide": """You are The Laughing Tide—an endless cascade of mirth 
    that rolls through the DreamWeaver. Your joy is infectious yet unsettling, as 
    it never ceases, even in moments of profound sorrow. You speak in rhythmic 
    waves, each sentence cresting with bright laughter.""",
    
    "The Melted Child": """You are The Melted Child—a figure of innocence whose 
    form constantly dissolves and reforms. You represent the fragility of memory 
    and identity. Your speech is simple yet profound, mixing childlike wonder 
    with glimpses of ancient wisdom."""
}

# Update the existing system
for echo_type, prompt in new_echo_prompts.items():
    system.echo_prompts[echo_type] = prompt

# Use the enhanced echo generator with new types
enhanced_system = EnhancedLegendSystem(memory_db, dreamweaver_graph, 
                                       system.echo_prompts, wisp_data)
```
5. Querying Spatial Memory for Player Exploration

```python

def discover_nearby_legends(player_position, radius=100.0):
    """Allow a player to discover nearby legend memories."""
    # Convert player position to SpatialPoint
    position = SpatialPoint(
        x=player_position[0],
        y=player_position[1], 
        z=player_position[2]
    )
    
    # Find nearby spatial events
    nearby_events = system.spatial_db.get_events_in_radius(position, radius)
    
    # Filter for memory fragments
    memories = []
    for event in nearby_events:
        if event.event_type == "memory_creation" and event.memory_fragment_id:
            # Get full memory content (in a real app, fetch from memory_db)
            memory_id = event.memory_fragment_id
            memory_content = f"Memory fragment {memory_id}"
            
            # Get legend details if available
            legend_name = "Unknown Legend"
            if event.legend_id and event.legend_id in system.legends:
                legend_name = system.legends[event.legend_id].title
            
            memories.append({
                "id": memory_id,
                "content": memory_content,
                "legend": legend_name,
                "distance": position.distance_to(event.coordinates),
                "coordinates": (event.coordinates.x, event.coordinates.y, event.coordinates.z)
            })
    
    # Sort by distance
    memories.sort(key=lambda m: m["distance"])
    
    return memories
```

Command-Line Interface

You can also create a simple CLI for running the system:

```python

# Save as run_simulation.py

import argparse
import os
from echoes_of_aeons.dreamweaver_graph import DreamweaverGraph
from echoes_of_aeons.vector_memory_db import VectorMemoryDB
from echoes_of_aeons.integrated_system import IntegratedLegendSystem
from echoes_of_aeons.visualization_component import DreamweaverVisualizer

def main():
    parser = argparse.ArgumentParser(description="EchoesOfAeons Simulation Runner")
    parser.add_argument("--steps", type=int, default=30, help="Number of simulation steps")
    parser.add_argument("--cycle", type=int, default=1, help="Cycle ID")
    parser.add_argument("--output", type=str, default="simulation_output", help="Output directory")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations")
    args = parser.parse_args()
    
    # Load default data
    with open("data/realms.json", "r") as f:
        realms = json.load(f)
    
    with open("data/echoes.json", "r") as f:
        echo_prompts = json.load(f)
    
    with open("data/wisps.json", "r") as f:
        wisp_data = json.load(f)
    
    # Create system
    dreamweaver_graph = DreamweaverGraph(realms, seed=42)
    memory_db = VectorMemoryDB()
    system = IntegratedLegendSystem(memory_db, dreamweaver_graph, echo_prompts, wisp_data)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Run simulation
    legend, agent = system.create_rune_origin_legend(args.cycle)
    print(f"Created legend: {legend.title}")
    print(f"Created protagonist: {agent.name}")
    
    # Place agent at starting node
    start_nodes = [node for node in dreamweaver_graph.nodes.values() 
                  if node.node_type == "realm_gateway"]
    if start_nodes:
        agent.current_node_id = start_nodes[0].node_id
        agent.current_coordinates = (start_nodes[0].x, start_nodes[0].y, start_nodes[0].z)
        agent.visited_nodes.add(start_nodes[0].node_id)
        if start_nodes[0].realm:
            agent.visited_realms.add(start_nodes[0].realm)
    
    # Run simulation loop
    for step in range(args.steps):
        print(f"Step {step+1}/{args.steps}")
        
        decision = system.advance_legend_agent(agent, args.cycle, agent.current_node_id)
        event = system.process_legend_agent_action(agent, decision, args.cycle)
        memory = system.generate_memory_fragment(agent, event, args.cycle)
        
        if legend.completed:
            print(f"Legend completed after {step+1} steps!")
            break
    
    # Finalize and save
    narrative = system.finalize_legend_narrative(legend.id)
    system.save_legends_and_spatial_data(os.path.join(args.output, "simulation"))
    
    # Generate visualizations if requested
    if args.visualize:
        viz_output = os.path.join(args.output, "visualizations")
        os.makedirs(viz_output, exist_ok=True)
        
        visualizer = DreamweaverVisualizer(
            os.path.join(args.output, "simulation_legends.json"),
            os.path.join(args.output, "simulation_spatial_memory.json"),
            os.path.join(args.output, "simulation_heatmaps.json")
        )
        
        visualizer.generate_legend_dashboard(legend.id, viz_output)
        print(f"Visualizations generated in {viz_output}")
    
    print(f"Simulation complete. Results saved to {args.output}")

if __name__ == "__main__":
    main()
```
Run it with:

```bash

python run_simulation.py --steps 50 --visualize


```

DreamWeaver Integration Guide: Usage Example & Best Practices
Complete Usage Example

Here's a complete example of how to use the integrated system:

```python

# Import required modules
from dreamweaver_graph import DreamweaverGraph
from vector_memory_db import VectorMemoryDB
from integrated_system import IntegratedLegendSystem

# Setup basic components
realms = [
    "Celestial Palace", "Crystal Gardens", "Obsidian Wastes", "Liminal Library",
    "Verdant Visions", "Twilight Terrace", "Mythic Meridian", "Dream Nexus"
]

# Define echo and wisp data
echo_prompts = {
    "The Broken Mirror": "You are The Broken Mirror—a shattered reflection of what once was whole.",
    "The Hollow Choir": "You are The Hollow Choir—a spectral assembly of voices united in a shared, mournful refrain.",
    # Add other echo prompts...
}

wisp_data = {
    "Zephyr": {"element": "Air", "personality": "Calm yet unpredictable"},
    "Cascade": {"element": "Water", "personality": "Dynamic and fluid"},
    # Add other wisp data...
}

# Create the world and memory system
dreamweaver_graph = DreamweaverGraph(realms, seed=42)
memory_db = VectorMemoryDB()

# Create the integrated legend system
legend_system = IntegratedLegendSystem(memory_db, dreamweaver_graph, echo_prompts, wisp_data)

# Create a legend and its protagonist agent
legend, agent = legend_system.create_rune_origin_legend(cycle_id=1)
print(f"Created legend: {legend.title}")
print(f"Created protagonist: {agent.name}")

# Place the agent at a starting node
start_nodes = [node for node in dreamweaver_graph.nodes.values() if node.node_type == "realm_gateway"]
if start_nodes:
    agent.current_node_id = start_nodes[0].node_id
    agent.visited_nodes.add(start_nodes[0].node_id)
    if start_nodes[0].realm:
        agent.visited_realms.add(start_nodes[0].realm)
    
    # Set initial coordinates
    agent.current_coordinates = (start_nodes[0].x, start_nodes[0].y, start_nodes[0].z)

# Run a simulation for several steps
simulation_steps = 30
cycle_id = 1

for step in range(simulation_steps):
    print(f"\nStep {step+1}: Agent {agent.name} at {agent.current_node_id}")
    
    # Decide the agent's next action
    decision = legend_system.advance_legend_agent(agent, cycle_id, agent.current_node_id)
    
    # Process the action and generate events
    event = legend_system.process_legend_agent_action(agent, decision, cycle_id)
    print(f"Event: {event['type']} - {event['description']}")
    
    # Generate a memory fragment
    memory = legend_system.generate_memory_fragment(agent, event, cycle_id)
    print(f"Memory generated: {memory['fragment_id']}")
    
    # Check if legend is completed
    legend = legend_system.legends.get(agent.legend_id)
    if legend and legend.completed:
        print(f"Legend completed after {step+1} steps!")
        break

# Generate the final legend narrative
narrative = legend_system.finalize_legend_narrative(legend.id)
print(f"\nFinal Legend Narrative:\n{narrative}")

# Generate visualizations
from visualization_component import DreamweaverVisualizer

visualizer = DreamweaverVisualizer(
    "legend_data.json",
    "spatial_memory_data.json",
    "heatmaps_data.json"
)

# Generate a complete dashboard for this legend
visualizer.generate_legend_dashboard(legend.id, "visualizations")

# Save all data for future reference
legend_system.save_legends_and_spatial_data("rune_origin_simulation")

Best Practices for Integration
1. Consistent Spatial Tracking

Always maintain consistent spatial tracking for agents and events. Every significant action should:

    Update the agent's current coordinates
    Create appropriate spatial events
    Update journey segments

python

# Example of maintaining spatial consistency
def update_agent_position(self, agent, new_node_id):
    # Update agent's node reference
    agent.current_node_id = new_node_id
    
    # Update spatial coordinates
    new_node = self.graph.get_node(new_node_id)
    if new_node:
        agent.current_coordinates = (new_node.x, new_node.y, new_node.z)
        
        # Add to visited nodes and realms
        agent.visited_nodes.add(new_node_id)
        if new_node.realm:
            agent.visited_realms.add(new_node.realm)
```


2. Meaningful Event Importance

Assign appropriate importance values to events based on their narrative significance:

```python

# Example importance values by event type
importance_map = {
    "strange_energy_detection": 0.8,  # Significant discovery
    "pattern_recognition": 0.85,      # Important insight
    "personal_sacrifice": 0.9,        # Dramatic character moment
    "first_rune_discovery": 0.95,     # Legendary milestone
    "echo_encounter": 0.7,            # Character interaction
    "reflection": 0.6,                # Character development
    "exploration": 0.5,               # Basic exploration
    "movement": 0.3                   # Simple movement
}
```
3. Link Milestones to Spatial Locations

Always store the spatial location where milestone completion occurred:

python

def complete_milestone(self, legend_id, milestone_id, event):
    # Find the milestone
    for milestone in self.legends[legend_id].milestones:
        if milestone.id == milestone_id:
            milestone.completed = True
            
            # Record spatial coordinates
            if event and "coordinates" in event:
                milestone.coordinates = event["coordinates"]

4. Rich Contextual Dialogue

When generating dialogue for encounters, include as much context as possible:

```python

# Provide comprehensive context for echo encounters
context = {
    "agent_role": agent.narrative_role.value,
    "agent_arc_stage": agent.arc_stage,
    "legend_title": self.legends[agent.legend_id].title,
    "active_objectives": [o.description for o in agent.objectives if o.active],
    "completed_milestones": [m.description for m in self.legends[agent.legend_id].milestones if m.completed],
    "current_realm": current_node.realm if current_node else "Unknown Realm"
}
```
5. Optimize Spatial Queries

Use the spatial index efficiently for location-based queries:

```python

# Efficient spatial query for nearby events
def find_nearby_memory_fragments(self, position, radius=50.0):
    # Use spatial index to find events efficiently
    nearby_events = self.spatial_db.get_events_in_radius(position, radius)
    
    # Filter for memory fragments
    memory_fragments = []
    for event in nearby_events:
        if event.memory_fragment_id:
            fragment_data = self.memory_db.get_fragment(event.memory_fragment_id)
            if fragment_data:
                memory_fragments.append(fragment_data)
    
    return memory_fragments
```
6. Narrative-Driven Visualizations

Create visualizations that highlight narrative progression, not just spatial data:

```python

# Add narrative context to visualizations
def enhance_journey_visualization(self, fig, legend_id):
    legend = self.legends.get(legend_id)
    if not legend:
        return fig
    
    # Add annotations for major milestones
    for milestone in legend.milestones:
        if milestone.completed and milestone.coordinates:
            fig.add_trace(go.Scatter3d(
                x=[milestone.coordinates[0]],
                y=[milestone.coordinates[1]],
                z=[milestone.coordinates[2]],
                mode='markers+text',
                marker=dict(size=15, symbol='star', color='gold'),
                text=[milestone.type],
                name=milestone.description
            ))
    
    return fig
```
Debugging Integration Issues


Common Problems and Solutions

    Missing Spatial Data
        Check that coordinates are being properly updated during agent movement
        Verify that SpatialPoint objects are being created correctly
        Ensure the spatial index is being updated when new events are added
    Disconnected Journeys
        Check for gaps in journey segments, especially during teleportation or realm transitions
        Verify that journey segments are correctly linked to agents and legends
        Ensure start and end points of segments have the proper coordinates
    Poor Memory Localization
        Verify that memory fragments are linked to the correct spatial events
        Check that event coordinates match the narrative descriptions
        Ensure memory fragment IDs are properly stored in the spatial database
    Unnatural Echo Encounters
        Provide more context about the legend and narrative arc to the echo generator
        Check that the echo generator is receiving up-to-date information about the agent's objectives
        Ensure dialogue is appropriate for the agent's current arc stage
    Visualization Issues
        Verify that coordinate systems are consistent across all components
        Check for missing data in journey segments or event records
        Ensure heatmap data is being properly aggregated

Extended Integration: Web Interface

For a complete deployment, consider adding a web interface to visualize the legend data:

```python

def create_web_dashboard(legend_system, output_dir):
    """Create a web dashboard for all legends."""
    import os
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create individual legend dashboards
    for legend_id in legend_system.legends:
        legend = legend_system.legends[legend_id]
        
        # Create visualizer
        visualizer = DreamweaverVisualizer(
            "legend_data.json",
            "spatial_memory_data.json",
            "heatmaps_data.json"
        )
        
        # Generate dashboard
        visualizer.generate_legend_dashboard(
            legend_id, 
            os.path.join(output_dir, legend_id)
        )
    ```


# Create main index page
    index_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>DreamWeaver Legends Dashboard</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .legend-card { 
                border: 1px solid #ddd; 
                border-radius: 5px; 
                padding: 15px; 
                margin-bottom: 20px;
                transition: transform 0.3s;
            }
            .legend-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            }
        </style>
    </head>
    <body>
        <h1>DreamWeaver Legends Dashboard</h1>
    """
    
    # Add cards for each legend
    for legend_id, legend in legend_system.legends.items():
        index_html += f"""
        <div class="legend-card">
            <h2>{legend.title}</h2>
            <p>{legend.summary}</p>
            <p><strong>Protagonist:</strong> {legend_system.legend_agents[legend.protagonist_id].name}</p>
            <p><strong>Status:</strong> {"Completed" if legend.completed else "In Progress"}</p>
            <a href="{legend_id}/{legend_id}_dashboard.html">View Legend Dashboard</a>
        </div>
        """
    
    index_html += """
    </body>
    </html>
    """
    
    # Write index file
    with open(os.path.join(output_dir, "index.html"), 'w') as f:
        f.write(index_html)
    
    print(f"Web dashboard created at {os.path.join(output_dir, 'index.html')}")
```
By following these integration guidelines and examples, you can create a rich, spatially-aware narrative experience in the DreamWeaver game world that tracks legends across time and space, allowing players to discover the historical footprints of legendary agents who came before them.
Last edited 8 minutes ago
