# DreamWeaver Integration Guide: Spatial Memory & Legend Agent Systems

This guide explains how to integrate the Spatial Memory Tracking System with the Legend Agent System to create a cohesive gameplay experience in the DreamWeaver game world. The integration creates a rich, spatially-aware narrative experience that tracks legend agents' journeys and generates historically significant locations based on their actions.

## System Overview

The integrated system combines:

1. **Legend Agent System**: Creates and manages special narrative agents that follow story arcs and generate foundational legends in the DreamWeaver universe.

2. **Spatial Memory System**: Tracks all events, memory fragments, and agent journeys with precise spatial coordinates, creating queryable database of the game world's history.

3. **Echo & Wisp Encounter Generator**: Creates rich, contextual encounters between legend agents and the various entities in the DreamWeaver.

4. **Visualization Component**: Renders spatial data as interactive 3D maps, heatmaps, and narrative timelines.

I've created a comprehensive integration of the Spatial Memory Tracking System with the Legend Agent System for your DreamWeaver project. This integration enables your game to track memory fragments, events, and agent journeys with precise spatial coordinates, creating a rich narrative environment where players can discover the historical footprints of legendary figures.
The integration includes several key components:
Core Integration System

IntegratedLegendSystem: The central class that combines functionality from both systems, managing legend agents while tracking their spatial journey.
SpatialMemoryDB: A specialized database using R-tree indexing to efficiently store and query all spatial data.
CompleteJourney: Tracks the paths of legend agents through the DreamWeaver, creating a historical record that can be visualized.

Echo & Wisp Encounter Enhancement

EnhancedLegendSystem: Extends the integrated system with specialized echo and wisp encounter generation.
Contextual Dialogue: Dialogue is informed by the agent's legend narrative and current objectives, creating meaningful interactions.
Narrative Theme Analysis: Extracts recurring themes and patterns from all encounters to enrich the legend's story.

Visualization Components

DreamweaverVisualizer: Creates interactive 3D visualizations of legend journeys, heatmaps, and timelines.
Legend Dashboard: Generates a complete web dashboard showing all aspects of a legend's development.
Narrative Arc Visualization: Shows how the significance of events evolves through the legend's story arc.

Integration Guide
I've also provided a comprehensive integration guide with:

Step-by-step implementation instructions
Best practices for spatial tracking and event importance
Solutions for common integration issues
Example code for using the integrated system

This integration creates a seamless experience where narrative elements are tied to spatial locations, allowing future players to discover the memories and significant events of legendary figures from the past as they explore the same locations in the DreamWeaver.






## Integration Architecture

### Core Components

```
┌─────────────────────────┐           ┌──────────────────────────┐
│                         │           │                          │
│  Legend Agent System    │◄─────────►│  Spatial Memory System   │
│                         │           │                          │
└───────────┬─────────────┘           └──────────────┬───────────┘
            │                                        │
            │                                        │
            ▼                                        ▼
┌─────────────────────────┐           ┌──────────────────────────┐
│                         │           │                          │
│  Echo Encounter System  │◄─────────►│  Visualization Component │
│                         │           │                          │
└─────────────────────────┘           └──────────────────────────┘
```

### Key Integration Points

1. **Agent Journey Tracking**: As legend agents move through the world, their journey is recorded in the spatial memory system, creating a historical path that can be visualized and analyzed.

2. **Spatial Event Recording**: All significant events (discoveries, encounters, milestones) are recorded with precise spatial coordinates and linked to both the agent and their legend narrative.

3. **Memory Fragment Spatialization**: Memory fragments created by agents are tied to specific locations, allowing future players to discover these memories by visiting the same locations.

4. **Enhanced Echo Encounters**: Echo encounters are informed by the agent's legend narrative and current objectives, creating contextual dialogue that advances the legend story.

5. **Significance Heatmaps**: The system generates heatmaps showing areas of high historical significance based on event importance and density.

## Implementation Steps

### 1. Set Up the Integrated System Class

The `IntegratedLegendSystem` class serves as the central integration point, combining functionality from both systems:

```python
from spatial_memory_db import SpatialMemoryDB
from legend_system import LegendSystem

class IntegratedLegendSystem:
    def __init__(self, memory_db, dreamweaver_graph, echo_prompts, wisp_data):
        # Base components
        self.memory_db = memory_db
        self.graph = dreamweaver_graph
        self.echo_prompts = echo_prompts
        self.wisp_data = wisp_data
        
        # Legend data
        self.legends = {}
        self.legend_agents = {}
        
        # Spatial tracking
        self.spatial_db = SpatialMemoryDB()
```

### 2. Link Agent Actions to Spatial Events

When processing agent actions, create corresponding spatial events and journey segments:

```python
def process_legend_agent_action(self, agent, decision, cycle_id):
    # Process the action and generate event data
    event_data = super().process_legend_agent_action(agent, decision, cycle_id)
    
    # Create a spatial event for this action
    if "coordinates" in event_data:
        spatial_event = SpatialEvent(
            event_id=f"event-{uuid.uuid4()}",
            event_type=event_data["type"],
            coordinates=SpatialPoint(
                x=event_data["coordinates"][0],
                y=event_data["coordinates"][1],
                z=event_data["coordinates"][2],
                realm=event_data.get("realm")
            ),
            # Add other event properties...
        )
        self.spatial_db.add_event(spatial_event)
    
    # Update agent journey if moving between nodes
    if event_data["type"] == "movement" and agent.current_node_id:
        # Create journey segment...
        self.spatial_db.add_journey_segment(segment, agent.journey_id)
    
    return event_data
```

### 3. Link Memory Fragments to Spatial Locations

When generating memory fragments, link them to the spatial event system:

```python
def generate_memory_fragment(self, agent, event, cycle_id):
    # Generate memory content
    memory_fragment = super().generate_memory_fragment(agent, event, cycle_id)
    
    # Create a spatial event for this memory
    if "coordinates" in event:
        spatial_event = SpatialEvent(
            event_id=f"event-mem-{memory_fragment['fragment_id']}",
            event_type="memory_creation",
            coordinates=SpatialPoint(...),
            memory_fragment_id=memory_fragment["fragment_id"],
            # Add other properties...
        )
        self.spatial_db.add_event(spatial_event)
    
    return memory_fragment
```

### 4. Create Journey Records for Legends

When creating a new legend, create a spatial journey record to track its path:

```python
def create_rune_origin_legend(self, cycle_id):
    # Create the legend and agent
    legend, agent = super().create_rune_origin_legend(cycle_id)
    
    # Create a spatial journey for this legend
    journey_id = self.spatial_db.create_legend_journey(legend.id, [agent.id])
    legend.journey_id = journey_id
    agent.journey_id = journey_id
    
    return legend, agent
```

### 5. Enhance Echo Encounters with Legend Context

When generating echo encounters, provide context from the legend narrative:

```python
def _generate_legend_echo_dialogue(self, agent, echo_type, legend_id):
    # Get the legend data
    legend = self.legends.get(legend_id)
    
    # Create context for the dialogue
    system_prompt = f"""
    You are generating dialogue for an encounter between {echo_type} and {agent.name}, 
    who is a legend figure in the DreamWeaver universe.
    
    LEGEND: {legend.title}
    SUMMARY: {legend.summary}
    NARRATIVE STAGE: {agent.arc_stage}
    """
    
    # Generate dialogue...
```

### 6. Generate Visualizations from Spatial Data

Use the visualization component to generate interactive views of the legend's spatial footprint:

```python
def visualize_legend(self, legend_id, output_dir):
    # Extract data for the visualizer
    visualizer = DreamweaverVisualizer(
        self.legends,
        self.spatial_db.events,
        self.spatial_db.journeys
    )
    
    # Generate dashboard
    visualizer.generate_legend_dashboard(legend_id, output_dir)
```

# DreamWeaver Integration Guide: Usage Example & Best Practices

## Complete Usage Example

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
```

## Best Practices for Integration

### 1. Consistent Spatial Tracking

Always maintain consistent spatial tracking for agents and events. Every significant action should:

- Update the agent's current coordinates
- Create appropriate spatial events
- Update journey segments

```python
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

### 2. Meaningful Event Importance

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

### 3. Link Milestones to Spatial Locations

Always store the spatial location where milestone completion occurred:

```python
def complete_milestone(self, legend_id, milestone_id, event):
    # Find the milestone
    for milestone in self.legends[legend_id].milestones:
        if milestone.id == milestone_id:
            milestone.completed = True
            
            # Record spatial coordinates
            if event and "coordinates" in event:
                milestone.coordinates = event["coordinates"]
```

### 4. Rich Contextual Dialogue

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

### 5. Optimize Spatial Queries

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

### 6. Narrative-Driven Visualizations

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

## Debugging Integration Issues

### Common Problems and Solutions

1. **Missing Spatial Data**
   - Check that coordinates are being properly updated during agent movement
   - Verify that `SpatialPoint` objects are being created correctly
   - Ensure the spatial index is being updated when new events are added

2. **Disconnected Journeys**
   - Check for gaps in journey segments, especially during teleportation or realm transitions
   - Verify that journey segments are correctly linked to agents and legends
   - Ensure start and end points of segments have the proper coordinates

3. **Poor Memory Localization**
   - Verify that memory fragments are linked to the correct spatial events
   - Check that event coordinates match the narrative descriptions
   - Ensure memory fragment IDs are properly stored in the spatial database

4. **Unnatural Echo Encounters**
   - Provide more context about the legend and narrative arc to the echo generator
   - Check that the echo generator is receiving up-to-date information about the agent's objectives
   - Ensure dialogue is appropriate for the agent's current arc stage

5. **Visualization Issues**
   - Verify that coordinate systems are consistent across all components
   - Check for missing data in journey segments or event records
   - Ensure heatmap data is being properly aggregated

## Extended Integration: Web Interface

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

By following these integration guidelines and examples, you can create a rich, spatially-aware narrative experience in the DreamWeaver game world that tracks legends across time and space, allowing players to discover t