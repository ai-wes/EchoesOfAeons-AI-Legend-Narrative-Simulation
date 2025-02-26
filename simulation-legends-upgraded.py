#!/usr/bin/env python3
"""
DreamWeaver Integrated Spatial Memory & Legend Agent System

This module integrates the Spatial Memory Tracking System with the Legend Agent System,
allowing for comprehensive tracking of spatial memory fragments, event histories,
and legend agent journeys throughout the DreamWeaver universe.

Features:
    - Legend Agent creation with narrative objectives
    - Spatial indexing of all memory fragments and events
    - Journey path tracking for legend agents
    - Dynamic heatmap generation of historically significant areas
    - Story arc progression and key moment recognition
    - Legend artifact generation and preservation
    - Cross-cycle narrative influence
"""

import logging
import random
import uuid
import json
import datetime
import os
import math
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Set, Tuple, Union
from enum import Enum
import numpy as np
import rtree  # Spatial indexing

from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("dreamweaver_integrated.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("integrated_system")

# ======================================================
# Legend Data Models
# ======================================================

class NarrativeRole(Enum):
    """Archetypal roles within a narrative structure."""
    PROTAGONIST = "protagonist"
    MENTOR = "mentor"
    ALLY = "ally"
    GUARDIAN = "guardian"
    MESSENGER = "messenger"
    SHAPESHIFTER = "shapeshifter"
    SHADOW = "shadow"
    TRICKSTER = "trickster"
    HERALD = "herald"
    
class LegendTone(Enum):
    """The emotional and narrative tone of a legend."""
    HEROIC = "heroic"
    TRAGIC = "tragic"
    MYSTERIOUS = "mysterious"
    CAUTIONARY = "cautionary"
    REVELATORY = "revelatory"
    PROPHETIC = "prophetic"
    TRANSFORMATIVE = "transformative"
    CREATION = "creation"

class MilestoneType(Enum):
    """Types of narrative milestones in a legend."""
    DISCOVERY = "discovery"
    CONFRONTATION = "confrontation" 
    REVELATION = "revelation"
    TRANSFORMATION = "transformation"
    SACRIFICE = "sacrifice"
    TRIUMPH = "triumph"
    FALL = "fall"
    REBIRTH = "rebirth"
    LEGACY = "legacy"

@dataclass
class NarrativeMilestone:
    """A key plot point or objective in a legend's story arc."""
    id: str
    type: MilestoneType
    description: str
    completed: bool = False
    completion_cycle: Optional[int] = None
    completion_event: Optional[Dict[str, Any]] = None
    coordinates: Optional[Tuple[float, float, float]] = None  # Spatial tracking of milestone completion
    
@dataclass
class LegendArtifact:
    """A special item or relic created during a legend's formation."""
    id: str
    name: str
    description: str
    powers: List[str]
    location: Optional[Tuple[float, float, float]] = None
    realm: Optional[str] = None
    creation_cycle: int = 0
    discovery_requirements: Dict[str, Any] = field(default_factory=dict)
    creator_id: Optional[str] = None
    
@dataclass 
class LegendData:
    """Data structure containing the complete information for a legend."""
    id: str
    title: str
    summary: str
    cycles_active: List[int]
    protagonist_id: str
    supporting_agents: List[str]
    antagonist_id: Optional[str]
    milestones: List[NarrativeMilestone]
    artifacts: List[LegendArtifact]
    memory_fragments: List[str]  # IDs of related memory fragments
    tone: LegendTone
    keywords: List[str]
    full_narrative: Optional[str] = None
    completed: bool = False
    significance: float = 0.5  # 0-1 scale of legend's impact on DreamWeaver
    journey_id: Optional[str] = None  # Link to the spatial journey record

@dataclass
class LegendObjective:
    """A specific objective for a Legend Agent to pursue."""
    id: str
    description: str
    triggers: List[Dict[str, Any]]  # Conditions that activate this objective
    resolution_conditions: List[Dict[str, Any]]  # Conditions that resolve this objective
    related_milestone_id: Optional[str] = None
    priority: int = 1  # Higher number = higher priority
    active: bool = False
    completed: bool = False

@dataclass
class LegendAgent:
    """
    An agent with a special narrative purpose who follows a legend's
    story arc and creates historically significant events.
    """
    agent_id: str
    name: str
    legend_id: str
    narrative_role: NarrativeRole
    personality: Dict[str, float]
    background: str
    special_abilities: List[str]
    knowledge: Dict[str, int]
    motivations: List[str]
    weaknesses: List[str]
    inventory: List[str]
    relationships: Dict[str, float]  # Agent ID -> Relationship value (-1.0 to 1.0)
    objectives: List[LegendObjective]
    current_state: Dict[str, Any] = field(default_factory=dict)
    
    # Navigation and tracking
    current_node_id: Optional[str] = None
    visited_nodes: Set[str] = field(default_factory=set)
    visited_realms: Set[str] = field(default_factory=set)
    discovered_echoes: List[str] = field(default_factory=list)
    bonded_wisps: List[str] = field(default_factory=list)
    
    # Spatial tracking
    current_coordinates: Optional[Tuple[float, float, float]] = None
    journey_id: Optional[str] = None  # Link to the agent's journey record
    
    # Narrative progression tracking
    arc_stage: str = "introduction"  # introduction, rising_action, climax, falling_action, resolution
    significance_events: List[Dict[str, Any]] = field(default_factory=list)
    key_decisions: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, handling sets properly."""
        agent_dict = asdict(self)
        agent_dict["visited_nodes"] = list(self.visited_nodes)
        agent_dict["visited_realms"] = list(self.visited_realms)
        return agent_dict

# ======================================================
# Spatial Data Models
# ======================================================

@dataclass
class SpatialPoint:
    """A specific location in the DreamWeaver Graph."""
    x: float
    y: float
    z: float
    realm: Optional[str] = None
    node_id: Optional[str] = None
    
    def distance_to(self, other: 'SpatialPoint') -> float:
        """Calculate Euclidean distance to another point."""
        return math.sqrt(
            (self.x - other.x) ** 2 +
            (self.y - other.y) ** 2 +
            (self.z - other.z) ** 2
        )
    
    def to_tuple(self) -> Tuple[float, float, float]:
        """Convert to coordinate tuple."""
        return (self.x, self.y, self.z)

@dataclass
class SpatialEvent:
    """An event that occurred at a specific location."""
    event_id: str
    event_type: str
    coordinates: SpatialPoint
    timestamp: str
    cycle_id: int
    agent_id: str
    description: str
    importance: float = 0.5
    related_events: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    legend_id: Optional[str] = None
    memory_fragment_id: Optional[str] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        # Convert SpatialPoint to tuple for serialization
        result["coordinates"] = self.coordinates.to_tuple()
        return result

@dataclass
class JourneySegment:
    """A segment of an agent's journey through the DreamWeaver."""
    segment_id: str
    agent_id: str
    start_point: SpatialPoint
    end_point: SpatialPoint
    start_time: str
    end_time: str
    cycle_id: int
    events: List[str] = field(default_factory=list)  # Event IDs that occurred during this segment
    nodes_visited: List[str] = field(default_factory=list)
    distance: float = 0.0
    significance: float = 0.0  # Calculated based on events that occurred

@dataclass
class CompleteJourney:
    """The complete journey of an agent or lineage through the DreamWeaver."""
    journey_id: str
    agent_ids: List[str]  # Can include multiple agents if part of a lineage
    legend_id: Optional[str]
    cycles: List[int]
    segments: List[JourneySegment] = field(default_factory=list)
    total_distance: float = 0.0
    significant_points: List[SpatialPoint] = field(default_factory=list)
    
    def add_segment(self, segment: JourneySegment):
        """Add a segment to the journey and update totals."""
        self.segments.append(segment)
        self.total_distance += segment.distance
        
        # If the agent ID isn't already tracked, add it
        if segment.agent_id not in self.agent_ids:
            self.agent_ids.append(segment.agent_id)
        
        # If the cycle isn't already tracked, add it
        if segment.cycle_id not in self.cycles:
            self.cycles.append(segment.cycle_id)

# ======================================================
# Spatial Memory Database
# ======================================================

class SpatialMemoryDB:
    """
    A database for tracking spatial memory fragments, events, and journeys
    throughout the DreamWeaver with advanced indexing and query capabilities.
    """
    
    def __init__(self):
        # Event storage
        self.events: Dict[str, SpatialEvent] = {}
        
        # Journey storage
        self.journeys: Dict[str, CompleteJourney] = {}
        self.agent_journeys: Dict[str, str] = {}  # Agent ID -> Journey ID
        self.lineage_journeys: Dict[str, str] = {}  # Lineage ID -> Journey ID
        self.legend_journeys: Dict[str, str] = {}  # Legend ID -> Journey ID
        
        # Memory fragment ID mapping
        self.memory_to_event: Dict[str, str] = {}  # Memory Fragment ID -> Event ID
        
        # Initialize spatial index (R-tree)
        p = rtree.index.Property()
        p.dimension = 3  # 3D coordinates
        self.spatial_index = rtree.index.Index(properties=p)
        
        # Current index counter for R-tree
        self.index_counter = 0
    
    def add_event(self, event: SpatialEvent) -> str:
        """
        Add an event to the database and spatial index.
        Returns the event ID.
        """
        # Store the event
        self.events[event.event_id] = event
        
        # Add to spatial index
        coords = event.coordinates
        self.spatial_index.insert(
            self.index_counter,
            (coords.x, coords.y, coords.z, coords.x, coords.y, coords.z),
            obj=event.event_id
        )
        
        # Associate memory fragment if provided
        if event.memory_fragment_id:
            self.memory_to_event[event.memory_fragment_id] = event.event_id
        
        # Update counter
        self.index_counter += 1
        
        logger.info(f"Added event {event.event_id} at coordinates {coords.to_tuple()}")
        return event.event_id
    
    def add_journey_segment(self, segment: JourneySegment, journey_id: Optional[str] = None) -> str:
        """
        Add a journey segment to the database.
        Optionally specify an existing journey to add to, or create a new one.
        Returns the journey ID.
        """
        # Calculate distance if not already set
        if segment.distance == 0.0:
            segment.distance = segment.start_point.distance_to(segment.end_point)
        
        # If no journey specified, check if agent already has one
        if not journey_id and segment.agent_id in self.agent_journeys:
            journey_id = self.agent_journeys[segment.agent_id]
        
        # If no existing journey, create a new one
        if not journey_id:
            journey_id = f"journey-{uuid.uuid4()}"
            self.journeys[journey_id] = CompleteJourney(
                journey_id=journey_id,
                agent_ids=[segment.agent_id],
                legend_id=None,
                cycles=[segment.cycle_id]
            )
            self.agent_journeys[segment.agent_id] = journey_id
        
        # Add segment to journey
        journey = self.journeys[journey_id]
        journey.add_segment(segment)
        
        # Calculate segment significance based on events
        if segment.events:
            significance_sum = sum(
                self.events[event_id].importance
                for event_id in segment.events
                if event_id in self.events
            )
            segment.significance = significance_sum / len(segment.events)
            
            # If segment is significant, add the end point to significant points
            if segment.significance > 0.7:
                journey.significant_points.append(segment.end_point)
        
        logger.info(f"Added journey segment to journey {journey_id}")
        return journey_id
    
    def create_legend_journey(self, legend_id: str, agent_ids: List[str]) -> str:
        """
        Create a journey specifically for a legend, tracking all related agents.
        Returns the journey ID.
        """
        journey_id = f"legend-journey-{uuid.uuid4()}"
        
        # Create a new journey for this legend
        journey = CompleteJourney(
            journey_id=journey_id,
            agent_ids=agent_ids.copy(),
            legend_id=legend_id,
            cycles=[]
        )
        
        # Store the journey
        self.journeys[journey_id] = journey
        self.legend_journeys[legend_id] = journey_id
        
        logger.info(f"Created legend journey {journey_id} for legend {legend_id}")
        return journey_id
    
    def create_lineage_journey(self, lineage_id: str, agent_ids: List[str], legend_id: Optional[str] = None) -> str:
        """
        Create a journey that represents a lineage's combined path through history.
        Returns the journey ID.
        """
        journey_id = f"lineage-journey-{uuid.uuid4()}"
        
        # Create a new journey for this lineage
        journey = CompleteJourney(
            journey_id=journey_id,
            agent_ids=agent_ids.copy(),
            legend_id=legend_id,
            cycles=[]
        )
        
        # Add segments from each agent's journey
        for agent_id in agent_ids:
            if agent_id in self.agent_journeys:
                agent_journey_id = self.agent_journeys[agent_id]
                agent_journey = self.journeys.get(agent_journey_id)
                
                if agent_journey:
                    for segment in agent_journey.segments:
                        journey.add_segment(segment)
        
        # Store the journey
        self.journeys[journey_id] = journey
        self.lineage_journeys[lineage_id] = journey_id
        
        logger.info(f"Created lineage journey {journey_id} for lineage {lineage_id}")
        return journey_id
    
    def get_events_in_radius(self, center: Union[SpatialPoint, Tuple[float, float, float]], radius: float) -> List[SpatialEvent]:
        """
        Find all events within a radius of the given point.
        """
        if isinstance(center, tuple):
            x, y, z = center
        else:
            x, y, z = center.x, center.y, center.z
        
        # Create bounding box for query
        bbox = (
            x - radius, y - radius, z - radius,
            x + radius, y + radius, z + radius
        )
        
        # Query spatial index
        hits = list(self.spatial_index.intersection(bbox, objects=True))
        
        # Filter results by actual distance and return events
        results = []
        for hit in hits:
            event_id = hit.object
            event = self.events.get(event_id)
            
            if event:
                point = SpatialPoint(x, y, z)
                if event.coordinates.distance_to(point) <= radius:
                    results.append(event)
        
        return results
    
    def get_events_by_agent(self, agent_id: str) -> List[SpatialEvent]:
        """Get all events associated with a specific agent."""
        return [event for event in self.events.values() if event.agent_id == agent_id]
    
    def get_events_by_legend(self, legend_id: str) -> List[SpatialEvent]:
        """Get all events associated with a specific legend."""
        return [event for event in self.events.values() if event.legend_id == legend_id]
    
    def get_events_by_type(self, event_type: str) -> List[SpatialEvent]:
        """Get all events of a specific type."""
        return [event for event in self.events.values() if event.event_type == event_type]
    
    def get_events_by_cycle(self, cycle_id: int) -> List[SpatialEvent]:
        """Get all events from a specific cycle."""
        return [event for event in self.events.values() if event.cycle_id == cycle_id]
    
    def get_events_by_realm(self, realm: str) -> List[SpatialEvent]:
        """Get all events from a specific realm."""
        return [
            event for event in self.events.values() 
            if event.coordinates.realm == realm
        ]
    
    def get_journey_by_agent(self, agent_id: str) -> Optional[CompleteJourney]:
        """Get the journey associated with a specific agent."""
        journey_id = self.agent_journeys.get(agent_id)
        if journey_id:
            return self.journeys.get(journey_id)
        return None
    
    def get_journey_by_legend(self, legend_id: str) -> Optional[CompleteJourney]:
        """Get the journey associated with a specific legend."""
        journey_id = self.legend_journeys.get(legend_id)
        if journey_id:
            return self.journeys.get(journey_id)
        return None
    
    def get_journey_by_lineage(self, lineage_id: str) -> Optional[CompleteJourney]:
        """Get the journey associated with a specific lineage."""
        journey_id = self.lineage_journeys.get(lineage_id)
        if journey_id:
            return self.journeys.get(journey_id)
        return None
    
    def get_events_along_path(self, start: SpatialPoint, end: SpatialPoint, width: float = 10.0) -> List[SpatialEvent]:
        """
        Find all events that occurred along a path between start and end points,
        within a certain width of the path.
        """
        # Vector from start to end
        path_vector = np.array([end.x - start.x, end.y - start.y, end.z - start.z])
        path_length = np.linalg.norm(path_vector)
        
        if path_length == 0:
            return self.get_events_in_radius(start, width)
        
        # Normalize the path vector
        path_unit_vector = path_vector / path_length
        
        # Get all events within a bounding box that could contain the path
        min_x = min(start.x, end.x) - width
        min_y = min(start.y, end.y) - width
        min_z = min(start.z, end.z) - width
        max_x = max(start.x, end.x) + width
        max_y = max(start.y, end.y) + width
        max_z = max(start.z, end.z) + width
        
        bbox = (min_x, min_y, min_z, max_x, max_y, max_z)
        hits = list(self.spatial_index.intersection(bbox, objects=True))
        
        # Filter results by distance to the path
        results = []
        for hit in hits:
            event_id = hit.object
            event = self.events.get(event_id)
            
            if event:
                # Vector from start to event
                event_vector = np.array([
                    event.coordinates.x - start.x,
                    event.coordinates.y - start.y,
                    event.coordinates.z - start.z
                ])
                
                # Project event vector onto path
                projection = np.dot(event_vector, path_unit_vector)
                
                # Check if projection is within path length
                if 0 <= projection <= path_length:
                    # Calculate perpendicular distance to path
                    projected_point = np.array([
                        start.x + path_unit_vector[0] * projection,
                        start.y + path_unit_vector[1] * projection,
                        start.z + path_unit_vector[2] * projection
                    ])
                    
                    event_point = np.array([
                        event.coordinates.x,
                        event.coordinates.y,
                        event.coordinates.z
                    ])
                    
                    perpendicular_distance = np.linalg.norm(event_point - projected_point)
                    
                    if perpendicular_distance <= width:
                        results.append(event)
        
        return results
    
    def get_significant_locations(self, minimum_importance: float = 0.7) -> List[Tuple[SpatialPoint, float]]:
        """
        Find locations of high historical significance based on event importance.
        Returns a list of (point, significance) tuples.
        """
        # Group events by location (rounded coordinates for clustering)
        location_groups = {}
        
        for event in self.events.values():
            # Round coordinates to cluster nearby events
            rounded_coords = (
                round(event.coordinates.x / 10) * 10,
                round(event.coordinates.y / 10) * 10,
                round(event.coordinates.z / 10) * 10
            )
            
            if rounded_coords not in location_groups:
                location_groups[rounded_coords] = []
            
            location_groups[rounded_coords].append(event)
        
        # Calculate significance for each location
        significant_locations = []
        
        for coords, events in location_groups.items():
            if not events:
                continue
            
            total_importance = sum(event.importance for event in events)
            avg_importance = total_importance / len(events)
            
            if avg_importance >= minimum_importance:
                # Calculate centroid of actual event locations
                x_sum = sum(event.coordinates.x for event in events)
                y_sum = sum(event.coordinates.y for event in events)
                z_sum = sum(event.coordinates.z for event in events)
                
                avg_x = x_sum / len(events)
                avg_y = y_sum / len(events)
                avg_z = z_sum / len(events)
                
                # Use realm from most important event
                most_important = max(events, key=lambda e: e.importance)
                realm = most_important.coordinates.realm
                
                point = SpatialPoint(avg_x, avg_y, avg_z, realm)
                significant_locations.append((point, avg_importance))
        
        # Sort by significance (highest first)
        significant_locations.sort(key=lambda x: x[1], reverse=True)
        
        return significant_locations
    
    def generate_heatmap_data(self, resolution: float = 20.0) -> Dict[str, Any]:
        """
        Generate data for a heatmap visualization of historical significance.
        
        Args:
            resolution: Size of grid cells for the heatmap
            
        Returns:
            Dictionary with heatmap data suitable for visualization
        """
        # Find bounds of the data
        if not self.events:
            return {"points": [], "min": [0, 0, 0], "max": [0, 0, 0], "resolution": resolution}
        
        min_x = min(event.coordinates.x for event in self.events.values())
        min_y = min(event.coordinates.y for event in self.events.values())
        min_z = min(event.coordinates.z for event in self.events.values())
        
        max_x = max(event.coordinates.x for event in self.events.values())
        max_y = max(event.coordinates.y for event in self.events.values())
        max_z = max(event.coordinates.z for event in self.events.values())
        
        # Create grid
        grid = {}
        
        for event in self.events.values():
            # Round coordinates to grid cells
            grid_x = math.floor(event.coordinates.x / resolution) * resolution
            grid_y = math.floor(event.coordinates.y / resolution) * resolution
            grid_z = math.floor(event.coordinates.z / resolution) * resolution
            
            grid_key = (grid_x, grid_y, grid_z)
            
            if grid_key not in grid:
                grid[grid_key] = {
                    "position": grid_key,
                    "value": 0,
                    "event_count": 0,
                    "realms": set(),
                    "legend_ids": set()
                }
            
            # Add event importance to grid cell
            grid[grid_key]["value"] += event.importance
            grid[grid_key]["event_count"] += 1
            
            if event.coordinates.realm:
                grid[grid_key]["realms"].add(event.coordinates.realm)
                
            if event.legend_id:
                grid[grid_key]["legend_ids"].add(event.legend_id)
        
        # Normalize values and convert to list
        points = []
        
        for cell in grid.values():
            # Average importance in this cell
            cell["value"] /= cell["event_count"]
            
            # Convert sets to lists for serialization
            cell["realms"] = list(cell["realms"])
            cell["legend_ids"] = list(cell["legend_ids"])
            
            points.append({
                "position": cell["position"],
                "value": cell["value"],
                "event_count": cell["event_count"],
                "realms": cell["realms"],
                "legend_ids": cell["legend_ids"]
            })
        
        return {
            "points": points,
            "min": [min_x, min_y, min_z],
            "max": [max_x, max_y, max_z],
            "resolution": resolution
        }
    
    def generate_legend_heatmap(self, legend_id: str, resolution: float = 20.0) -> Dict[str, Any]:
        """
        Generate a heatmap specific to a legend's events and significant locations.
        
        Args:
            legend_id: ID of the legend to generate heatmap for
            resolution: Size of grid cells for the heatmap
            
        Returns:
            Dictionary with heatmap data specific to the legend
        """
        legend_events = self.get_events_by_legend(legend_id)
        
        if not legend_events:
            return {"points": [], "min": [0, 0, 0], "max": [0, 0, 0], "resolution": resolution}
        
        # Find bounds of the data
        min_x = min(event.coordinates.x for event in legend_events)
        min_y = min(event.coordinates.y for event in legend_events)
        min_z = min(event.coordinates.z for event in legend_events)
        
        max_x = max(event.coordinates.x for event in legend_events)
        max_y = max(event.coordinates.y for event in legend_events)
        max_z = max(event.coordinates.z for event in legend_events)
        
        # Create grid
        grid = {}
        
        for event in legend_events:
            # Round coordinates to grid cells
            grid_x = math.floor(event.coordinates.x / resolution) * resolution
            grid_y = math.floor(event.coordinates.y / resolution) * resolution
            grid_z = math.floor(event.coordinates.z / resolution) * resolution
            
            grid_key = (grid_x, grid_y, grid_z)
            
            if grid_key not in grid:
                grid[grid_key] = {
                    "position": grid_key,
                    "value": 0,
                    "event_count": 0,
                    "realms": set(),
                    "event_types": set()
                }
            
            # Add event importance to grid cell
            grid[grid_key]["value"] += event.importance
            grid[grid_key]["event_count"] += 1
            
            if event.coordinates.realm:
                grid[grid_key]["realms"].add(event.coordinates.realm)
            
            grid[grid_key]["event_types"].add(event.event_type)
        
        # Normalize values and convert to list
        points = []
        
        for cell in grid.values():
            # Average importance in this cell
            cell["value"] /= cell["event_count"]
            
            # Convert sets to lists for serialization
            cell["realms"] = list(cell["realms"])
            cell["event_types"] = list(cell["event_types"])
            
            points.append({
                "position": cell["position"],
                "value": cell["value"],
                "event_count": cell["event_count"],
                "realms": cell["realms"],
                "event_types": cell["event_types"]
            })
        
        return {
            "legend_id": legend_id,
            "points": points,
            "min": [min_x, min_y, min_z],
            "max": [max_x, max_y, max_z],
            "resolution": resolution
        }
    
def save_to_file(self, filename: str):
        """Save the database to a JSON file."""
        data = {
            "events": {eid: event.to_dict() for eid, event in self.events.items()},
            "journeys": {jid: asdict(journey) for jid, journey in self.journeys.items()},
            "agent_journeys": self.agent_journeys,
            "lineage_journeys": self.lineage_journeys,
            "legend_journeys": self.legend_journeys,
            "memory_to_event": self.memory_to_event
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved spatial memory database to {filename}")
    
@classmethod
def load_from_file(cls, filename: str) -> 'SpatialMemoryDB':
    """Load the database from a JSON file."""
    with open(filename, 'r') as f:
        data = json.load(f)
    
    db = cls()
    
    # Load events
    for eid, event_data in data.get("events", {}).items():
        # Convert coordinates tuple back to SpatialPoint
        coords = event_data.pop("coordinates")
        if isinstance(coords, list):
            coords = tuple(coords)
        
        event_data["coordinates"] = SpatialPoint(
            x=coords[0],
            y=coords[1],
            z=coords[2],
            realm=event_data.get("realm")
        )
        
        # Recreate the event
        event = SpatialEvent(**event_data)
        db.events[eid] = event
        
        # Add to spatial index
        db.spatial_index.insert(
            db.index_counter,
            (
                event.coordinates.x, event.coordinates.y, event.coordinates.z,
                event.coordinates.x, event.coordinates.y, event.coordinates.z
            ),
            obj=eid
        )
        db.index_counter += 1
    
    # Load journeys (more complex due to nested objects)
    # This would need proper recreation of all nested objects
    
    # Load simple mappings
    db.agent_journeys = data.get("agent_journeys", {})
    db.lineage_journeys = data.get("lineage_journeys", {})
    db.legend_journeys = data.get("legend_journeys", {})
    db.memory_to_event = data.get("memory_to_event", {})
    
    logger.info(f"Loaded spatial memory database from {filename}")
    return db

# ======================================================
# Integrated Legend & Spatial System
# ======================================================

class IntegratedLegendSystem:
    """
    Core system that integrates Legend Agent management with Spatial Memory tracking,
    creating a comprehensive system for managing narrative legends and their spatial footprint.
    """
    
    def __init__(self, memory_db, dreamweaver_graph, echo_prompts, wisp_data):
        # Base components
        self.memory_db = memory_db
        self.graph = dreamweaver_graph
        self.echo_prompts = echo_prompts
        self.wisp_data = wisp_data
        
        # Legend data
        self.legends: Dict[str, LegendData] = {}
        self.legend_agents: Dict[str, LegendAgent] = {}
        
        # Spatial tracking
        self.spatial_db = SpatialMemoryDB()
        
        # Load predefined legend templates
        self.legend_templates = self._load_legend_templates()
    
    def _load_legend_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load predefined legend templates for common myth structures."""
        try:
            with open("legend_templates.json", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            # If file doesn't exist, return default templates
            return {
                "hero_journey": {
                    "title": "The Hero's Journey",
                    "tone": LegendTone.HEROIC.value,
                    "milestones": [
                        {"type": MilestoneType.DISCOVERY.value, "description": "The hero discovers their calling"},
                        {"type": MilestoneType.CONFRONTATION.value, "description": "The hero faces their first major challenge"},
                        {"type": MilestoneType.TRANSFORMATION.value, "description": "The hero undergoes a profound change"},
                        {"type": MilestoneType.TRIUMPH.value, "description": "The hero achieves victory against the odds"},
                        {"type": MilestoneType.LEGACY.value, "description": "The hero's actions leave a lasting impact"}
                    ]
                },
                "origin_story": {
                    "title": "Origin of a Wonder",
                    "tone": LegendTone.CREATION.value,
                    "milestones": [
                        {"type": MilestoneType.DISCOVERY.value, "description": "A hidden power is discovered"},
                        {"type": MilestoneType.REVELATION.value, "description": "The true nature of the power is revealed"},
                        {"type": MilestoneType.TRANSFORMATION.value, "description": "The discoverer is changed by the power"},
                        {"type": MilestoneType.LEGACY.value, "description": "Knowledge of the power is passed down"}
                    ]
                },
                "tragic_fall": {
                    "title": "The Tragic Fall",
                    "tone": LegendTone.TRAGIC.value,
                    "milestones": [
                        {"type": MilestoneType.DISCOVERY.value, "description": "A protagonist discovers great potential"},
                        {"type": MilestoneType.REVELATION.value, "description": "Hidden dangers are revealed too late"},
                        {"type": MilestoneType.FALL.value, "description": "The protagonist succumbs to hubris or fate"},
                        {"type": MilestoneType.LEGACY.value, "description": "The cautionary tale echoes through time"}
                    ]
                }
            }
    
    def create_rune_origin_legend(self, cycle_id: int) -> Tuple[LegendData, LegendAgent]:
        """
        Create a specific legend about the first discovery of Dream Runes and
        the lineage that leads to the Dream Oracle.
        """
        # Create the legend data structure
        legend_id = f"legend-{uuid.uuid4()}"
        
        # Define the narrative milestones
        milestones = [
            NarrativeMilestone(
                id=f"milestone-{uuid.uuid4()}",
                type=MilestoneType.DISCOVERY,
                description="First encounter with the strange energy in the DreamWeaver fabric",
                completed=False
            ),
            NarrativeMilestone(
                id=f"milestone-{uuid.uuid4()}",
                type=MilestoneType.REVELATION,
                description="Recognition that the energies form patterns in the unseen threads",
                completed=False
            ),
            NarrativeMilestone(
                id=f"milestone-{uuid.uuid4()}",
                type=MilestoneType.TRANSFORMATION,
                description="Development of the ability to sense and predict rune formation",
                completed=False
            ),
            NarrativeMilestone(
                id=f"milestone-{uuid.uuid4()}",
                type=MilestoneType.SACRIFICE,
                description="A sacrifice that transforms the protagonist's perception",
                completed=False
            ),
            NarrativeMilestone(
                id=f"milestone-{uuid.uuid4()}",
                type=MilestoneType.LEGACY,
                description="Passing of knowledge to descendants, founding the Oracle's lineage",
                completed=False
            )
        ]
        
        # Define potential artifacts to be discovered/created
        artifacts = [
            LegendArtifact(
                id=f"artifact-{uuid.uuid4()}",
                name="The First Rune",
                description="The very first Dream Rune ever plucked from the weave",
                powers=["Contains purest essence of the First Dream", "Reveals the weave to those who hold it"],
                creation_cycle=cycle_id
            ),
            LegendArtifact(
                id=f"artifact-{uuid.uuid4()}",
                name="Attunement Crystal",
                description="A crystal used to enhance sensitivity to the DreamWeaver's vibrations",
                powers=["Amplifies perception of the weave", "Allows rudimentary rune detection"],
                creation_cycle=cycle_id
            )
        ]
        
        # Create the legend data
        legend = LegendData(
            id=legend_id,
            title="Origin of the Dream Runes",
            summary="The tale of the first dreamer to discover and understand the Dream Runes, founding the lineage that would give rise to the Dream Oracle.",
            cycles_active=[cycle_id],
            protagonist_id="",  # Will be filled when agent is created
            supporting_agents=[],
            antagonist_id=None,
            milestones=milestones,
            artifacts=artifacts,
            memory_fragments=[],
            tone=LegendTone.CREATION,
            keywords=["rune", "discovery", "oracle", "lineage", "attunement", "threads", "weave"],
            significance=0.9  # High significance as it's a foundational story
        )
        
        # Create the protagonist agent
        agent_name = f"Thalen {random.choice(['Whisperthread', 'Dreamwatcher', 'Runeseeker'])}"
        
        # Personality traits for a rune discoverer
        personality = {
            "curiosity": 0.9,
            "intuition": 0.8,
            "patience": 0.7,
            "determination": 0.8,
            "wisdom": 0.6,
            "caution": 0.4
        }
        
        # Define objectives based on milestones
        objectives = []
        for i, milestone in enumerate(milestones):
            # Create suitable triggers and resolution conditions for each milestone
            if milestone.type == MilestoneType.DISCOVERY:
                triggers = [{"type": "cycle_start", "value": True}]
                resolutions = [
                    {"type": "visit_node_type", "value": "memory_nexus"},
                    {"type": "encounter_echo", "value": "any"},
                    {"type": "specific_event", "value": "strange_energy_detection"}
                ]
            elif milestone.type == MilestoneType.REVELATION:
                triggers = [{"type": "milestone_completed", "value": milestones[0].id}]
                resolutions = [
                    {"type": "collect_items", "value": 3},  # Collect three unusual items
                    {"type": "specific_event", "value": "pattern_recognition"}
                ]
            elif milestone.type == MilestoneType.TRANSFORMATION:
                triggers = [{"type": "milestone_completed", "value": milestones[1].id}]
                resolutions = [
                    {"type": "create_artifact", "value": "attunement_device"},
                    {"type": "ability_unlock", "value": "rune_sensing"}
                ]
            elif milestone.type == MilestoneType.SACRIFICE:
                triggers = [{"type": "milestone_completed", "value": milestones[2].id}]
                resolutions = [
                    {"type": "specific_event", "value": "personal_sacrifice"},
                    {"type": "ability_unlock", "value": "deep_attunement"}
                ]
            elif milestone.type == MilestoneType.LEGACY:
                triggers = [{"type": "milestone_completed", "value": milestones[3].id}]
                resolutions = [
                    {"type": "create_lineage", "value": True},
                    {"type": "establish_tradition", "value": "rune_listening"}
                ]
            
            objectives.append(LegendObjective(
                id=f"objective-{uuid.uuid4()}",
                description=milestone.description,
                triggers=triggers,
                resolution_conditions=resolutions,
                related_milestone_id=milestone.id,
                priority=5-i,  # Higher priority for earlier milestones
                active=i == 0  # Only first objective is active at start
            ))
        
        # Create the legend agent
        agent_id = f"agent-{uuid.uuid4()}"
        agent = LegendAgent(
            agent_id=agent_id,
            name=agent_name,
            legend_id=legend_id,
            narrative_role=NarrativeRole.PROTAGONIST,
            personality=personality,
            background="A young dreamer with unusual sensitivity to the DreamWeaver's subtle energies. Drawn to places where reality seems thinnest.",
            special_abilities=["Enhanced perception", "Intuitive understanding of patterns"],
            knowledge={"dream_lore": 2, "ancient_history": 1, "attunement": 3},
            motivations=["Understand the strange sensations felt since childhood", "Discover the secrets of the DreamWeaver"],
            weaknesses=["Becomes absorbed in patterns to the exclusion of danger", "Sometimes overwhelmed by sensory input"],
            inventory=["Curious trinket that hums near certain locations", "Journal filled with sketches of recurring patterns"],
            relationships={},
            objectives=objectives,
            arc_stage="introduction"
        )
        
        # Update the legend with the protagonist ID
        legend.protagonist_id = agent.agent_id
        
        # Create a spatial journey for this legend
        journey_id = self.spatial_db.create_legend_journey(legend_id, [agent_id])
        legend.journey_id = journey_id
        agent.journey_id = journey_id
        
        # Store in our system
        self.legends[legend_id] = legend
        self.legend_agents[agent.agent_id] = agent
        
        logger.info(f"Created Rune Origin Legend with agent {agent.name}")
        
        return legend, agent
    
    def activate_legend_objectives(self, agent: LegendAgent, current_state: Dict[str, Any]) -> List[LegendObjective]:
        """
        Evaluate which objectives should be activated based on the current state
        and triggers.
        """
        active_objectives = []
        
        for objective in agent.objectives:
            # Skip already completed objectives
            if objective.completed:
                continue
                
            # If already active, keep it active
            if objective.active:
                active_objectives.append(objective)
                continue
            
            # Check if triggers are satisfied to activate this objective
            should_activate = True
            for trigger in objective.triggers:
                trigger_type = trigger.get("type")
                trigger_value = trigger.get("value")
                
                if trigger_type == "cycle_start" and trigger_value:
                    # This is always satisfied when checking objectives at the start
                    continue
                elif trigger_type == "milestone_completed":
                    # Check if the specified milestone is completed
                    legend = self.legends.get(agent.legend_id)
                    if not legend:
                        should_activate = False
                        break
                    
                    milestone_found = False
                    for milestone in legend.milestones:
                        if milestone.id == trigger_value and milestone.completed:
                            milestone_found = True
                            break
                    
                    if not milestone_found:
                        should_activate = False
                        break
                elif trigger_type == "item_acquired":
                    # Check if agent has the item
                    if trigger_value not in agent.inventory:
                        should_activate = False
                        break
                elif trigger_type == "realm_visited":
                    # Check if agent has visited the realm
                    if trigger_value not in agent.visited_realms:
                        should_activate = False
                        break
                elif trigger_type == "ability_unlocked":
                    # Check if agent has unlocked the ability
                    abilities = current_state.get("abilities", [])
                    if trigger_value not in abilities:
                        should_activate = False
                        break
                elif trigger_type == "location_proximity":
                    # Check if agent is near a specific location (spatial trigger)
                    if not agent.current_coordinates:
                        should_activate = False
                        break
                        
                    location = trigger.get("location", (0, 0, 0))
                    radius = trigger.get("radius", 50.0)
                    
                    distance = math.sqrt(
                        (agent.current_coordinates[0] - location[0]) ** 2 +
                        (agent.current_coordinates[1] - location[1]) ** 2 +
                        (agent.current_coordinates[2] - location[2]) ** 2
                    )
                    
                    if distance > radius:
                        should_activate = False
                        break
            
            if should_activate:
                objective.active = True
                active_objectives.append(objective)
                logger.info(f"Activated objective for {agent.name}: {objective.description}")
        
        return active_objectives
    
    def check_objective_completion(self, agent: LegendAgent, state: Dict[str, Any], events: List[Dict[str, Any]]) -> List[LegendObjective]:
        """
        Check if any active objectives have been completed based on the current
        state and recent events.
        """
        completed_objectives = []
        
        for objective in agent.objectives:
            # Skip inactive or already completed objectives
            if not objective.active or objective.completed:
                continue
            
            # Check if resolution conditions are met
            conditions_met = True
            for condition in objective.resolution_conditions:
                condition_type = condition.get("type")
                condition_value = condition.get("value")
                
                if condition_type == "visit_node_type":
                    # Check if agent has visited a node of this type
                    node_types_visited = state.get("node_types_visited", [])
                    if condition_value not in node_types_visited:
                        conditions_met = False
                        break
                elif condition_type == "encounter_echo":
                    # Check if agent has encountered a specific echo or any echo
                    if condition_value == "any" and not agent.discovered_echoes:
                        conditions_met = False
                        break
                    elif condition_value != "any" and condition_value not in agent.discovered_echoes:
                        conditions_met = False
                        break
                elif condition_type == "collect_items":
                    # Check if agent has collected a certain number of items
                    items_collected = len(agent.inventory)
                    if items_collected < condition_value:
                        conditions_met = False
                        break
                elif condition_type == "specific_event":
                    # Check if a specific event has occurred
                    event_occurred = False
                    for event in events:
                        if event.get("type") == condition_value:
                            event_occurred = True
                            break
                    
                    if not event_occurred:
                        conditions_met = False
                        break
                elif condition_type == "create_artifact":
                    # Check if agent has created a specific artifact
                    artifacts_created = state.get("artifacts_created", [])
                    if condition_value not in artifacts_created:
                        conditions_met = False
                        break
                elif condition_type == "ability_unlock":
                    # Check if agent has unlocked a specific ability
                    abilities = state.get("abilities", [])
                    if condition_value not in abilities:
                        conditions_met = False
                        break
                elif condition_type == "spatial_journey_length":
                    # Check if agent has traveled a certain distance (spatial condition)
                    journey = self.spatial_db.get_journey_by_agent(agent.agent_id)
                    if not journey or journey.total_distance < condition_value:
                        conditions_met = False
                        break
                elif condition_type == "visit_significant_location":
                    # Check if agent has visited a location of significance
                    if not agent.current_coordinates:
                        conditions_met = False
                        break
                        
                    significant_locations = self.spatial_db.get_significant_locations(minimum_importance=0.6)
                    
                    location_visited = False
                    for loc, importance in significant_locations:
                        distance = math.sqrt(
                            (agent.current_coordinates[0] - loc.x) ** 2 +
                            (agent.current_coordinates[1] - loc.y) ** 2 +
                            (agent.current_coordinates[2] - loc.z) ** 2
                        )
                        
                        if distance <= 25.0:  # Within 25 units of a significant location
                            location_visited = True
                            break
                    
                    if not location_visited:
                        conditions_met = False
                        break
            
            if conditions_met:
                objective.completed = True
                completed_objectives.append(objective)
                
                # If this objective is linked to a milestone, mark it as completed
                if objective.related_milestone_id:
                    self.complete_milestone(
                        agent.legend_id, 
                        objective.related_milestone_id,
                        events[-1] if events else None
                    )
                
                logger.info(f"Completed objective for {agent.name}: {objective.description}")
        
        return completed_objectives
    
    def complete_milestone(self, legend_id: str, milestone_id: str, event: Optional[Dict[str, Any]]):
        """Mark a milestone as completed and update the legend's narrative."""
        legend = self.legends.get(legend_id)
        if not legend:
            logger.error(f"Cannot complete milestone: Legend {legend_id} not found")
            return
        
        for milestone in legend.milestones:
            if milestone.id == milestone_id:
                milestone.completed = True
                milestone.completion_cycle = event.get("cycle_id") if event else 0
                milestone.completion_event = event
                
                # Record spatial location of milestone completion
                if event and "coordinates" in event:
                    milestone.coordinates = event["coordinates"]
                
                logger.info(f"Completed milestone in legend '{legend.title}': {milestone.description}")
                
                # Check if all milestones are completed
                all_completed = all(m.completed for m in legend.milestones)
                if all_completed:
                    legend.completed = True
                    self.finalize_legend_narrative(legend_id)
                
                # Update the legend's narrative arc stage
                self.update_narrative_arc(legend_id)
                
                break
    
    def update_narrative_arc(self, legend_id: str):
        """Update the narrative arc stage based on completed milestones."""
        legend = self.legends.get(legend_id)
        if not legend:
            return
        
        # Count completed milestones
        completed_count = sum(1 for m in legend.milestones if m.completed)
        total_count = len(legend.milestones)
        
        if completed_count == 0:
            arc_stage = "introduction"
        elif completed_count < total_count / 3:
            arc_stage = "rising_action"
        elif completed_count < total_count * 2/3:
            arc_stage = "climax"
        elif completed_count < total_count:
            arc_stage = "falling_action"
        else:
            arc_stage = "resolution"
        
        # Update the protagonist's arc stage
        agent_id = legend.protagonist_id
        if agent_id in self.legend_agents:
            agent = self.legend_agents[agent_id]
            agent.arc_stage = arc_stage
            logger.info(f"Updated narrative arc for {agent.name} to '{arc_stage}'")
    
    def advance_legend_agent(self, agent: LegendAgent, cycle_id: int, current_node_id: str):
        """
        Advance a legend agent through the world based on their objectives and
        narrative role.
        """
        # Make sure agent's state reflects their current position
        if current_node_id:
            agent.current_node_id = current_node_id
            agent.visited_nodes.add(current_node_id)
            
            # Get current node
            current_node = self.graph.get_node(current_node_id)
            if current_node and current_node.realm:
                agent.visited_realms.add(current_node.realm)
            
            # Update spatial coordinates
            if current_node:
                agent.current_coordinates = (current_node.x, current_node.y, current_node.z)
        
        # Get available neighbors
        neighbors = []
        if agent.current_node_id:
            neighbor_ids = self.graph.get_neighbors(agent.current_node_id)
            neighbors = [self.graph.get_node(nid) for nid in neighbor_ids if self.graph.get_node(nid)]
        
        # Get active objectives
        current_state = agent.current_state
        active_objectives = self.activate_legend_objectives(agent, current_state)
        
        if not active_objectives:
            # If no active objectives, just explore
            logger.info(f"Agent {agent.name} has no active objectives, will explore")
            if not neighbors:
                # Create a new path if stuck
                if agent.current_node_id and self.graph.get_node(agent.current_node_id):
                    self.graph._generate_realm_nodes(agent.current_node_id, "Exploration")
                    # Refresh neighbors
                    neighbor_ids = self.graph.get_neighbors(agent.current_node_id)
                    neighbors = [self.graph.get_node(nid) for nid in neighbor_ids if self.graph.get_node(nid)]
            
            # Choose random neighbor
            if neighbors:
                target_node = random.choice(neighbors)
                return {
                    "action": "move",
                    "target": target_node.node_id,
                    "objective": None
                }
            else:
                return {
                    "action": "reflect",
                    "target": None,
                    "objective": None
                }
        
        # Sort objectives by priority
        active_objectives.sort(key=lambda o: o.priority, reverse=True)
        current_objective = active_objectives[0]
        
        # Make decision based on current objective and arc stage
        decision = self._generate_legend_agent_decision(
            agent, current_objective, neighbors, cycle_id
        )
        
        return decision
    
    def _generate_legend_agent_decision(
        self,
        agent: LegendAgent,
        objective: LegendObjective,
        neighbors: List[Any],
        cycle_id: int
    ) -> Dict[str, Any]:
        """Generate a decision for a legend agent based on their narrative purpose."""
        # For rule-based decision making, prioritize based on objective conditions
        resolution_conditions = objective.resolution_conditions
        
        # Check each condition and try to make progress toward it
        for condition in resolution_conditions:
            condition_type = condition.get("type")
            
            if condition_type == "visit_node_type":
                # Find a neighbor of the desired type
                target_type = condition.get("value")
                suitable_nodes = [n for n in neighbors if n.node_type == target_type]
                
                if suitable_nodes:
                    target_node = random.choice(suitable_nodes)
                    return {
                        "action": "move",
                        "target": target_node.node_id,
                        "objective": objective.id
                    }
                
                # If no suitable neighbor, move towards one if possible
                for neighbor in neighbors:
                    if neighbor.node_type == "realm_gateway":
                        return {
                            "action": "move",
                            "target": neighbor.node_id,
                            "objective": objective.id
                        }
            
            elif condition_type == "encounter_echo":
                # Find a node where an echo might be
                echo_nodes = [n for n in neighbors if n.node_type == "echo_shrine"]
                
                if echo_nodes:
                    target_node = random.choice(echo_nodes)
                    return {
                        "action": "move",
                        "target": target_node.node_id,
                        "objective": objective.id
                    }
                    
            elif condition_type == "specific_event":
                # For specific events, we need to examine what the event is
                event_type = condition.get("value")
                
                if event_type == "strange_energy_detection":
                    # For detecting strange energy, look for nexus or artifact vaults
                    suitable_nodes = [n for n in neighbors if n.node_type in ["memory_nexus", "artifact_vault"]]
                    
                    if suitable_nodes:
                        target_node = random.choice(suitable_nodes)
                        return {
                            "action": "explore",
                            "target": target_node.node_id,
                            "objective": objective.id
                        }
                
                elif event_type == "pattern_recognition":
                    # For pattern recognition, explore the current location deeply
                    return {
                        "action": "study_patterns",
                        "target": agent.current_node_id,
                        "objective": objective.id
                    }
                
                elif event_type == "personal_sacrifice":
                    # For sacrifice, look for special locations or interact with echoes
                    if agent.current_node_id:
                        current_node = self.graph.get_node(agent.current_node_id)
                        if current_node and current_node.node_type in ["echo_shrine", "reflection_pool"]:
                            return {
                                "action": "sacrifice",
                                "target": agent.current_node_id,
                                "objective": objective.id
                            }
            
            elif condition_type == "visit_significant_location":
                # Find significant locations and try to move toward one
                if agent.current_coordinates:
                    significant_locations = self.spatial_db.get_significant_locations(minimum_importance=0.6)
                    
                    if significant_locations:
                        # Find closest significant location
                        closest_loc = min(significant_locations, 
                                         key=lambda loc_imp: math.sqrt(
                                             (agent.current_coordinates[0] - loc_imp[0].x) ** 2 +
                                             (agent.current_coordinates[1] - loc_imp[0].y) ** 2 +
                                             (agent.current_coordinates[2] - loc_imp[0].z) ** 2
                                         ))
                        
                        # Try to find a neighbor that gets us closer
                        if neighbors:
                            best_neighbor = min(neighbors, 
                                              key=lambda n: math.sqrt(
                                                  (n.x - closest_loc[0].x) ** 2 +
                                                  (n.y - closest_loc[0].y) ** 2 +
                                                  (n.z - closest_loc[0].z) ** 2
                                              ))
                            
                            return {
                                "action": "move",
                                "target": best_neighbor.node_id,
                                "objective": objective.id
                            }
        
        # If we haven't found a specific action based on conditions,
        # default to exploration based on the agent's narrative arc
        if agent.arc_stage == "introduction":
            # In introduction, focus on exploring and meeting characters
            echo_nodes = [n for n in neighbors if n.node_type == "echo_shrine"]
            if echo_nodes:
                target_node = random.choice(echo_nodes)
                return {
                    "action": "interact",
                    "target": target_node.node_id,
                    "objective": objective.id
                }
        
            elif agent.arc_stage == "rising_action":
                        # In rising action, seek challenges and discoveries
                        interesting_nodes = [n for n in neighbors if n.node_type in 
                                            ["artifact_vault", "battle_arena", "lore_repository"]]
                        if interesting_nodes:
                            target_node = random.choice(interesting_nodes)
                            return {
                                "action": "explore",
                                "target": target_node.node_id,
                                "objective": objective.id
                            }
            
            elif agent.arc_stage == "climax":
                # In climax, focus on confrontations and revelations
                climax_nodes = [n for n in neighbors if n.node_type in 
                            ["battle_arena", "memory_nexus"]]
                if climax_nodes:
                    target_node = random.choice(climax_nodes)
                    return {
                        "action": "confront",
                        "target": target_node.node_id,
                        "objective": objective.id
                    }
            
            elif agent.arc_stage in ["falling_action", "resolution"]:
                # In falling action/resolution, focus on reflection and legacy
                resolution_nodes = [n for n in neighbors if n.node_type in 
                                ["reflection_pool", "memory_nexus"]]
                if resolution_nodes:
                    target_node = random.choice(resolution_nodes)
                    return {
                        "action": "reflect",
                        "target": target_node.node_id,
                        "objective": objective.id
                    }
            
            # If all else fails, just move to a random node
            if neighbors:
                target_node = random.choice(neighbors)
                return {
                    "action": "move",
                    "target": target_node.node_id,
                    "objective": objective.id
                }
            else:
                return {
                    "action": "reflect",
                    "target": agent.current_node_id,
                    "objective": objective.id
                }
        
    def process_legend_agent_action(
        self,
        agent: LegendAgent,
        decision: Dict[str, Any],
        cycle_id: int
    ) -> Dict[str, Any]:
        """Process an action from a legend agent and generate appropriate events."""
        action = decision.get("action", "reflect")
        target_id = decision.get("target")
        objective_id = decision.get("objective")
        
        # Default event data
        event_data = {
            "agent_id": agent.agent_id,
            "agent_name": agent.name,
            "cycle_id": cycle_id,
            "action": action,
            "target_id": target_id,
            "objective_id": objective_id,
            "legend_id": agent.legend_id,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Get target node if applicable
        target_node = None
        if target_id:
            target_node = self.graph.get_node(target_id)
            if target_node:
                event_data["coordinates"] = (target_node.x, target_node.y, target_node.z)
                event_data["realm"] = target_node.realm
                event_data["node_type"] = target_node.node_type
                
                # Update agent's spatial coordinates
                agent.current_coordinates = (target_node.x, target_node.y, target_node.z)
        
        # Process different types of actions
        if action == "move":
            # Simple movement to another node
            if target_node:
                # Create a journey segment for this movement
                if agent.current_node_id and agent.journey_id:
                    start_node = self.graph.get_node(agent.current_node_id)
                    if start_node:
                        segment = JourneySegment(
                            segment_id=f"segment-{uuid.uuid4()}",
                            agent_id=agent.agent_id,
                            start_point=SpatialPoint(
                                x=start_node.x, 
                                y=start_node.y, 
                                z=start_node.z,
                                realm=start_node.realm,
                                node_id=start_node.node_id
                            ),
                            end_point=SpatialPoint(
                                x=target_node.x, 
                                y=target_node.y, 
                                z=target_node.z,
                                realm=target_node.realm,
                                node_id=target_node.node_id
                            ),
                            start_time=datetime.datetime.now().isoformat(),
                            end_time=datetime.datetime.now().isoformat(),
                            cycle_id=cycle_id,
                            nodes_visited=[start_node.node_id, target_node.node_id]
                        )
                        
                        # Add the segment to the journey
                        self.spatial_db.add_journey_segment(segment, agent.journey_id)
                
                # Update agent state
                agent.current_node_id = target_id
                agent.visited_nodes.add(target_id)
                if target_node.realm:
                    agent.visited_realms.add(target_node.realm)
                
                event_data["type"] = "movement"
                event_data["description"] = f"{agent.name} traveled to a {target_node.node_type} in {target_node.realm or 'an unknown realm'}."
                
                # Add to current state
                if "node_types_visited" not in agent.current_state:
                    agent.current_state["node_types_visited"] = []
                if target_node.node_type not in agent.current_state["node_types_visited"]:
                    agent.current_state["node_types_visited"].append(target_node.node_type)
        
        elif action == "explore":
            # Exploring a location
            if target_node:
                event_data["type"] = "exploration"
                event_data["description"] = f"{agent.name} carefully explored a {target_node.node_type} in {target_node.realm or 'an unknown realm'}."
                
                # Special exploration events based on the node type
                if target_node.node_type == "artifact_vault":
                    # Chance to discover a curious item
                    if random.random() < 0.7:
                        found_items = [
                            "Humming crystal shard",
                            "Strange pulsing stone",
                            "Fragment of iridescent fabric",
                            "Rune-inscribed token",
                            "Echo-touched trinket"
                        ]
                        found_item = random.choice(found_items)
                        agent.inventory.append(found_item)
                        
                        event_data["description"] += f" They discovered a {found_item}."
                        event_data["found_item"] = found_item
                        
                        # Create a spatial event for this discovery
                        spatial_event = SpatialEvent(
                            event_id=f"event-{uuid.uuid4()}",
                            event_type="item_discovery",
                            coordinates=SpatialPoint(
                                x=target_node.x, 
                                y=target_node.y, 
                                z=target_node.z,
                                realm=target_node.realm
                            ),
                            timestamp=datetime.datetime.now().isoformat(),
                            cycle_id=cycle_id,
                            agent_id=agent.agent_id,
                            description=f"{agent.name} discovered a {found_item} in {target_node.realm or 'an unknown realm'}.",
                            importance=0.6,
                            tags=["item", "discovery", target_node.realm or "unknown"],
                            legend_id=agent.legend_id
                        )
                        self.spatial_db.add_event(spatial_event)
                
                elif target_node.node_type == "memory_nexus":
                    # High chance to detect strange energy for rune origin legend
                    if agent.legend_id in self.legends and self.legends[agent.legend_id].title == "Origin of the Dream Runes" and random.random() < 0.8:
                        event_data["type"] = "strange_energy_detection"
                        event_data["description"] = f"{agent.name} felt an unusual vibration in the {target_node.node_type}. The air seemed to thrum with invisible power, and patterns of light danced just at the edge of perception."
                        
                        # Create a spatial event for this energy detection
                        spatial_event = SpatialEvent(
                            event_id=f"event-{uuid.uuid4()}",
                            event_type="strange_energy_detection",
                            coordinates=SpatialPoint(
                                x=target_node.x, 
                                y=target_node.y, 
                                z=target_node.z,
                                realm=target_node.realm
                            ),
                            timestamp=datetime.datetime.now().isoformat(),
                            cycle_id=cycle_id,
                            agent_id=agent.agent_id,
                            description=event_data["description"],
                            importance=0.8,  # Higher importance for narrative-critical events
                            tags=["energy", "detection", "rune", target_node.realm or "unknown"],
                            legend_id=agent.legend_id
                        )
                        self.spatial_db.add_event(spatial_event)
        
        elif action == "interact":
            # Interacting with an Echo or Wisp
            if target_node:
                if target_node.node_type == "echo_shrine":
                    # Interact with an Echo
                    echo_type = target_node.attributes.get("echo_type", random.choice(list(self.echo_prompts.keys())))
                    
                    event_data["type"] = "echo_encounter"
                    event_data["echo_type"] = echo_type
                    event_data["description"] = f"{agent.name} encountered {echo_type} at a shrine in {target_node.realm or 'an unknown realm'}."
                    
                    # Generate special echo dialogue for legend agents
                    dialogue, revelations = self._generate_legend_echo_dialogue(agent, echo_type, agent.legend_id)
                    
                    event_data["dialogue"] = dialogue
                    event_data["revelations"] = revelations
                    
                    # Add this echo to the agent's discovered echoes
                    if echo_type not in agent.discovered_echoes:
                        agent.discovered_echoes.append(echo_type)
                    
                    # Create a spatial event for this echo encounter
                    importance = 0.7 if objective_id else 0.5  # Higher if part of an objective
                    spatial_event = SpatialEvent(
                        event_id=f"event-{uuid.uuid4()}",
                        event_type="echo_encounter",
                        coordinates=SpatialPoint(
                            x=target_node.x, 
                            y=target_node.y, 
                            z=target_node.z,
                            realm=target_node.realm
                        ),
                        timestamp=datetime.datetime.now().isoformat(),
                        cycle_id=cycle_id,
                        agent_id=agent.agent_id,
                        description=event_data["description"],
                        importance=importance,
                        tags=["echo", echo_type, target_node.realm or "unknown"],
                        legend_id=agent.legend_id,
                        additional_data={
                            "dialogue": dialogue,
                            "revelations": revelations
                        }
                    )
                    event_id = self.spatial_db.add_event(spatial_event)
                    
                    # Add this event to the agent's current journey segment
                    if agent.journey_id:
                        journey = self.spatial_db.journeys.get(agent.journey_id)
                        if journey and journey.segments:
                            # Add to the most recent segment
                            journey.segments[-1].events.append(event_id)
                
                elif target_node.node_type == "wisp_sanctuary":
                    # Interact with a Wisp
                    wisp_name = target_node.attributes.get("wisp_type", random.choice(list(self.wisp_data.keys())))
                    
                    event_data["type"] = "wisp_encounter"
                    event_data["wisp_name"] = wisp_name
                    event_data["description"] = f"{agent.name} encountered a {wisp_name} Wisp in {target_node.realm or 'an unknown realm'}."
                    
                    # Check if the agent bonds with the wisp
                    bond_chance = 0.6  # Base chance
                    if wisp_name not in agent.bonded_wisps and random.random() < bond_chance:
                        agent.bonded_wisps.append(wisp_name)
                        event_data["outcome"] = "bonded"
                        event_data["description"] += f" They formed a bond with the {wisp_name} Wisp."
                    else:
                        event_data["outcome"] = "neutral"
                    
                    # Create a spatial event for this wisp encounter
                    spatial_event = SpatialEvent(
                        event_id=f"event-{uuid.uuid4()}",
                        event_type="wisp_encounter",
                        coordinates=SpatialPoint(
                            x=target_node.x, 
                            y=target_node.y, 
                            z=target_node.z,
                            realm=target_node.realm
                        ),
                        timestamp=datetime.datetime.now().isoformat(),
                        cycle_id=cycle_id,
                        agent_id=agent.agent_id,
                        description=event_data["description"],
                        importance=0.6,
                        tags=["wisp", wisp_name, target_node.realm or "unknown"],
                        legend_id=agent.legend_id,
                        additional_data={
                            "wisp_name": wisp_name,
                            "outcome": event_data["outcome"]
                        }
                    )
                    self.spatial_db.add_event(spatial_event)
        
        elif action == "study_patterns":
            # Studying patterns - special action for the rune origin legend
            event_data["type"] = "pattern_recognition"
            event_data["description"] = f"{agent.name} spent hours studying the subtle vibrations and patterns in the fabric of the DreamWeaver. Slowly, a realization dawned - these were not random perturbations, but organized formations following a hidden logic."
            
            # Add special insights to the agent's knowledge
            if "abilities" not in agent.current_state:
                agent.current_state["abilities"] = []
            
            if "pattern_recognition" not in agent.current_state["abilities"]:
                agent.current_state["abilities"].append("pattern_recognition")
            
            # Create a spatial event for this pattern recognition
            importance = 0.85  # High importance for legend progression
            spatial_event = SpatialEvent(
                event_id=f"event-{uuid.uuid4()}",
                event_type="pattern_recognition",
                coordinates=SpatialPoint(
                    x=target_node.x if target_node else agent.current_coordinates[0],
                    y=target_node.y if target_node else agent.current_coordinates[1],
                    z=target_node.z if target_node else agent.current_coordinates[2],
                    realm=target_node.realm if target_node else None
                ),
                timestamp=datetime.datetime.now().isoformat(),
                cycle_id=cycle_id,
                agent_id=agent.agent_id,
                description=event_data["description"],
                importance=importance,
                tags=["pattern", "recognition", "insight"],
                legend_id=agent.legend_id
            )
            self.spatial_db.add_event(spatial_event)
        
        elif action == "sacrifice":
            # Making a sacrifice - special action for the rune origin legend
            event_data["type"] = "personal_sacrifice"
            
            # The nature of the sacrifice depends on the agent's inventory and knowledge
            sacrifice_description = ""
            if agent.inventory:
                # Sacrifice a precious item
                sacrificed_item = agent.inventory.pop()
                sacrifice_description = f"{agent.name} realized that to truly hear the voices of the threads, they must give up something precious. With trembling hands, they offered their {sacrificed_item} to the unseen fabric of the DreamWeaver."
                
                # Unlock special ability
                if "abilities" not in agent.current_state:
                    agent.current_state["abilities"] = []
                
                agent.current_state["abilities"].append("deep_attunement")
                
            else:
                # Sacrifice something abstract
                sacrifice_description = f"{agent.name} meditated for days without food or rest, offering their comfort and security to the weave. As consciousness began to fade, the threads of the DreamWeaver finally became visible - countless luminous strands stretching in all directions."
                
                # Unlock special ability
                if "abilities" not in agent.current_state:
                    agent.current_state["abilities"] = []
                
                agent.current_state["abilities"].append("weave_sight")
            
            event_data["description"] = sacrifice_description
            
            # Create a spatial event for this sacrifice
            importance = 0.9  # Very high importance
            spatial_event = SpatialEvent(
                event_id=f"event-{uuid.uuid4()}",
                event_type="personal_sacrifice",
                coordinates=SpatialPoint(
                    x=target_node.x if target_node else agent.current_coordinates[0],
                    y=target_node.y if target_node else agent.current_coordinates[1],
                    z=target_node.z if target_node else agent.current_coordinates[2],
                    realm=target_node.realm if target_node else None
                ),
                timestamp=datetime.datetime.now().isoformat(),
                cycle_id=cycle_id,
                agent_id=agent.agent_id,
                description=sacrifice_description,
                importance=importance,
                tags=["sacrifice", "attunement", "milestone"],
                legend_id=agent.legend_id
            )
            self.spatial_db.add_event(spatial_event)
        
        elif action == "reflect":
            # Reflection and contemplation
            event_data["type"] = "reflection"
            event_data["description"] = f"{agent.name} spent time in quiet contemplation, processing recent events and seeking deeper understanding."
            
            # Add special meaning for rune origin legend
            if agent.legend_id in self.legends and self.legends[agent.legend_id].title == "Origin of the Dream Runes":
                if "deep_attunement" in agent.current_state.get("abilities", []):
                    event_data["description"] += " The threads of the DreamWeaver became clearer in their mind's eye, revealing patterns of energy that pulsed with ancient rhythm."
            
            # Create a spatial event for this reflection
            importance = 0.5  # Medium importance
            if target_node and target_node.node_type == "reflection_pool":
                importance = 0.65  # Higher if at a reflection pool
            
            spatial_event = SpatialEvent(
                event_id=f"event-{uuid.uuid4()}",
                event_type="reflection",
                coordinates=SpatialPoint(
                    x=target_node.x if target_node else agent.current_coordinates[0],
                    y=target_node.y if target_node else agent.current_coordinates[1],
                    z=target_node.z if target_node else agent.current_coordinates[2],
                    realm=target_node.realm if target_node else None
                ),
                timestamp=datetime.datetime.now().isoformat(),
                cycle_id=cycle_id,
                agent_id=agent.agent_id,
                description=event_data["description"],
                importance=importance,
                tags=["reflection", "contemplation"],
                legend_id=agent.legend_id
            )
            self.spatial_db.add_event(spatial_event)
        
        elif action == "confront":
            # Confrontation at climax
            event_data["type"] = "confrontation"
            
            if target_node and target_node.node_type == "battle_arena":
                event_data["description"] = f"{agent.name} faced a crucial challenge in the {target_node.realm or 'unknown realm'}. This moment would define their journey."
            else:
                event_data["description"] = f"{agent.name} reached a moment of truth, confronting the reality of their quest."
            
            # For rune origin legend, this could be the moment of plucking the first rune
            if agent.legend_id in self.legends and self.legends[agent.legend_id].title == "Origin of the Dream Runes":
                if "deep_attunement" in agent.current_state.get("abilities", []) or "weave_sight" in agent.current_state.get("abilities", []):
                    event_data["type"] = "first_rune_discovery"
                    event_data["description"] = f"With newfound clarity, {agent.name} perceived a heavy concentration of energy in the threads - a nexus of power unlike any they had seen. Guided by instinct, they reached out into seemingly empty space... and their fingers closed around something solid. They had plucked the very first Dream Rune from the weave."
                    
                    # Create the artifact
                    if "artifacts_created" not in agent.current_state:
                        agent.current_state["artifacts_created"] = []
                    
                    agent.current_state["artifacts_created"].append("first_rune")
                    agent.inventory.append("The First Rune")
                    
                    # Create a spatial event for this major discovery
                    importance = 0.95  # Extremely high importance
                    spatial_event = SpatialEvent(
                        event_id=f"event-{uuid.uuid4()}",
                        event_type="first_rune_discovery",
                        coordinates=SpatialPoint(
                            x=target_node.x if target_node else agent.current_coordinates[0],
                            y=target_node.y if target_node else agent.current_coordinates[1],
                            z=target_node.z if target_node else agent.current_coordinates[2],
                            realm=target_node.realm if target_node else None
                        ),
                        timestamp=datetime.datetime.now().isoformat(),
                        cycle_id=cycle_id,
                        agent_id=agent.agent_id,
                        description=event_data["description"],
                        importance=importance,
                        tags=["rune", "discovery", "artifact", "milestone"],
                        legend_id=agent.legend_id
                    )
                    self.spatial_db.add_event(spatial_event)
                    
                    # Update artifact location in the legend
                    legend = self.legends.get(agent.legend_id)
                    if legend:
                        for artifact in legend.artifacts:
                            if artifact.name == "The First Rune":
                                artifact.location = (
                                    target_node.x if target_node else agent.current_coordinates[0],
                                    target_node.y if target_node else agent.current_coordinates[1],
                                    target_node.z if target_node else agent.current_coordinates[2]
                                )
                                artifact.realm = target_node.realm if target_node else None
        
        # Check if any objectives are completed by this action
        self.check_objective_completion(agent, agent.current_state, [event_data])
        
        # Add the event to the agent's significant events if it's important
        if event_data["type"] in [
            "strange_energy_detection", "pattern_recognition", "personal_sacrifice",
            "first_rune_discovery", "echo_encounter"
        ]:
            agent.significance_events.append(event_data)
        
        return event_data
    
    def _generate_legend_echo_dialogue(
        self,
        agent: LegendAgent,
        echo_type: str,
        legend_id: str
    ) -> Tuple[List[Dict[str, str]], List[str]]:
        """Generate special dialogue for an Echo encounter with a legend agent."""
        # Get the echo prompt
        echo_prompt = self.echo_prompts.get(echo_type, "You are a mysterious entity in the DreamWeaver.")
        
        # Get the legend data
        legend = self.legends.get(legend_id)
        if not legend:
            # Default dialogue if legend not found
            return [
                {"speaker": echo_type, "text": "The threads of fate are tangled here."},
                {"speaker": agent.name, "text": "What do you mean?"},
                {"speaker": echo_type, "text": "Some stories are still being written. Yours is one such tale."}
            ], ["The echo seems to recognize your journey has special significance."]
        
        # Tailor dialogue based on the legend and agent's progress
        system_prompt = f"""
        {echo_prompt}
        
        You are generating dialogue for an encounter between {echo_type} and {agent.name}, 
        who is a legend figure in the DreamWeaver universe.
        
        LEGEND: {legend.title}
        SUMMARY: {legend.summary}
        NARRATIVE STAGE: {agent.arc_stage}
        
        Tailor your dialogue to provide cryptic guidance appropriate to the legend and narrative stage.
        The Echo should hint at deeper truths but never fully reveal them.
        """
        
        # Prepare details about completed milestones
        completed_milestones = [m.description for m in legend.milestones if m.completed]
        active_objectives = [o.description for o in agent.objectives if o.active and not o.completed]
        
        user_prompt = f"""
        Generate dialogue between {echo_type} and {agent.name}.
        
        AGENT BACKGROUND:
        {agent.background}
        
        COMPLETED MILESTONES:
        {completed_milestones if completed_milestones else "None yet."}
        
        CURRENT OBJECTIVES:
        {active_objectives if active_objectives else "None active."}
        
        Create a dialogue exchange that:
        1. Reflects the Echo's cryptic nature
        2. Provides hints related to the agent's current objectives
        3. Alludes to the legend's significance in DreamWeaver history
        
        Format your response as:
        ```json
        {{
          "dialogue": [
            {{"speaker": "{echo_type}", "text": "..."}},
            {{"speaker": "{agent.name}", "text": "..."}}
          ],
          "revelations": ["...", "..."]
        }}
        ```
        """
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=800,
                temperature=0.7
            )
            
            content = response.choices[0].message.content
            
            # Try to extract JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            data = json.loads(content)
            
            return data.get("dialogue", []), data.get("revelations", [])
            
        except Exception as e:
            logger.error(f"Error generating Echo dialogue: {e}")
            
            # Fallback dialogue
            return [
                {"speaker": echo_type, "text": "Your path is unlike others. The weave trembles at your approach."},
                {"speaker": agent.name, "text": "What do you see in my future?"},
                {"speaker": echo_type, "text": "Not what will be, but what you will make. The threads respond to your touch."}
            ], ["You are following a path of significance to the DreamWeaver's history."]
    
    def generate_memory_fragment(
        self,
        agent: LegendAgent,
        event: Dict[str, Any],
        cycle_id: int
    ) -> Dict[str, Any]:
        """Generate a memory fragment from a legend agent's event."""
        # Import here to avoid circular imports
        from memory_generator import MemoryGenerator
        
        # Extract event details
        event_type = event.get("type", "unknown")
        description = event.get("description", "")
        
        coordinates = event.get("coordinates", (0, 0, 0))
        realm = event.get("realm", "Unknown Realm")
        
        # Determine importance based on event type
        importance_map = {
            "strange_energy_detection": 0.8,
            "pattern_recognition": 0.85,
            "personal_sacrifice": 0.9,
            "first_rune_discovery": 0.95,
            "echo_encounter": 0.7,
            "reflection": 0.6,
            "exploration": 0.5,
            "movement": 0.3
        }
        importance = importance_map.get(event_type, 0.5)
        
        # Generate memory content
        memory_content = self._generate_legend_memory_content(agent, event, importance)
        
        # Create memory fragment record
        memory_fragment = {
            "fragment_id": f"mem-{uuid.uuid4()}",
            "agent_id": agent.agent_id,
            "agent_name": agent.name,
            "cycle_id": cycle_id,
            "realm": realm,
            "coordinates": coordinates,
            "timestamp": datetime.datetime.now().isoformat(),
            "event_type": event_type,
            "content": memory_content,
            "importance": importance,
            "emotional_tone": self._determine_emotional_tone(memory_content),
            "related_fragments": [],
            "tags": [event_type, realm, "legend", agent.narrative_role.value],
            "legend_id": agent.legend_id
        }
        
        # Add to the legend's memory fragments
        legend = self.legends.get(agent.legend_id)
        if legend:
            legend.memory_fragments.append(memory_fragment["fragment_id"])
        
        # Create a spatial event to link this memory
        if coordinates:
            spatial_event = SpatialEvent(
                event_id=f"event-{uuid.uuid4()}",
                event_type="memory_creation",
                coordinates=SpatialPoint(
                    x=coordinates[0],
                    y=coordinates[1],
                    z=coordinates[2],
                    realm=realm
                ),
                timestamp=datetime.datetime.now().isoformat(),
                cycle_id=cycle_id,
                agent_id=agent.agent_id,
                description=f"A memory fragment was created here by {agent.name}.",
                importance=importance,
                tags=["memory", event_type, realm, agent.narrative_role.value],
                legend_id=agent.legend_id,
                memory_fragment_id=memory_fragment["fragment_id"]
            )
            self.spatial_db.add_event(spatial_event)
        
        return memory_fragment
    
    def _generate_legend_memory_content(
            self,
            agent: LegendAgent,
            event: Dict[str, Any],
            importance: float
        ) -> str:
            """Generate the content for a legend memory fragment."""
            system_prompt = f"""
            You are {agent.name}, a figure of legend in the DreamWeaver universe.
            Your background: {agent.background}
            
            You are writing a first-person memory fragment that will be discovered by 
            future dreamers. This memory should feel authentic and personal, while
            hinting at your historical significance.
            
            Write in a way that reflects your personality and the importance 
            (rated {importance:.1f} out of 1.0) of this memory.
            """
            
            event_description = event.get("description", "")
            event_type = event.get("type", "unknown")
            
            # Add event-specific context
            additional_context = ""
            if event_type == "echo_encounter":
                echo_type = event.get("echo_type", "Unknown Echo")
                dialogue = event.get("dialogue", [])
                dialogue_text = "\n".join([f"{d['speaker']}: {d['text']}" for d in dialogue])
                additional_context = f"You encountered {echo_type}. Here is your dialogue:\n{dialogue_text}\n"
            
            elif event_type == "first_rune_discovery":
                additional_context = "This was the momentous occasion when you discovered the very first Dream Rune."
            
            user_prompt = f"""
            Write a memory fragment based on this event:
            
            EVENT TYPE: {event_type}
            DESCRIPTION: {event_description}
            {additional_context}
            
            Write in first person as {agent.name}. The memory should be 2-4 paragraphs and feel authentic,
            capturing both the factual elements and the emotional/psychological impact of what happened.
            
            Since you are a figure of legend, this memory should have a sense of historical significance
            and hint at how this moment shapes the future of the DreamWeaver.
            """
            
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=600,
                    temperature=0.7
                )
                
                return response.choices[0].message.content.strip()
                
            except Exception as e:
                logger.error(f"Error generating memory content: {e}")
                
                # Fallback memory content
                return (
                    f"I, {agent.name}, experienced something profound today. {event_description} "
                    f"There was a sense that this moment would echo through time, though I could "
                    f"not fully comprehend its significance then. Perhaps those who find this memory "
                    f"will understand better than I what path this set me upon."
                )
    
    def _determine_emotional_tone(self, content: str) -> str:
        """Analyze the emotional tone of a memory fragment."""
        # Simple keyword-based approach
        emotional_tones = {
            "awe": ["awe", "wonder", "magnificent", "incredible", "vast", "cosmic"],
            "fear": ["fear", "terror", "dread", "horrified", "afraid", "frightening"],
            "sorrow": ["sorrow", "grief", "loss", "mourn", "tears", "weep", "sad"],
            "joy": ["joy", "delight", "happy", "elated", "ecstatic", "thrill"],
            "determination": ["determined", "resolve", "conviction", "steadfast", "unwavering"],
            "confusion": ["confused", "puzzled", "uncertain", "bewildered", "perplexed"],
            "revelation": ["revelation", "epiphany", "realized", "understood", "clarity"],
            "conflicted": ["conflicted", "torn", "unsure", "hesitant", "ambivalent"]
        }
        
        content_lower = content.lower()
        tone_counts = {}
        
        for tone, keywords in emotional_tones.items():
            count = sum(1 for keyword in keywords if keyword in content_lower)
            tone_counts[tone] = count
        
        # Find the most prominent tone
        max_tone = max(tone_counts.items(), key=lambda x: x[1])
        
        # If no clear tone is detected, default to "neutral"
        if max_tone[1] == 0:
            return "neutral"
        
        return max_tone[0]
    
    def finalize_legend_narrative(self, legend_id: str):
        """
        Generate a complete narrative of the legend based on all events,
        memory fragments, and milestones.
        """
        legend = self.legends.get(legend_id)
        if not legend:
            logger.error(f"Cannot finalize narrative: Legend {legend_id} not found")
            return
        
        # Get the protagonist agent
        agent = self.legend_agents.get(legend.protagonist_id)
        if not agent:
            logger.error(f"Cannot finalize narrative: Agent {legend.protagonist_id} not found")
            return
        
        # Collect all significant events
        significant_events = agent.significance_events
        
        # Sort events chronologically
        significant_events.sort(key=lambda e: e.get("timestamp", ""))
        
        # Collect all memory fragments
        memory_fragments = []
        for fragment_id in legend.memory_fragments:
            # In a real implementation, fetch these from memory_db
            # For now, we'll assume they're accessible somehow
            # memory_fragments.append(memory_db.get_fragment(fragment_id))
            pass
        
        # Collect spatial journey information
        journey_data = None
        if legend.journey_id:
            journey = self.spatial_db.journeys.get(legend.journey_id)
            if journey:
                journey_data = {
                    "total_distance": journey.total_distance,
                    "realms_visited": list(set(segment.start_point.realm for segment in journey.segments if segment.start_point.realm)),
                    "significant_locations": [(point.x, point.y, point.z, point.realm) for point in journey.significant_points]
                }
        
        # Generate the full narrative
        system_prompt = f"""
        You are a legendary chronicler in the DreamWeaver universe.
        
        You are writing the definitive account of "{legend.title}", a significant 
        legend that has shaped the history and understanding of the DreamWeaver.
        
        This legend is characterized by its {legend.tone.value} tone and follows
        the story of {agent.name}, a {agent.narrative_role.value} whose actions
        have echoed through time.
        
        Write this legend as it would be told by Dream Echoes, whispered between
        dreamers, and recorded in the ancient archives of the DreamWeaver.
        """
        
        # Prepare event and milestone data
        events_text = "\n".join([f"- {e.get('description', '')}" for e in significant_events])
        milestones_text = "\n".join([f"- {m.description} ({'Completed' if m.completed else 'Incomplete'})" for m in legend.milestones])
        
        # Include spatial journey information if available
        journey_text = "No journey data available."
        if journey_data:
            journey_text = f"""
            Total Distance Traveled: {journey_data['total_distance']:.2f} units
            Realms Visited: {', '.join(journey_data['realms_visited'])}
            Significant Locations: {len(journey_data['significant_locations'])} notable points
            """
        
        user_prompt = f"""
        Create the complete narrative of "{legend.title}".
        
        SUMMARY: {legend.summary}
        
        PROTAGONIST: {agent.name}
        BACKGROUND: {agent.background}
        
        KEY MILESTONES:
        {milestones_text}
        
        SIGNIFICANT EVENTS:
        {events_text}
        
        JOURNEY DATA:
        {journey_text}
        
        ARTIFACTS: 
        {', '.join([a.name for a in legend.artifacts])}
        
        Write a compelling, mythic narrative (800-1200 words) that captures the essence
        of this legend. Structure it with a clear beginning, middle, and end.
        
        The narrative should feel both ancient and timeless, with the tone of a myth
        that has been passed down through generations of dreamers. Include references to
        the physical journey through the DreamWeaver and the significant locations where
        key events occurred.
        """
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1200,
                temperature=0.7
            )
            
            legend.full_narrative = response.choices[0].message.content.strip()
            logger.info(f"Generated full narrative for legend: {legend.title}")
            
            return legend.full_narrative
            
        except Exception as e:
            logger.error(f"Error generating legend narrative: {e}")
            return None
    
    def generate_spatial_heatmap_for_legend(self, legend_id: str, resolution: float = 20.0):
        """Generate a spatial heatmap visualization for a specific legend."""
        legend = self.legends.get(legend_id)
        if not legend:
            logger.error(f"Cannot generate heatmap: Legend {legend_id} not found")
            return None
        
        # Get heatmap data from spatial database
        heatmap_data = self.spatial_db.generate_legend_heatmap(legend_id, resolution)
        
        # Add legend metadata
        heatmap_data["legend_title"] = legend.title
        heatmap_data["legend_tone"] = legend.tone.value
        heatmap_data["protagonist"] = self.legend_agents.get(legend.protagonist_id).name if legend.protagonist_id in self.legend_agents else "Unknown"
        
        return heatmap_data
    
    def save_legends_and_spatial_data(self, filename_prefix: str):
        """Save all legends, legend agents, and spatial data to JSON files."""
        # Save legend data
        legend_data = {}
        
        for legend_id, legend in self.legends.items():
            # Convert legend to dict
            legend_dict = asdict(legend)
            
            # Add agents associated with this legend
            legend_agents = []
            for agent_id, agent in self.legend_agents.items():
                if agent.legend_id == legend_id:
                    agent_dict = agent.to_dict()
                    legend_agents.append(agent_dict)
            
            legend_dict["agents"] = legend_agents
            legend_data[legend_id] = legend_dict
        
        with open(f"{filename_prefix}_legends.json", 'w') as f:
            json.dump(legend_data, f, indent=2)
        
        logger.info(f"Saved {len(self.legends)} legends to {filename_prefix}_legends.json")
        
        # Save spatial database
        self.spatial_db.save_to_file(f"{filename_prefix}_spatial_memory.json")
        
        # Generate and save heatmap data for each legend
        heatmap_data = {}
        for legend_id in self.legends:
            legend_heatmap = self.generate_spatial_heatmap_for_legend(legend_id)
            if legend_heatmap:
                heatmap_data[legend_id] = legend_heatmap
        
        with open(f"{filename_prefix}_heatmaps.json", 'w') as f:
            json.dump(heatmap_data, f, indent=2)
        
        logger.info(f"Saved heatmap data for {len(heatmap_data)} legends to {filename_prefix}_heatmaps.json")

# ======================================================
# Utility Functions
# ======================================================

def create_spatial_point_from_node(node):
    """Create a SpatialPoint from a graph node."""
    return SpatialPoint(
        x=node.x,
        y=node.y,
        z=node.z,
        realm=getattr(node, "realm", None),
        node_id=getattr(node, "node_id", None)
    )

def create_spatial_event_from_legend_event(event, legend_id=None):
    """Create a SpatialEvent from a legend event."""
    if "coordinates" not in event:
        return None
    
    coords = event["coordinates"]
    
    return SpatialEvent(
        event_id=f"event-{uuid.uuid4()}",
        event_type=event.get("type", "unknown"),
        coordinates=SpatialPoint(
            x=coords[0],
            y=coords[1],
            z=coords[2],
            realm=event.get("realm")
        ),
        timestamp=event.get("timestamp", datetime.datetime.now().isoformat()),
        cycle_id=event.get("cycle_id", 0),
        agent_id=event.get("agent_id", "unknown"),
        description=event.get("description", ""),
        importance=event.get("importance", 0.5),
        tags=event.get("tags", []),
        legend_id=legend_id
    )

def connect_memory_fragments_to_spatial_events(memory_fragments, spatial_db):
    """Link memory fragments to spatial events based on coordinates."""
    for fragment in memory_fragments:
        if "coordinates" not in fragment:
            continue
        
        # Create a spatial event for this memory
        event = SpatialEvent(
            event_id=f"event-{uuid.uuid4()}",
            event_type="memory_fragment",
            coordinates=SpatialPoint(
                x=fragment["coordinates"][0],
                y=fragment["coordinates"][1],
                z=fragment["coordinates"][2],
                realm=fragment.get("realm")
            ),
            timestamp=fragment.get("timestamp", datetime.datetime.now().isoformat()),
            cycle_id=fragment.get("cycle_id", 0),
            agent_id=fragment.get("agent_id", "unknown"),
            description=f"Memory fragment: {fragment.get('content', '')[:100]}...",
            importance=fragment.get("importance", 0.5),
            tags=fragment.get("tags", []),
            memory_fragment_id=fragment.get("fragment_id")
        )
        
        spatial_db.add_event(event)

# ======================================================
# Main Simulation Function
# ======================================================

def run_integrated_simulation():
    """Run a simulation using the integrated legend and spatial memory system."""
    # Import other modules
    from dreamweaver_graph import DreamweaverGraph
    from vector_memory_db import VectorMemoryDB
    
    # Setup basic components
    realms = [
        "Celestial Palace", "Crystal Gardens", "Obsidian Wastes", "Liminal Library",
        "Verdant Visions", "Twilight Terrace", "Mythic Meridian", "Dream Nexus"
    ]
    
    # Sample echo prompts
    echo_prompts = {
        "The Broken Mirror": "You are The Broken Mirror—a shattered reflection of what once was whole.",
        "The Hollow Choir": "You are The Hollow Choir—a spectral assembly of voices united in a shared, mournful refrain.",
        "The Unfinished": "You are The Unfinished—a story still in the making, an echo of potential and perpetual evolution.",
        "The Eversong": "You are The Eversong—a perpetual melody that weaves through the corridors of the DreamWeaver."
    }
    
    # Sample wisp data
    wisp_data = {
        "Zephyr": {
            "element": "Air",
            "personality": "Calm yet unpredictable"
        },
        "Cascade": {
            "element": "Water",
            "personality": "Dynamic and fluid"
        },
        "Flare": {
            "element": "Fire",
            "personality": "Aggressive and volatile"
        }
    }
    
    # Create the world and memory system
    dreamweaver_graph = DreamweaverGraph(realms, seed=42)
    memory_db = VectorMemoryDB()
    
    # Create the integrated legend system
    legend_system = IntegratedLegendSystem(memory_db, dreamweaver_graph, echo_prompts, wisp_data)
    
    # Create a rune origin legend
    legend, agent = legend_system.create_rune_origin_legend(cycle_id=1)
    logger.info(f"Created legend: {legend.title}")
    logger.info(f"Created protagonist: {agent.name}")
    
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
        logger.info(f"\nStep {step+1}: Agent {agent.name} at {agent.current_node_id}")
        
        # Decide the agent's next action
        decision = legend_system.advance_legend_agent(agent, cycle_id, agent.current_node_id)
        logger.info(f"Decision: {decision}")
        
        # Process the action and generate events
        event = legend_system.process_legend_agent_action(agent, decision, cycle_id)
        logger.info(f"Event: {event['type']} - {event['description']}")
        
        # Generate a memory fragment
        memory = legend_system.generate_memory_fragment(agent, event, cycle_id)
        logger.info(f"Memory generated: {memory['fragment_id']}")
        
        # Print milestone status
        legend = legend_system.legends.get(agent.legend_id)
        if legend:
            completed = sum(1 for m in legend.milestones if m.completed)
            total = len(legend.milestones)
            logger.info(f"Milestone progress: {completed}/{total}")
        
        # Check if all milestones are completed
        if legend and legend.completed:
            logger.info(f"Legend completed after {step+1} steps!")
            break
    
    # Finalize the legend narrative
    narrative = legend_system.finalize_legend_narrative(legend.id)
    logger.info(f"\nFinal Legend Narrative:\n{narrative}")
    
    # Generate heatmap
    heatmap_data = legend_system.generate_spatial_heatmap_for_legend(legend.id)
    logger.info(f"Generated heatmap with {len(heatmap_data['points'])} points")
    
    # Save all data
    legend_system.save_legends_and_spatial_data("rune_origin_simulation")
    logger.info("Simulation data saved")

if __name__ == "__main__":
    run_integrated_simulation()