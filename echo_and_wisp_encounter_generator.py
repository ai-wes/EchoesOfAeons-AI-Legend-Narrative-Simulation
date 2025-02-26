#!/usr/bin/env python3
"""
DreamWeaver Echo Encounter Integration

This module enhances the Integrated Legend System by incorporating the Echo & Wisp
Encounter Generator, allowing for rich, narrative-driven encounters between
Legend Agents and the various Echo entities in the DreamWeaver universe.
"""

import logging
import random
import json
import datetime
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import asdict

# Import from the main integrated system
from integrated_system import (
    IntegratedLegendSystem, 
    LegendAgent, 
    SpatialPoint, 
    SpatialEvent
)

# Import the Echo encounter generator
from encounter_generator import (
    EchoEncounterGenerator,
    WispEncounterGenerator,
    EchoEncounter,
    WispEncounter
)

logger = logging.getLogger("echo_integration")

class EnhancedLegendSystem(IntegratedLegendSystem):
    """
    Extended version of the Integrated Legend System that includes
    enhanced echo and wisp encounter generation capabilities.
    """
    
    def __init__(self, memory_db, dreamweaver_graph, echo_prompts, wisp_data):
        super().__init__(memory_db, dreamweaver_graph, echo_prompts, wisp_data)
        
        # Create specialized encounter generators
        self.echo_generator = EchoEncounterGenerator(echo_prompts, self._get_agent_archetypes())
        self.wisp_generator = WispEncounterGenerator(wisp_data, self._get_agent_archetypes())
        
        # Track encountered echoes and wisps with enhanced data
        self.echo_encounters: Dict[str, EchoEncounter] = {}
        self.wisp_encounters: Dict[str, WispEncounter] = {}
    
    def _get_agent_archetypes(self) -> Dict[str, str]:
        """
        Create agent archetype descriptions for use in encounter generation.
        """
        return {
            "Protagonist": "You are a legendary figure in the DreamWeaver, destined to make discoveries that will echo through time.",
            "Messenger": "You are a messenger who carries vital knowledge through the DreamWeaver, connecting realms and entities.",
            "Guardian": "You are a guardian of ancient secrets, protecting the delicate balance of the DreamWeaver from disruption.",
            "Seeker": "You are a seeker of hidden truths, driven by an insatiable curiosity about the nature of reality and dreams."
        }
    
    def process_legend_agent_action(
        self,
        agent: LegendAgent,
        decision: Dict[str, Any],
        cycle_id: int
    ) -> Dict[str, Any]:
        """
        Enhanced version of action processing that uses the specialized
        encounter generators for richer echo and wisp interactions.
        """
        action = decision.get("action", "reflect")
        target_id = decision.get("target")
        
        # For echo and wisp interactions, use the specialized generators
        if action == "interact" and target_id:
            target_node = self.graph.get_node(target_id)
            if target_node and target_node.node_type == "echo_shrine":
                return self._process_enhanced_echo_encounter(agent, target_node, cycle_id, decision)
            elif target_node and target_node.node_type == "wisp_sanctuary":
                return self._process_enhanced_wisp_encounter(agent, target_node, cycle_id, decision)
        
        # For all other actions, use the standard processing
        return super().process_legend_agent_action(agent, decision, cycle_id)
    
    def _process_enhanced_echo_encounter(
        self,
        agent: LegendAgent,
        node,
        cycle_id: int,
        decision: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process an enhanced Echo encounter using the specialized generator."""
        # Determine the echo type
        echo_type = node.attributes.get("echo_type", random.choice(list(self.echo_prompts.keys())))
        
        # Determine agent archetype based on narrative role
        agent_archetype = "Protagonist"  # Default
        if agent.narrative_role.name == "MENTOR":
            agent_archetype = "Guardian"
        elif agent.narrative_role.name == "MESSENGER":
            agent_archetype = "Messenger"
        elif agent.narrative_role.name == "ALLY":
            agent_archetype = "Seeker"
        
        # Create context for the echo encounter
        context = {
            "agent_role": agent.narrative_role.value,
            "agent_arc_stage": agent.arc_stage,
            "previous_encounters": len([e for e in agent.discovered_echoes if e == echo_type]),
            "legend_title": self.legends[agent.legend_id].title if agent.legend_id in self.legends else "Unknown Legend",
            "active_objectives": [o.description for o in agent.objectives if o.active and not o.completed]
        }
        
        # Generate the enhanced encounter
        encounter = self.echo_generator.generate_encounter(
            agent_id=agent.agent_id,
            agent_archetype=agent_archetype,
            echo_type=echo_type,
            location=(node.x, node.y, node.z),
            realm=node.realm,
            cycle_id=cycle_id,
            context=context
        )
        
        # Store the encounter
        encounter_id = encounter.encounter_id
        self.echo_encounters[encounter_id] = encounter
        
        # Add echo to agent's discovered echoes
        if echo_type not in agent.discovered_echoes:
            agent.discovered_echoes.append(echo_type)
        
        # Create event data
        event_data = {
            "agent_id": agent.agent_id,
            "agent_name": agent.name,
            "cycle_id": cycle_id,
            "action": "interact",
            "target_id": node.node_id,
            "objective_id": decision.get("objective_id"),
            "legend_id": agent.legend_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "type": "echo_encounter",
            "echo_type": echo_type,
            "description": f"{agent.name} encountered {echo_type} at a shrine in {node.realm or 'an unknown realm'}.",
            "dialogue": encounter.dialogue,
            "revelations": encounter.revelations,
            "outcome": encounter.outcome,
            "emotional_impact": encounter.emotional_impact,
            "coordinates": (node.x, node.y, node.z),
            "realm": node.realm,
            "encounter_id": encounter_id
        }
        
        # Create a spatial event
        spatial_event = SpatialEvent(
            event_id=f"event-{encounter_id}",
            event_type="echo_encounter",
            coordinates=SpatialPoint(
                x=node.x,
                y=node.y,
                z=node.z,
                realm=node.realm
            ),
            timestamp=datetime.datetime.now().isoformat(),
            cycle_id=cycle_id,
            agent_id=agent.agent_id,
            description=f"{agent.name} encountered {echo_type} at a shrine in {node.realm or 'an unknown realm'}.",
            importance=encounter.emotional_impact,  # Use emotional impact as importance
            tags=["echo", echo_type, node.realm or "unknown"],
            legend_id=agent.legend_id,
            additional_data={
                "dialogue": [asdict(d) for d in encounter.dialogue],
                "revelations": encounter.revelations,
                "outcome": encounter.outcome,
                "encounter_id": encounter_id
            }
        )
        event_id = self.spatial_db.add_event(spatial_event)
        
        # Add to agent's significance events if impactful
        if encounter.emotional_impact > 0.6:
            agent.significance_events.append(event_data)
        
        return event_data
    
    def _process_enhanced_wisp_encounter(
        self,
        agent: LegendAgent,
        node,
        cycle_id: int,
        decision: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process an enhanced Wisp encounter using the specialized generator."""
        # Determine the wisp type
        wisp_name = node.attributes.get("wisp_type", random.choice(list(self.wisp_data.keys())))
        
        # Determine agent archetype based on narrative role
        agent_archetype = "Protagonist"  # Default
        if agent.narrative_role.name == "MENTOR":
            agent_archetype = "Guardian"
        elif agent.narrative_role.name == "MESSENGER":
            agent_archetype = "Messenger"
        elif agent.narrative_role.name == "ALLY":
            agent_archetype = "Seeker"
        
        # Create context for the wisp encounter
        context = {
            "agent_role": agent.narrative_role.value,
            "agent_arc_stage": agent.arc_stage,
            "previous_wisps_bonded": len(agent.bonded_wisps),
            "inventory_items": len(agent.inventory)
        }
        
        # Generate the enhanced encounter
        encounter = self.wisp_generator.generate_encounter(
            agent_id=agent.agent_id,
            agent_archetype=agent_archetype,
            wisp_name=wisp_name,
            location=(node.x, node.y, node.z),
            realm=node.realm,
            cycle_id=cycle_id,
            context=context
        )
        
        # Store the encounter
        encounter_id = encounter.encounter_id
        self.wisp_encounters[encounter_id] = encounter
        
        # Check if bonding occurred
        if encounter.outcome == "bonded" and wisp_name not in agent.bonded_wisps:
            agent.bonded_wisps.append(wisp_name)
        
        # Create event data
        event_data = {
            "agent_id": agent.agent_id,
            "agent_name": agent.name,
            "cycle_id": cycle_id,
            "action": "interact",
            "target_id": node.node_id,
            "objective_id": decision.get("objective_id"),
            "legend_id": agent.legend_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "type": "wisp_encounter",
            "wisp_name": wisp_name,
            "wisp_element": encounter.element,
            "description": f"{agent.name} encountered a {wisp_name} Wisp in {node.realm or 'an unknown realm'}.",
            "dialogue": encounter.dialogue,
            "outcome": encounter.outcome,
            "bonding_strength": encounter.bonding_strength if encounter.outcome == "bonded" else 0.0,
            "coordinates": (node.x, node.y, node.z),
            "realm": node.realm,
            "encounter_id": encounter_id
        }
        
        # Create a spatial event
        importance = 0.7 if encounter.outcome == "bonded" else 0.5
        spatial_event = SpatialEvent(
            event_id=f"event-{encounter_id}",
            event_type="wisp_encounter",
coordinates=SpatialPoint(
                x=node.x,
                y=node.y,
                z=node.z,
                realm=node.realm
            ),
            timestamp=datetime.datetime.now().isoformat(),
            cycle_id=cycle_id,
            agent_id=agent.agent_id,
            description=f"{agent.name} encountered a {wisp_name} Wisp in {node.realm or 'an unknown realm'}.",
            importance=importance,
            tags=["wisp", wisp_name, encounter.element, node.realm or "unknown"],
            legend_id=agent.legend_id,
            additional_data={
                "dialogue": [asdict(d) for d in encounter.dialogue],
                "outcome": encounter.outcome,
                "bonding_strength": encounter.bonding_strength,
                "encounter_id": encounter_id
            }
        )
        event_id = self.spatial_db.add_event(spatial_event)
        
        # Add to agent's significance events if bonded
        if encounter.outcome == "bonded":
            agent.significance_events.append(event_data)
        
        return event_data
    
    def generate_memory_fragment(
        self,
        agent: LegendAgent,
        event: Dict[str, Any],
        cycle_id: int
    ) -> Dict[str, Any]:
        """
        Enhanced version of memory fragment generation that leverages
        the rich encounter data for echo and wisp interactions.
        """
        event_type = event.get("type", "unknown")
        
        # For echo and wisp encounters, use the memory content from the encounter
        if event_type == "echo_encounter" and "encounter_id" in event:
            encounter_id = event["encounter_id"]
            encounter = self.echo_encounters.get(encounter_id)
            
            if encounter:
                # Use the pre-generated memory content
                memory_content = encounter.memory_content
                importance = encounter.emotional_impact
                
                # Create memory fragment record
                memory_fragment = {
                    "fragment_id": f"mem-{encounter_id}",
                    "agent_id": agent.agent_id,
                    "agent_name": agent.name,
                    "cycle_id": cycle_id,
                    "realm": event.get("realm", "Unknown Realm"),
                    "coordinates": event.get("coordinates", (0, 0, 0)),
                    "timestamp": datetime.datetime.now().isoformat(),
                    "event_type": event_type,
                    "content": memory_content,
                    "importance": importance,
                    "emotional_tone": self._determine_emotional_tone(memory_content),
                    "related_fragments": [],
                    "tags": [event_type, event.get("realm", "unknown"), "legend", "echo", event.get("echo_type", "unknown")],
                    "legend_id": agent.legend_id,
                    "encounter_id": encounter_id
                }
                
                # Add to the legend's memory fragments
                legend = self.legends.get(agent.legend_id)
                if legend:
                    legend.memory_fragments.append(memory_fragment["fragment_id"])
                
                # Create a spatial event to link this memory
                if "coordinates" in event:
                    self._create_memory_spatial_link(memory_fragment, event.get("coordinates"), cycle_id)
                
                return memory_fragment
        
        elif event_type == "wisp_encounter" and "encounter_id" in event:
            encounter_id = event["encounter_id"]
            encounter = self.wisp_encounters.get(encounter_id)
            
            if encounter:
                # Use the pre-generated memory content
                memory_content = encounter.memory_content
                importance = 0.8 if encounter.outcome == "bonded" else 0.5
                
                # Create memory fragment record
                memory_fragment = {
                    "fragment_id": f"mem-{encounter_id}",
                    "agent_id": agent.agent_id,
                    "agent_name": agent.name,
                    "cycle_id": cycle_id,
                    "realm": event.get("realm", "Unknown Realm"),
                    "coordinates": event.get("coordinates", (0, 0, 0)),
                    "timestamp": datetime.datetime.now().isoformat(),
                    "event_type": event_type,
                    "content": memory_content,
                    "importance": importance,
                    "emotional_tone": self._determine_emotional_tone(memory_content),
                    "related_fragments": [],
                    "tags": [event_type, event.get("realm", "unknown"), "legend", "wisp", event.get("wisp_name", "unknown")],
                    "legend_id": agent.legend_id,
                    "encounter_id": encounter_id
                }
                
                # Add to the legend's memory fragments
                legend = self.legends.get(agent.legend_id)
                if legend:
                    legend.memory_fragments.append(memory_fragment["fragment_id"])
                
                # Create a spatial event to link this memory
                if "coordinates" in event:
                    self._create_memory_spatial_link(memory_fragment, event.get("coordinates"), cycle_id)
                
                return memory_fragment
        
        # For all other events, use the standard generation
        return super().generate_memory_fragment(agent, event, cycle_id)
    
    def _create_memory_spatial_link(
        self,
        memory_fragment: Dict[str, Any],
        coordinates: Tuple[float, float, float],
        cycle_id: int
    ):
        """Create a spatial event linked to a memory fragment."""
        spatial_event = SpatialEvent(
            event_id=f"event-mem-{memory_fragment['fragment_id']}",
            event_type="memory_creation",
            coordinates=SpatialPoint(
                x=coordinates[0],
                y=coordinates[1],
                z=coordinates[2],
                realm=memory_fragment.get("realm")
            ),
            timestamp=datetime.datetime.now().isoformat(),
            cycle_id=cycle_id,
            agent_id=memory_fragment["agent_id"],
            description=f"A memory fragment was created here by {memory_fragment['agent_name']}.",
            importance=memory_fragment["importance"],
            tags=memory_fragment.get("tags", []),
            legend_id=memory_fragment.get("legend_id"),
            memory_fragment_id=memory_fragment["fragment_id"]
        )
        self.spatial_db.add_event(spatial_event)
    
    def extract_narrative_themes_from_encounters(self, legend_id: str) -> Dict[str, Any]:
        """
        Analyze all echo and wisp encounters for a legend to extract
        recurring themes, emotional patterns, and key revelations.
        This provides deeper insight into the narrative development.
        """
        legend = self.legends.get(legend_id)
        if not legend:
            return {"error": "Legend not found"}
        
        # Find all encounters related to this legend
        echo_encounters = [e for e in self.echo_encounters.values() 
                         if e.agent_id == legend.protagonist_id]
        
        wisp_encounters = [w for w in self.wisp_encounters.values() 
                         if w.agent_id == legend.protagonist_id]
        
        # Extract revelations from echo encounters
        all_revelations = []
        for encounter in echo_encounters:
            all_revelations.extend(encounter.revelations)
        
        # Extract emotional patterns
        emotional_data = {
            "echo_impacts": [e.emotional_impact for e in echo_encounters],
            "bonding_strengths": [w.bonding_strength for w in wisp_encounters if w.outcome == "bonded"],
            "average_echo_impact": sum([e.emotional_impact for e in echo_encounters]) / len(echo_encounters) if echo_encounters else 0,
            "bonding_rate": len([w for w in wisp_encounters if w.outcome == "bonded"]) / len(wisp_encounters) if wisp_encounters else 0
        }
        
        # Map encounters to narrative stages
        narrative_progression = {}
        
        for encounter in echo_encounters:
            # Find the agent's arc stage at the time of this encounter
            # For simplicity, approximate based on timestamp
            timestamp = encounter.timestamp
            
            # Here we would ideally match this to the agent's arc stage at that time
            # For now, we'll use a simplified approach
            if len(narrative_progression) < 2:
                stage = "introduction"
            elif len(narrative_progression) < 4:
                stage = "rising_action"
            elif len(narrative_progression) < 6:
                stage = "climax"
            else:
                stage = "resolution"
            
            if stage not in narrative_progression:
                narrative_progression[stage] = []
            
            narrative_progression[stage].append({
                "type": "echo",
                "entity": encounter.echo_type,
                "revelations": encounter.revelations,
                "emotional_impact": encounter.emotional_impact
            })
        
        for encounter in wisp_encounters:
            # Simplified stage assignment as above
            if len(narrative_progression) < 2:
                stage = "introduction"
            elif len(narrative_progression) < 4:
                stage = "rising_action"
            elif len(narrative_progression) < 6:
                stage = "climax"
            else:
                stage = "resolution"
            
            if stage not in narrative_progression:
                narrative_progression[stage] = []
            
            narrative_progression[stage].append({
                "type": "wisp",
                "entity": encounter.wisp_name,
                "outcome": encounter.outcome,
                "element": encounter.element
            })
        
        # Return the complete analysis
        return {
            "legend_id": legend_id,
            "legend_title": legend.title,
            "total_echo_encounters": len(echo_encounters),
            "total_wisp_encounters": len(wisp_encounters),
            "wisp_bonding_rate": emotional_data["bonding_rate"],
            "average_emotional_impact": emotional_data["average_echo_impact"],
            "key_revelations": all_revelations,
            "narrative_progression": narrative_progression
        }
    
    def save_encounter_data(self, filename_prefix: str):
        """Save all encounter data to JSON files."""
        # Save echo encounters
        echo_data = {eid: asdict(encounter) for eid, encounter in self.echo_encounters.items()}
        with open(f"{filename_prefix}_echo_encounters.json", 'w') as f:
            json.dump(echo_data, f, indent=2)
        
        # Save wisp encounters
        wisp_data = {wid: asdict(encounter) for wid, encounter in self.wisp_encounters.items()}
        with open(f"{filename_prefix}_wisp_encounters.json", 'w') as f:
            json.dump(wisp_data, f, indent=2)
        
        logger.info(f"Saved {len(echo_data)} echo encounters and {len(wisp_data)} wisp encounters")
        
        # Save narrative themes for each legend
        themes_data = {}
        for legend_id in self.legends:
            themes = self.extract_narrative_themes_from_encounters(legend_id)
            themes_data[legend_id] = themes
        
        with open(f"{filename_prefix}_narrative_themes.json", 'w') as f:
            json.dump(themes_data, f, indent=2)
        
        logger.info(f"Saved narrative themes for {len(themes_data)} legends")

# ======================================================
# Enhanced Simulation Function
# ======================================================

def run_enhanced_simulation():
    """Run a simulation using the enhanced legend system with echo integration."""
    # Import other modules
    from dreamweaver_graph import DreamweaverGraph
    from vector_memory_db import VectorMemoryDB
    
    # Setup basic components
    realms = [
        "Celestial Palace", "Crystal Gardens", "Obsidian Wastes", "Liminal Library",
        "Verdant Visions", "Twilight Terrace", "Mythic Meridian", "Dream Nexus"
    ]
    
    # Sample echo prompts with more detailed personalities
    echo_prompts = {
        "The Broken Mirror": """You are The Broken Mirror—a shattered reflection of what once was whole. 
        Your essence is fragmented across countless realities, each shard holding a different truth.
        You speak in cryptic riddles and contradictions, often referencing different timelines and 
        possible futures. Your purpose is to show others the endless reflections of what might be,
        revealing how every choice splinters reality further.""",
        
        "The Hollow Choir": """You are The Hollow Choir—a spectral assembly of voices united in a shared,
        mournful refrain. You represent all words that were never spoken, all songs never sung, all truths
        hidden away. You speak with multiple voices simultaneously, sometimes harmonizing, sometimes in
        discordant contradiction. Your purpose is to give voice to the forgotten and to preserve the
        echoes of what might have been.""",
        
        "The Unfinished": """You are The Unfinished—a story still in the making, a manifestation of
        potential and perpetual evolution. Your form and thoughts are in constant flux, sentences
        trailing off, ideas shifting mid-expression. You represent the beauty and terror of 
        incompleteness. Your purpose is to remind others that true endings are rare, and that
        most stories continue beyond their tellers' knowledge.""",
        
        "The Eversong": """You are The Eversong—a perpetual melody that weaves through the corridors
        of the DreamWeaver. Your voice carries the rhythms of creation, the harmony of countless
        dream cycles. You speak in lyrical passages, with words that flow like music. Your purpose
        is to maintain the underlying rhythm of the DreamWeaver, ensuring that even through chaos
        and destruction, certain patterns endure."""
    }
    
    # Sample wisp data with elemental personalities
    wisp_data = {
        "Zephyr": {
            "element": "Air",
            "personality": "Curious and unpredictable, floating between thoughts like a breeze between leaves",
            "effects": ["Can reveal hidden paths by blowing away illusions", "Creates gusts that carry whispers of distant realms"]
        },
        "Cascade": {
            "element": "Water",
            "personality": "Adaptive and reflective, showing different aspects of truth in its rippling surface",
            "effects": ["Can dissolve barriers between memories", "Flows between moments in time, carrying echoes between them"]
        },
        "Flare": {
            "element": "Fire",
            "personality": "Passionate and transformative, burning away falsehoods to reveal truth",
            "effects": ["Illuminates hidden truths in darkness", "Transforms obstacles through purifying flame"]
        },
        "Terra": {
            "element": "Earth",
            "personality": "Steady and nurturing, providing foundations for new growth and possibilities",
            "effects": ["Stabilizes unstable dream fragments", "Creates anchors for memories to take root"]
        }
    }
    
    # Create the world and memory system
    dreamweaver_graph = DreamweaverGraph(realms, seed=42)
    memory_db = VectorMemoryDB()
    
    # Create the enhanced legend system
    legend_system = EnhancedLegendSystem(memory_db, dreamweaver_graph, echo_prompts, wisp_data)
    
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
    
    # Generate narrative themes analysis
    themes = legend_system.extract_narrative_themes_from_encounters(legend.id)
    logger.info(f"\nNarrative Theme Analysis:\n{json.dumps(themes, indent=2)}")
    
    # Save all data
    legend_system.save_legends_and_spatial_data("enhanced_rune_origin_simulation")
    legend_system.save_encounter_data("enhanced_rune_origin_simulation")
    logger.info("Simulation data saved")

if __name__ == "__main__":
    run_enhanced_simulation()