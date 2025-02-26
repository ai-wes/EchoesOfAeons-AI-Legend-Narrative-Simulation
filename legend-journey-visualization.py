#!/usr/bin/env python3
"""
DreamWeaver Visualization Component

This module provides visualization tools for rendering legend journeys, 
spatial heatmaps, and event timelines in interactive 3D and 2D formats.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple

class DreamweaverVisualizer:
    """Generates visualizations for legend data and spatial memory."""
    
    def __init__(self, legends_file: str, spatial_memory_file: str, heatmaps_file: Optional[str] = None):
        """
        Initialize the visualizer with data files.
        
        Args:
            legends_file: Path to JSON file containing legend data
            spatial_memory_file: Path to JSON file containing spatial memory data
            heatmaps_file: Optional path to JSON file containing heatmap data
        """
        # Load data
        with open(legends_file, 'r') as f:
            self.legends_data = json.load(f)
        
        with open(spatial_memory_file, 'r') as f:
            self.spatial_data = json.load(f)
        
        if heatmaps_file:
            with open(heatmaps_file, 'r') as f:
                self.heatmaps_data = json.load(f)
        else:
            self.heatmaps_data = {}
        
        # Process data for visualization
        self.events_df = self._create_events_dataframe()
        self.journeys_df = self._create_journeys_dataframe()
    
    def _create_events_dataframe(self) -> pd.DataFrame:
        """Create a pandas DataFrame from the events data for easier visualization."""
        events = []
        
        for event_id, event_data in self.spatial_data.get("events", {}).items():
            # Extract coordinates
            if isinstance(event_data.get("coordinates"), list):
                coords = event_data["coordinates"]
            elif isinstance(event_data.get("coordinates"), dict):
                coords = [
                    event_data["coordinates"].get("x", 0),
                    event_data["coordinates"].get("y", 0),
                    event_data["coordinates"].get("z", 0)
                ]
            else:
                continue
            
            # Create event record
            events.append({
                "event_id": event_id,
                "type": event_data.get("event_type", "unknown"),
                "x": coords[0],
                "y": coords[1],
                "z": coords[2],
                "realm": event_data.get("realm", "unknown"),
                "agent_id": event_data.get("agent_id", "unknown"),
                "legend_id": event_data.get("legend_id"),
                "importance": event_data.get("importance", 0.5),
                "timestamp": event_data.get("timestamp", ""),
                "description": event_data.get("description", ""),
                "memory_fragment_id": event_data.get("memory_fragment_id")
            })
        
        return pd.DataFrame(events)
    
    def _create_journeys_dataframe(self) -> pd.DataFrame:
        """Create a pandas DataFrame from the journeys data for easier visualization."""
        journey_segments = []
        
        for journey_id, journey_data in self.spatial_data.get("journeys", {}).items():
            for segment in journey_data.get("segments", []):
                # Extract start and end points
                if "start_point" in segment:
                    if isinstance(segment["start_point"], dict):
                        start_x = segment["start_point"].get("x", 0)
                        start_y = segment["start_point"].get("y", 0)
                        start_z = segment["start_point"].get("z", 0)
                        start_realm = segment["start_point"].get("realm", "unknown")
                    else:
                        continue
                else:
                    continue
                
                if "end_point" in segment:
                    if isinstance(segment["end_point"], dict):
                        end_x = segment["end_point"].get("x", 0)
                        end_y = segment["end_point"].get("y", 0)
                        end_z = segment["end_point"].get("z", 0)
                        end_realm = segment["end_point"].get("realm", "unknown")
                    else:
                        continue
                else:
                    continue
                
                # Create journey segment record
                journey_segments.append({
                    "journey_id": journey_id,
                    "segment_id": segment.get("segment_id", "unknown"),
                    "agent_id": segment.get("agent_id", "unknown"),
                    "start_x": start_x,
                    "start_y": start_y,
                    "start_z": start_z,
                    "end_x": end_x,
                    "end_y": end_y,
                    "end_z": end_z,
                    "start_realm": start_realm,
                    "end_realm": end_realm,
                    "distance": segment.get("distance", 0.0),
                    "significance": segment.get("significance", 0.0),
                    "cycle_id": segment.get("cycle_id", 0),
                    "start_time": segment.get("start_time", ""),
                    "end_time": segment.get("end_time", ""),
                    "legend_id": journey_data.get("legend_id")
                })
        
        return pd.DataFrame(journey_segments)
    
    def plot_legend_journey_3d(self, legend_id: str, save_path: Optional[str] = None):
        """
        Generate a 3D plot of a legend's journey through the DreamWeaver.
        
        Args:
            legend_id: ID of the legend to visualize
            save_path: Optional path to save the plot as an image
        """
        # Filter data for this legend
        legend_journey_df = self.journeys_df[self.journeys_df["legend_id"] == legend_id]
        legend_events_df = self.events_df[self.events_df["legend_id"] == legend_id]
        
        if legend_journey_df.empty:
            print(f"No journey data found for legend {legend_id}")
            return
        
        # Get legend title
        legend_title = "Legend Journey"
        if legend_id in self.legends_data:
            legend_title = self.legends_data[legend_id].get("title", "Legend Journey")
        
        # Create 3D plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot journey segments as lines
        for _, segment in legend_journey_df.iterrows():
            ax.plot([segment.start_x, segment.end_x],
                    [segment.start_y, segment.end_y],
                    [segment.start_z, segment.end_z],
                    'b-', alpha=0.6, linewidth=1)
        
        # Plot significant events as points
        if not legend_events_df.empty:
            # Filter for high importance events
            significant_events = legend_events_df[legend_events_df["importance"] > 0.7]
            
            # Define colors based on event type
            event_colors = {
                "echo_encounter": "purple",
                "wisp_encounter": "green",
                "strange_energy_detection": "orange",
                "pattern_recognition": "red",
                "personal_sacrifice": "black",
                "first_rune_discovery": "yellow",
                "memory_creation": "cyan"
            }
            
            # Group by event type
            for event_type, group in significant_events.groupby("type"):
                color = event_colors.get(event_type, "gray")
                ax.scatter(group.x, group.y, group.z, 
                          c=color, s=50*group.importance, alpha=0.8,
                          label=event_type)
        
        # Set labels and title
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_zlabel('Z Coordinate')
        ax.set_title(f"{legend_title} - Journey Map")
        ax.legend()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.tight_layout()
        plt.show()
    
    def create_interactive_journey_map(self, legend_id: str, save_path: Optional[str] = None):
        """
        Create an interactive 3D journey map using Plotly.
        
        Args:
            legend_id: ID of the legend to visualize
            save_path: Optional path to save the HTML file
        """
        # Filter data for this legend
        legend_journey_df = self.journeys_df[self.journeys_df["legend_id"] == legend_id]
        legend_events_df = self.events_df[self.events_df["legend_id"] == legend_id]
        
        if legend_journey_df.empty:
            print(f"No journey data found for legend {legend_id}")
            return
        
        # Get legend title
        legend_title = "Legend Journey"
        if legend_id in self.legends_data:
            legend_title = self.legends_data[legend_id].get("title", "Legend Journey")
        
        # Create figure
        fig = go.Figure()
        
        # Plot journey segments as lines
        for _, segment in legend_journey_df.iterrows():
            fig.add_trace(go.Scatter3d(
                x=[segment.start_x, segment.end_x],
                y=[segment.start_y, segment.end_y],
                z=[segment.start_z, segment.end_z],
                mode='lines',
                line=dict(color='royalblue', width=2),
                opacity=0.7,
                name=f"Journey Segment",
                hovertext=f"Agent: {segment.agent_id}<br>Distance: {segment.distance:.2f}<br>Significance: {segment.significance:.2f}"
            ))
        
        # Add markers for start and end points
        fig.add_trace(go.Scatter3d(
            x=[legend_journey_df.start_x.iloc[0]],
            y=[legend_journey_df.start_y.iloc[0]],
            z=[legend_journey_df.start_z.iloc[0]],
            mode='markers',
            marker=dict(size=10, color='green', symbol='circle'),
            name='Journey Start'
        ))
        
        fig.add_trace(go.Scatter3d(
            x=[legend_journey_df.end_x.iloc[-1]],
            y=[legend_journey_df.end_y.iloc[-1]],
            z=[legend_journey_df.end_z.iloc[-1]],
            mode='markers',
            marker=dict(size=10, color='red', symbol='circle'),
            name='Current Position'
        ))
        
        # Plot significant events
        if not legend_events_df.empty:
            # Filter for high importance events
            significant_events = legend_events_df[legend_events_df["importance"] > 0.7]
            
            # Define colors based on event type
            event_colors = {
                "echo_encounter": "purple",
                "wisp_encounter": "green",
                "strange_energy_detection": "orange",
                "pattern_recognition": "red",
                "personal_sacrifice": "black",
                "first_rune_discovery": "gold",
                "memory_creation": "cyan"
            }
            
            # Group by event type
            for event_type, group in significant_events.groupby("type"):
                color = event_colors.get(event_type, "gray")
                
                fig.add_trace(go.Scatter3d(
                    x=group.x,
                    y=group.y,
                    z=group.z,
                    mode='markers',
                    marker=dict(
                        size=group.importance * 20,
                        color=color,
                        opacity=0.8
                    ),
                    name=event_type,
                    hovertext=group.apply(lambda row: f"Type: {row.type}<br>Description: {row.description}<br>Importance: {row.importance:.2f}", axis=1)
                ))
        
# Add milestones if completed
        if legend_id in self.legends_data:
            legend = self.legends_data[legend_id]
            
            # Extract completed milestones with coordinates
            completed_milestones = []
            for milestone in legend.get("milestones", []):
                if milestone.get("completed") and milestone.get("coordinates"):
                    coords = milestone["coordinates"]
                    if isinstance(coords, list) and len(coords) >= 3:
                        completed_milestones.append({
                            "description": milestone.get("description", "Unknown milestone"),
                            "x": coords[0],
                            "y": coords[1],
                            "z": coords[2],
                            "type": milestone.get("type", "DISCOVERY")
                        })
            
            # Add milestone markers
            if completed_milestones:
                milestone_df = pd.DataFrame(completed_milestones)
                
                fig.add_trace(go.Scatter3d(
                    x=milestone_df.x,
                    y=milestone_df.y,
                    z=milestone_df.z,
                    mode='markers',
                    marker=dict(
                        size=15,
                        color='yellow',
                        symbol='diamond',
                        line=dict(color='black', width=1)
                    ),
                    name='Completed Milestones',
                    hovertext=milestone_df.description
                ))
        
        # Update layout
        fig.update_layout(
            title=f"{legend_title} - Interactive Journey Map",
            scene=dict(
                xaxis_title='X Coordinate',
                yaxis_title='Y Coordinate',
                zaxis_title='Z Coordinate'
            ),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            margin=dict(l=0, r=0, b=0, t=30)
        )
        
        # Save if requested
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def create_heatmap_visualization(self, legend_id: str, save_path: Optional[str] = None):
        """
        Create a 3D heatmap visualization of significance across the DreamWeaver for a legend.
        
        Args:
            legend_id: ID of the legend to visualize
            save_path: Optional path to save the HTML file
        """
        # Check if we have heatmap data for this legend
        if not hasattr(self, 'heatmaps_data') or legend_id not in self.heatmaps_data:
            print(f"No heatmap data found for legend {legend_id}")
            return
        
        heatmap_data = self.heatmaps_data[legend_id]
        points = heatmap_data.get("points", [])
        
        if not points:
            print(f"No heatmap points found for legend {legend_id}")
            return
        
        # Get legend title
        legend_title = "Legend Heatmap"
        if legend_id in self.legends_data:
            legend_title = self.legends_data[legend_id].get("title", "Legend Heatmap")
        
        # Convert to DataFrame
        heatmap_df = pd.DataFrame([
            {
                "x": p["position"][0],
                "y": p["position"][1],
                "z": p["position"][2],
                "value": p["value"],
                "event_count": p["event_count"],
                "realms": ", ".join(p["realms"]) if "realms" in p else "Unknown",
                "event_types": ", ".join(p["event_types"]) if "event_types" in p else "Various"
            }
            for p in points
        ])
        
        # Create figure
        fig = go.Figure()
        
        # Add heatmap points
        fig.add_trace(go.Scatter3d(
            x=heatmap_df.x,
            y=heatmap_df.y,
            z=heatmap_df.z,
            mode='markers',
            marker=dict(
                size=heatmap_df.event_count * 2,  # Size based on event count
                color=heatmap_df.value,  # Color based on importance
                colorscale='Viridis',
                opacity=0.8,
                colorbar=dict(title="Significance")
            ),
            hovertext=heatmap_df.apply(
                lambda row: f"Significance: {row.value:.2f}<br>Events: {row.event_count}<br>Realms: {row.realms}<br>Types: {row.event_types}",
                axis=1
            ),
            name='Significance Hotspots'
        ))
        
        # Update layout
        fig.update_layout(
            title=f"{legend_title} - Significance Heatmap",
            scene=dict(
                xaxis_title='X Coordinate',
                yaxis_title='Y Coordinate',
                zaxis_title='Z Coordinate'
            ),
            margin=dict(l=0, r=0, b=0, t=30)
        )
        
        # Save if requested
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def create_event_timeline(self, legend_id: str, save_path: Optional[str] = None):
        """
        Create a timeline visualization of significant events for a legend.
        
        Args:
            legend_id: ID of the legend to visualize
            save_path: Optional path to save the HTML file
        """
        # Filter events for this legend
        legend_events_df = self.events_df[self.events_df["legend_id"] == legend_id].copy()
        
        if legend_events_df.empty:
            print(f"No event data found for legend {legend_id}")
            return
        
        # Get legend title
        legend_title = "Legend Timeline"
        if legend_id in self.legends_data:
            legend_title = self.legends_data[legend_id].get("title", "Legend Timeline")
        
        # Try to convert timestamp to datetime for sorting
        # This will depend on your timestamp format
        try:
            legend_events_df["datetime"] = pd.to_datetime(legend_events_df["timestamp"])
            legend_events_df = legend_events_df.sort_values("datetime")
        except:
            # If conversion fails, keep original order
            pass
        
        # Filter for significant events
        significant_events = legend_events_df[legend_events_df["importance"] > 0.6]
        
        # Define colors based on event type
        event_colors = {
            "echo_encounter": "purple",
            "wisp_encounter": "green",
            "strange_energy_detection": "orange",
            "pattern_recognition": "red",
            "personal_sacrifice": "black",
            "first_rune_discovery": "gold",
            "memory_creation": "cyan",
            "reflection": "blue",
            "confrontation": "crimson"
        }
        
        # Create figure
        if "datetime" in significant_events.columns:
            fig = px.scatter(
                significant_events,
                x="datetime",
                y="importance",
                color="type",
                color_discrete_map=event_colors,
                size="importance",
                size_max=20,
                hover_data=["description", "realm"],
                title=f"{legend_title} - Event Timeline"
            )
        else:
            # If datetime conversion failed, use a simple index instead
            significant_events["event_index"] = range(len(significant_events))
            
            fig = px.scatter(
                significant_events,
                x="event_index",
                y="importance",
                color="type",
                color_discrete_map=event_colors,
                size="importance",
                size_max=20,
                hover_data=["description", "realm"],
                title=f"{legend_title} - Event Timeline (Event Order)"
            )
            
            fig.update_layout(xaxis_title="Event Order")
        
        # Add milestone markers if available
        if legend_id in self.legends_data:
            legend = self.legends_data[legend_id]
            
            # Extract milestone completion events
            milestone_events = []
            for milestone in legend.get("milestones", []):
                if milestone.get("completed") and milestone.get("completion_event"):
                    event = milestone["completion_event"]
                    
                    # Try to get timestamp
                    timestamp = event.get("timestamp", "")
                    try:
                        datetime_obj = pd.to_datetime(timestamp)
                    except:
                        datetime_obj = None
                    
                    milestone_events.append({
                        "description": milestone.get("description", "Unknown milestone"),
                        "type": milestone.get("type", "DISCOVERY"),
                        "timestamp": timestamp,
                        "datetime": datetime_obj,
                        "importance": 1.0  # Maximum importance for milestones
                    })
            
            if milestone_events:
                milestone_df = pd.DataFrame(milestone_events)
                
                # Add as annotations
                if "datetime" in milestone_df.columns and "datetime" in fig.data[0]:
                    for _, milestone in milestone_df.iterrows():
                        if milestone["datetime"] is not None:
                            fig.add_annotation(
                                x=milestone["datetime"],
                                y=milestone["importance"],
                                text=f"MILESTONE: {milestone['description']}",
                                showarrow=True,
                                arrowhead=1,
                                ax=0,
                                ay=-40
                            )
        
        # Update layout
        fig.update_layout(
            yaxis_title="Event Importance",
            legend_title="Event Type",
            hovermode="closest"
        )
        
        # Save if requested
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def create_realm_distribution_chart(self, legend_id: str, save_path: Optional[str] = None):
        """
        Create a chart showing the distribution of significant events across realms.
        
        Args:
            legend_id: ID of the legend to visualize
            save_path: Optional path to save the HTML file
        """
        # Filter events for this legend
        legend_events_df = self.events_df[self.events_df["legend_id"] == legend_id]
        
        if legend_events_df.empty:
            print(f"No event data found for legend {legend_id}")
            return
        
        # Get legend title
        legend_title = "Realm Distribution"
        if legend_id in self.legends_data:
            legend_title = self.legends_data[legend_id].get("title", "Realm Distribution")
        
        # Filter for significant events and group by realm
        significant_events = legend_events_df[legend_events_df["importance"] > 0.6]
        
        realm_counts = significant_events.groupby("realm").size().reset_index(name="count")
        realm_counts = realm_counts.sort_values("count", ascending=False)
        
        # Also calculate average importance by realm
        realm_importance = significant_events.groupby("realm")["importance"].mean().reset_index(name="avg_importance")
        
        # Merge the two datasets
        realm_data = pd.merge(realm_counts, realm_importance, on="realm")
        
        # Create bar chart
        fig = px.bar(
            realm_data,
            x="realm",
            y="count",
            color="avg_importance",
            color_continuous_scale="Viridis",
            labels={"count": "Number of Significant Events", "realm": "Realm", "avg_importance": "Average Importance"},
            title=f"{legend_title} - Realm Distribution of Significant Events"
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title="Realm",
            yaxis_title="Number of Significant Events",
            coloraxis_colorbar=dict(title="Avg. Importance")
        )
        
        # Save if requested
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def create_narrative_arc_visualization(self, legend_id: str, save_path: Optional[str] = None):
        """
        Create a visualization that shows the narrative arc of the legend through event significance.
        
        Args:
            legend_id: ID of the legend to visualize
            save_path: Optional path to save the HTML file
        """
        # Filter events for this legend
        legend_events_df = self.events_df[self.events_df["legend_id"] == legend_id].copy()
        
        if legend_events_df.empty:
            print(f"No event data found for legend {legend_id}")
            return
        
        # Get legend title and milestones
        legend_title = "Narrative Arc"
        milestones = []
        if legend_id in self.legends_data:
            legend_data = self.legends_data[legend_id]
            legend_title = legend_data.get("title", "Narrative Arc")
            milestones = legend_data.get("milestones", [])
        
        # Try to convert timestamp to datetime for sorting
        try:
            legend_events_df["datetime"] = pd.to_datetime(legend_events_df["timestamp"])
            legend_events_df = legend_events_df.sort_values("datetime")
        except:
            # If conversion fails, create an event index
            legend_events_df["event_index"] = range(len(legend_events_df))
        
        # Create figure
        if "datetime" in legend_events_df.columns:
            # Use rolling average to smooth the significance curve
            legend_events_df["rolling_importance"] = legend_events_df["importance"].rolling(window=5, min_periods=1).mean()
            
            fig = px.line(
                legend_events_df,
                x="datetime",
                y="rolling_importance",
                title=f"{legend_title} - Narrative Arc",
                labels={"rolling_importance": "Narrative Significance", "datetime": "Timeline"}
            )
            
            # Mark significant events
            high_significance = legend_events_df[legend_events_df["importance"] > 0.75]
            
            fig.add_trace(go.Scatter(
                x=high_significance["datetime"],
                y=high_significance["importance"],
                mode="markers",
                marker=dict(size=10, color="red"),
                name="Key Moments",
                hovertext=high_significance["description"]
            ))
            
            # Add milestone markers
            if milestones:
                completed_milestones = [m for m in milestones if m.get("completed")]
                milestone_events = []
                
                for milestone in completed_milestones:
                    if milestone.get("completion_event") and milestone["completion_event"].get("timestamp"):
                        try:
                            timestamp = milestone["completion_event"]["timestamp"]
                            datetime_obj = pd.to_datetime(timestamp)
                            
                            milestone_events.append({
                                "datetime": datetime_obj,
                                "description": milestone.get("description", "Unknown milestone"),
                                "type": milestone.get("type", "DISCOVERY")
                            })
                        except:
                            pass
                
                if milestone_events:
                    milestone_df = pd.DataFrame(milestone_events)
                    
                    fig.add_trace(go.Scatter(
                        x=milestone_df["datetime"],
                        y=[1.0] * len(milestone_df),  # Place at top of chart
                        mode="markers+text",
                        marker=dict(size=15, symbol="star", color="gold"),
                        name="Milestones",
                        text=milestone_df["type"],
                        textposition="top center",
                        hovertext=milestone_df["description"]
                    ))
        else:
            # Use event index if datetime conversion failed
            legend_events_df["rolling_importance"] = legend_events_df["importance"].rolling(window=5, min_periods=1).mean()
            
            fig = px.line(
                legend_events_df,
                x="event_index",
                y="rolling_importance",
                title=f"{legend_title} - Narrative Arc",
                labels={"rolling_importance": "Narrative Significance", "event_index": "Event Progression"}
            )
            
            # Mark significant events
            high_significance = legend_events_df[legend_events_df["importance"] > 0.75]
            
            fig.add_trace(go.Scatter(
                x=high_significance["event_index"],
                y=high_significance["importance"],
                mode="markers",
                marker=dict(size=10, color="red"),
                name="Key Moments",
                hovertext=high_significance["description"]
            ))
        
        # Add narrative arc stages
        arc_stages = ["introduction", "rising_action", "climax", "falling_action", "resolution"]
        
        # We'll divide the x-axis into 5 equal segments for the narrative arc stages
        if "datetime" in legend_events_df.columns:
            min_date = legend_events_df["datetime"].min()
            max_date = legend_events_df["datetime"].max()
            date_range = (max_date - min_date).total_seconds()
            
            for i, stage in enumerate(arc_stages):
                stage_position = min_date + pd.Timedelta(seconds=date_range * i / len(arc_stages))
                
                fig.add_vline(
                    x=stage_position,
                    line_dash="dash",
                    line_color="gray",
                    opacity=0.5
                )
                
                fig.add_annotation(
                    x=stage_position + pd.Timedelta(seconds=date_range / (2 * len(arc_stages))),
                    y=0.2,
                    text=stage.replace("_", " ").title(),
                    showarrow=False,
                    textangle=-90
                )
        else:
            max_index = legend_events_df["event_index"].max()
            
            for i, stage in enumerate(arc_stages):
                stage_position = max_index * i / len(arc_stages)
                
                fig.add_vline(
                    x=stage_position,
                    line_dash="dash",
                    line_color="gray",
                    opacity=0.5
                )
                
                fig.add_annotation(
                    x=stage_position + (max_index / (2 * len(arc_stages))),
                    y=0.2,
                    text=stage.replace("_", " ").title(),
                    showarrow=False,
                    textangle=-90
                )
        
        # Update layout
        fig.update_layout(
            hovermode="closest",
            yaxis=dict(range=[0, 1.1])  # Set y-axis range to accommodate milestone stars
        )
        
        # Save if requested
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def generate_legend_dashboard(self, legend_id: str, output_dir: str):
        """
        Generate a complete dashboard of visualizations for a legend.
        
        Args:
            legend_id: ID of the legend to visualize
            output_dir: Directory to save the visualization files
        """
        import os
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate visualizations
        print(f"Generating journey map...")
        self.create_interactive_journey_map(
            legend_id, 
            save_path=os.path.join(output_dir, f"{legend_id}_journey_map.html")
        )
        
        print(f"Generating heatmap...")
        self.create_heatmap_visualization(
            legend_id, 
            save_path=os.path.join(output_dir, f"{legend_id}_heatmap.html")
        )
        
        print(f"Generating event timeline...")
        self.create_event_timeline(
            legend_id, 
            save_path=os.path.join(output_dir, f"{legend_id}_timeline.html")
        )
        
        print(f"Generating realm distribution...")
        self.create_realm_distribution_chart(
            legend_id, 
            save_path=os.path.join(output_dir, f"{legend_id}_realm_distribution.html")
        )
        
        print(f"Generating narrative arc...")
        self.create_narrative_arc_visualization(
            legend_id, 
            save_path=os.path.join(output_dir, f"{legend_id}_narrative_arc.html")
        )
        
        # Create an index HTML file that links to all the visualizations
        legend_title = "Legend Dashboard"
        if legend_id in self.legends_data:
            legend_title = self.legends_data[legend_id].get("title", "Legend Dashboard")
        
        index_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{legend_title} Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                .card {{ border: 1px solid #ddd; border-radius: 5px; padding: 15px; margin-bottom: 20px; }}
                .card h2 {{ margin-top: 0; }}
                iframe {{ width: 100%; height: 600px; border: none; }}
            </style>
        </head>
        <body>
            <h1>{legend_title} Dashboard</h1>
            
            <div class="card">
                <h2>Journey Map</h2>
                <iframe src="{legend_id}_journey_map.html"></iframe>
            </div>
            
            <div class="card">
                <h2>Significance Heatmap</h2>
                <iframe src="{legend_id}_heatmap.html"></iframe>
            </div>
            
            <div class="card">
                <h2>Event Timeline</h2>
                <iframe src="{legend_id}_timeline.html"></iframe>
            </div>
            
            <div class="card">
                <h2>Realm Distribution</h2>
                <iframe src="{legend_id}_realm_distribution.html"></iframe>
            </div>
            
            <div class="card">
                <h2>Narrative Arc</h2>
                <iframe src="{legend_id}_narrative_arc.html"></iframe>
            </div>
        </body>
        </html>
        """
        
        with open(os.path.join(output_dir, f"{legend_id}_dashboard.html"), 'w') as f:
            f.write(index_html)
        
        print(f"Dashboard generated at {os.path.join(output_dir, f'{legend_id}_dashboard.html')}")

# Example usage
if __name__ == "__main__":
    # Paths to data files generated by the simulation
    legends_file = "rune_origin_simulation_legends.json"
    spatial_memory_file = "rune_origin_simulation_spatial_memory.json"
    heatmaps_file = "rune_origin_simulation_heatmaps.json"
    
    # Create visualizer
    visualizer = DreamweaverVisualizer(legends_file, spatial_memory_file, heatmaps_file)
    
    # Get first legend ID
    legend_id = list(visualizer.legends_data.keys())[0]
    
    # Generate dashboard
    visualizer.generate_legend_dashboard(legend_id, "visualizations")