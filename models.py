"""
models.py - Core intelligence and spatial mathematics for Lumina v3 PRODUCTION
FIXED: 12 o'clock dead zone using math.floor instead of int
"""
from dataclasses import dataclass, field
from typing import Optional, Tuple
import numpy as np
import math
from enum import Enum


class ClockPosition(Enum):
    """Clock-face positions for intuitive directional guidance."""
    TWELVE = "12 o'clock (straight ahead)"
    ONE = "1 o'clock (slightly right)"
    TWO = "2 o'clock (right front)"
    THREE = "3 o'clock (directly right)"
    FOUR = "4 o'clock (right back)"
    FIVE = "5 o'clock (slightly behind right)"
    SIX = "6 o'clock (directly behind)"
    SEVEN = "7 o'clock (slightly behind left)"
    EIGHT = "8 o'clock (left back)"
    NINE = "9 o'clock (directly left)"
    TEN = "10 o'clock (left front)"
    ELEVEN = "11 o'clock (slightly left)"


@dataclass
class Coordinates3D:
    """3D spatial coordinates with comparison and distance methods.
    
    Attributes:
        x: Position along X-axis (meters)
        y: Position along Y-axis (meters)
        z: Position along Z-axis (meters, typically height)
    """
    x: float
    y: float
    z: float
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for vector operations."""
        return np.array([self.x, self.y, self.z], dtype=np.float64)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for database storage."""
        return {'x': float(self.x), 'y': float(self.y), 'z': float(self.z)}
    
    def distance_to(self, other: 'Coordinates3D') -> float:
        """Calculate Euclidean distance to another coordinate.
        
        Uses the formula: √((x₂-x₁)² + (y₂-y₁)² + (z₂-z₁)²)
        
        Args:
            other: Target coordinates
            
        Returns:
            Distance in meters
        """
        return np.linalg.norm(self.to_array() - other.to_array())
    
    def horizontal_distance_to(self, other: 'Coordinates3D') -> float:
        """Calculate horizontal distance ignoring Z-axis.
        
        Args:
            other: Target coordinates
            
        Returns:
            Horizontal distance in meters
        """
        dx = self.x - other.x
        dy = self.y - other.y
        return math.sqrt(dx**2 + dy**2)
    
    def is_close_to(self, other: 'Coordinates3D', threshold: float = 0.3) -> bool:
        """Check if coordinates are within threshold distance.
        
        Args:
            other: Coordinates to compare
            threshold: Maximum distance to consider "close" (meters)
            
        Returns:
            True if within threshold distance
        """
        return self.distance_to(other) < threshold


@dataclass
class Pose6DOF:
    """6 Degree-of-Freedom pose (position + orientation).
    
    Attributes:
        coords: 3D position coordinates
        roll: Rotation around X-axis (radians)
        pitch: Rotation around Y-axis (radians)
        yaw: Rotation around Z-axis (radians, heading direction)
    """
    coords: Coordinates3D
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    
    def rotation_difference(self, other: 'Pose6DOF') -> float:
        """Calculate angular difference in yaw (heading).
        
        Args:
            other: Pose to compare
            
        Returns:
            Absolute angular difference in degrees
        """
        diff = abs(self.yaw - other.yaw)
        # Normalize to [0, 180]
        if diff > math.pi:
            diff = 2 * math.pi - diff
        return math.degrees(diff)
    
    def has_significant_change(
        self, 
        other: 'Pose6DOF',
        position_threshold: float = 0.5,
        rotation_threshold: float = 15.0
    ) -> bool:
        """Check if pose has changed significantly.
        
        Args:
            other: Previous pose
            position_threshold: Minimum movement in meters
            rotation_threshold: Minimum rotation in degrees
            
        Returns:
            True if moved or rotated beyond thresholds
        """
        moved = self.coords.distance_to(other.coords) > position_threshold
        rotated = self.rotation_difference(other) > rotation_threshold
        return moved or rotated


@dataclass
class ObjectMemory:
    """Represents a stored object in spatial memory.
    
    Attributes:
        object_id: Unique identifier
        object_label: Human-readable description
        coords: 3D coordinates where object was observed
        timestamp: Unix timestamp of observation
        embedding: 512-dim CLIP embedding vector
        confidence: Detection confidence score (0-1)
        last_seen: Timestamp of last verification
    """
    object_id: str
    object_label: str
    coords: Coordinates3D
    timestamp: int
    embedding: Optional[np.ndarray] = None
    confidence: float = 1.0
    last_seen: Optional[int] = None
    
    def __post_init__(self):
        if self.last_seen is None:
            self.last_seen = self.timestamp
    
    def to_payload(self) -> dict:
        """Convert to Qdrant payload format."""
        return {
            'object_label': self.object_label,
            'x': self.coords.x,
            'y': self.coords.y,
            'z': self.coords.z,
            'timestamp': self.timestamp,
            'confidence': self.confidence,
            'last_seen': self.last_seen
        }
    
    def is_stale(self, current_time: int, max_age_seconds: int = 3600) -> bool:
        """Check if memory is outdated.
        
        Args:
            current_time: Current Unix timestamp
            max_age_seconds: Maximum age before considering stale
            
        Returns:
            True if object hasn't been seen recently
        """
        return (current_time - self.last_seen) > max_age_seconds


@dataclass
class SpatialQueryResult:
    """Result from a spatial memory search with navigation guidance.
    
    Attributes:
        object_id: Unique identifier
        object_label: Description of found object
        coords: Object's 3D coordinates
        similarity_score: Semantic similarity (0-1)
        distance: 3D Euclidean distance (meters)
        horizontal_distance: Ground distance ignoring height (meters)
        bearing: Horizontal angle in radians (-π to π)
        elevation: Vertical angle in radians
        clock_position: Clock-face direction
    """
    object_id: str
    object_label: str
    coords: Coordinates3D
    similarity_score: float
    distance: float
    horizontal_distance: float
    bearing: float
    elevation: float
    clock_position: ClockPosition
    
    def guidance_message(self) -> str:
        """Generate human-readable navigation guidance.
        
        Returns:
            Formatted guidance string with confidence
        """
        confidence_pct = int(self.similarity_score * 100)
        
        return (
            f"Object found at {self.clock_position.value}, "
            f"{self.horizontal_distance:.1f} meters away. "
            f"Confidence: {confidence_pct}%"
        )


class SpatialMath:
    """Utility class for spatial calculations and transformations."""
    
    @staticmethod
    def euclidean_distance_3d(coord1: Coordinates3D, coord2: Coordinates3D) -> float:
        """Calculate 3D Euclidean distance between two points.
        
        Formula: √((x₂-x₁)² + (y₂-y₁)² + (z₂-z₁)²)
        
        Args:
            coord1: First coordinate
            coord2: Second coordinate
            
        Returns:
            Distance in meters
        """
        dx = coord2.x - coord1.x
        dy = coord2.y - coord1.y
        dz = coord2.z - coord1.z
        return math.sqrt(dx**2 + dy**2 + dz**2)
    
    @staticmethod
    def calculate_bearing(
        from_pose: Pose6DOF,
        to_coords: Coordinates3D
    ) -> Tuple[float, float]:
        """Calculate relative bearing from observer to target.
        
        Uses Y-axis as forward (12 o'clock).
        
        Args:
            from_pose: Observer's 6DOF pose
            to_coords: Target coordinates
            
        Returns:
            Tuple of (horizontal_angle, vertical_angle) in radians
        """
        # Calculate displacement vector
        dx = to_coords.x - from_pose.coords.x
        dy = to_coords.y - from_pose.coords.y
        dz = to_coords.z - from_pose.coords.z
        
        # Horizontal angle in world frame
        # Using atan2(dx, dy) so that (0, 1) = 0° (forward/north)
        # This aligns Y-axis as "forward" (12 o'clock)
        angle_world = math.atan2(dx, dy)
        
        # Adjust for observer's heading (yaw)
        relative_bearing = angle_world - from_pose.yaw
        
        # Normalize to [-π, π]
        relative_bearing = math.atan2(
            math.sin(relative_bearing),
            math.cos(relative_bearing)
        )
        
        # Vertical angle (elevation)
        horizontal_dist = math.sqrt(dx**2 + dy**2)
        elevation = math.atan2(dz, horizontal_dist) if horizontal_dist > 0 else 0.0
        
        return relative_bearing, elevation
    
    @staticmethod
    def bearing_to_clock_position(bearing: float) -> ClockPosition:
        """Convert bearing angle to clock-face position.
        
        FIXED: Uses math.floor instead of int to handle negative numbers correctly.
        
        12 o'clock is straight ahead (0°)
        3 o'clock is right (90°)
        6 o'clock is behind (180°)
        9 o'clock is left (-90° or 270°)
        
        Args:
            bearing: Horizontal angle in radians (-π to π)
            
        Returns:
            ClockPosition enum
        """
        # Convert to degrees and normalize to 0-360
        degrees = math.degrees(bearing)
        if degrees < 0:
            degrees += 360
        
        # Map to clock positions (each hour is 30 degrees)
        # 0° = 12, 30° = 1, 60° = 2, etc.
        clock_positions = [
            ClockPosition.TWELVE,   # 345-15°
            ClockPosition.ONE,      # 15-45°
            ClockPosition.TWO,      # 45-75°
            ClockPosition.THREE,    # 75-105°
            ClockPosition.FOUR,     # 105-135°
            ClockPosition.FIVE,     # 135-165°
            ClockPosition.SIX,      # 165-195°
            ClockPosition.SEVEN,    # 195-225°
            ClockPosition.EIGHT,    # 225-255°
            ClockPosition.NINE,     # 255-285°
            ClockPosition.TEN,      # 285-315°
            ClockPosition.ELEVEN    # 315-345°
        ]
        
        # FIXED: Use math.floor instead of int for correct negative handling
        hour_index = math.floor((degrees + 15) / 30) % 12
        return clock_positions[hour_index]
    
    @staticmethod
    def calculate_navigation_metrics(
        from_pose: Pose6DOF,
        to_coords: Coordinates3D
    ) -> Tuple[float, float, float, float, ClockPosition]:
        """Calculate comprehensive navigation metrics.
        
        Args:
            from_pose: Observer's current pose
            to_coords: Target coordinates
            
        Returns:
            Tuple of (3d_distance, horizontal_distance, bearing, elevation, clock_pos)
        """
        # 3D Euclidean distance
        distance_3d = SpatialMath.euclidean_distance_3d(
            from_pose.coords,
            to_coords
        )
        
        # Horizontal distance
        distance_horiz = from_pose.coords.horizontal_distance_to(to_coords)
        
        # Bearing and elevation
        bearing, elevation = SpatialMath.calculate_bearing(from_pose, to_coords)
        
        # Clock position
        clock_pos = SpatialMath.bearing_to_clock_position(bearing)
        
        return distance_3d, distance_horiz, bearing, elevation, clock_pos