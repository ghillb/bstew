"""
Detailed Resource Collection Logic for NetLogo BEE-STEWARD v2 Parity
===================================================================

Advanced resource collection system matching NetLogo's detailed physiological
constraints, proboscis-corolla compatibility, crop volume limits, and resource
depletion dynamics.
"""

from typing import Dict, List, Tuple, Any
from pydantic import BaseModel, Field
from enum import Enum
from dataclasses import dataclass



class ResourceType(Enum):
    """Types of resources that can be collected"""
    NECTAR = "nectar"
    POLLEN = "pollen"
    WATER = "water"
    PROPOLIS = "propolis"
    HONEY = "honey"


class CollectionResult(Enum):
    """Result of a resource collection attempt"""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"
    INCOMPATIBLE = "incompatible"
    DEPLETED = "depleted"
    CAPACITY_FULL = "capacity_full"


@dataclass
class FlowerCharacteristics:
    """Physical characteristics of a flower"""
    corolla_depth: float  # mm
    corolla_diameter: float  # mm
    nectar_volume: float  # μL
    sugar_concentration: float  # % (0-1)
    pollen_availability: float  # mg
    accessibility_index: float  # 0-1
    handling_time: float  # seconds
    reward_quality: float  # 0-1


@dataclass
class BeePhysiology:
    """Bee physiological characteristics affecting collection"""
    proboscis_length: float  # mm
    proboscis_diameter: float  # mm
    crop_capacity: float  # μL
    current_crop_volume: float  # μL
    pollen_basket_capacity: float  # mg
    current_pollen_load: float  # mg
    pumping_rate: float  # μL/second
    energy_level: float  # 0-100
    collection_efficiency: float  # 0-1


class ResourceExtractionModel(BaseModel):
    """Model for resource extraction efficiency"""
    
    model_config = {"validate_assignment": True}
    
    # Proboscis-corolla compatibility parameters
    optimal_depth_ratio: float = Field(default=0.8, ge=0.0, le=1.0, description="Optimal proboscis:corolla depth ratio")
    compatibility_threshold: float = Field(default=0.3, ge=0.0, le=1.0, description="Minimum compatibility threshold")
    
    # Extraction efficiency parameters
    base_extraction_rate: float = Field(default=0.5, ge=0.0, le=1.0, description="Base extraction rate")
    experience_factor: float = Field(default=0.2, ge=0.0, le=1.0, description="Experience improvement factor")
    energy_efficiency_factor: float = Field(default=0.3, ge=0.0, le=1.0, description="Energy level impact factor")
    
    # Time-based parameters
    base_handling_time: float = Field(default=3.0, ge=0.0, description="Base handling time (seconds)")
    complexity_multiplier: float = Field(default=1.5, ge=1.0, description="Complexity handling multiplier")
    
    def calculate_proboscis_compatibility(self, bee_proboscis_length: float, 
                                        corolla_depth: float,
                                        corolla_diameter: float) -> float:
        """Calculate proboscis-corolla compatibility score"""
        
        # Check if proboscis can reach nectar
        if bee_proboscis_length < corolla_depth:
            return 0.0  # Cannot reach nectar
        
        # Calculate depth ratio compatibility
        depth_ratio = bee_proboscis_length / corolla_depth if corolla_depth > 0 else 1.0
        
        # Optimal ratio is around 0.8 (80% of proboscis length)
        depth_score = 1.0 - abs(depth_ratio - self.optimal_depth_ratio)
        depth_score = max(0.0, depth_score)
        
        # Diameter compatibility (simplified)
        diameter_score = min(1.0, corolla_diameter / 2.0)  # Assume min 2mm diameter needed
        
        # Combined compatibility
        compatibility = (depth_score * 0.7 + diameter_score * 0.3)
        
        return max(0.0, compatibility)
    
    def calculate_extraction_efficiency(self, bee_physiology: BeePhysiology,
                                      flower: FlowerCharacteristics,
                                      compatibility: float,
                                      experience_level: float) -> float:
        """Calculate resource extraction efficiency"""
        
        if compatibility < self.compatibility_threshold:
            return 0.0  # Incompatible
        
        # Base efficiency from compatibility
        base_efficiency = compatibility * self.base_extraction_rate
        
        # Experience improvement
        experience_bonus = experience_level * self.experience_factor
        
        # Energy level impact
        energy_factor = (bee_physiology.energy_level / 100.0) * self.energy_efficiency_factor
        
        # Flower accessibility impact
        accessibility_factor = flower.accessibility_index * 0.2
        
        # Bee individual efficiency
        individual_efficiency = bee_physiology.collection_efficiency * 0.3
        
        # Combined efficiency
        total_efficiency = (base_efficiency + experience_bonus + energy_factor + 
                          accessibility_factor + individual_efficiency)
        
        return min(1.0, total_efficiency)
    
    def calculate_handling_time(self, flower: FlowerCharacteristics,
                              compatibility: float,
                              efficiency: float) -> float:
        """Calculate handling time for resource collection"""
        
        # Base handling time
        handling_time = self.base_handling_time
        
        # Flower-specific handling time
        if flower.handling_time > 0:
            handling_time = flower.handling_time
        
        # Compatibility adjustment (better compatibility = faster handling)
        if compatibility > 0:
            handling_time *= (2.0 - compatibility)
        
        # Efficiency adjustment (higher efficiency = faster handling)
        if efficiency > 0:
            handling_time *= (1.5 - efficiency * 0.5)
        
        # Complexity adjustment based on flower characteristics
        complexity_factor = 1.0
        if flower.corolla_depth > 8.0:  # Deep flowers are more complex
            complexity_factor *= self.complexity_multiplier
        if flower.accessibility_index < 0.5:  # Hard to access flowers
            complexity_factor *= 1.3
        
        return handling_time * complexity_factor


class CropVolumeManager(BaseModel):
    """Manager for bee crop volume and capacity limits"""
    
    model_config = {"validate_assignment": True}
    
    # Capacity parameters
    max_crop_capacity: float = Field(default=70.0, ge=0.0, description="Maximum crop capacity (μL)")
    optimal_load_factor: float = Field(default=0.85, ge=0.0, le=1.0, description="Optimal load factor")
    
    # Volume management
    nectar_concentration_factor: float = Field(default=0.8, ge=0.0, le=1.0, description="Nectar concentration factor")
    water_dilution_factor: float = Field(default=1.2, ge=1.0, description="Water dilution factor")
    
    # Physiological constraints
    flight_efficiency_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Flight efficiency threshold")
    energy_cost_multiplier: float = Field(default=1.5, ge=1.0, description="Energy cost multiplier when full")
    
    def calculate_available_capacity(self, current_volume: float, 
                                   resource_type: ResourceType) -> float:
        """Calculate available crop capacity for resource type"""
        
        available_capacity = self.max_crop_capacity - current_volume
        
        # Adjust for resource type
        if resource_type == ResourceType.NECTAR:
            # Nectar can be concentrated
            available_capacity *= self.nectar_concentration_factor
        elif resource_type == ResourceType.WATER:
            # Water takes more space
            available_capacity /= self.water_dilution_factor
        
        return max(0.0, available_capacity)
    
    def calculate_collection_limit(self, current_volume: float,
                                 resource_type: ResourceType,
                                 available_amount: float) -> float:
        """Calculate how much can be collected given current state"""
        
        available_capacity = self.calculate_available_capacity(current_volume, resource_type)
        
        # Limit collection to available capacity
        collection_limit = min(available_amount, available_capacity)
        
        return max(0.0, collection_limit)
    
    def calculate_flight_efficiency(self, current_volume: float) -> float:
        """Calculate flight efficiency based on crop load"""
        
        load_factor = current_volume / self.max_crop_capacity
        
        if load_factor <= self.optimal_load_factor:
            return 1.0  # No penalty
        else:
            # Efficiency decreases with overload
            overload_factor = (load_factor - self.optimal_load_factor) / (1.0 - self.optimal_load_factor)
            efficiency = 1.0 - (overload_factor * 0.5)  # Up to 50% penalty
            return max(0.3, efficiency)  # Minimum 30% efficiency
    
    def calculate_energy_cost_multiplier(self, current_volume: float) -> float:
        """Calculate energy cost multiplier based on crop load"""
        
        load_factor = current_volume / self.max_crop_capacity
        
        if load_factor <= self.flight_efficiency_threshold:
            return 1.0  # No additional cost
        else:
            # Energy cost increases with heavy load
            excess_load = load_factor - self.flight_efficiency_threshold
            multiplier = 1.0 + (excess_load * (self.energy_cost_multiplier - 1.0))
            return min(self.energy_cost_multiplier, multiplier)


class ResourceDepletionModel(BaseModel):
    """Model for resource depletion and regeneration"""
    
    model_config = {"validate_assignment": True}
    
    # Depletion parameters
    base_depletion_rate: float = Field(default=0.1, ge=0.0, le=1.0, description="Base depletion rate per collection")
    visitor_impact_factor: float = Field(default=0.05, ge=0.0, description="Additional depletion per visitor")
    
    # Regeneration parameters
    base_regeneration_rate: float = Field(default=0.02, ge=0.0, description="Base regeneration rate per time step")
    environmental_factor: float = Field(default=1.0, ge=0.0, description="Environmental regeneration factor")
    
    # Resource-specific parameters
    nectar_regeneration_time: float = Field(default=30.0, ge=0.0, description="Nectar regeneration time (minutes)")
    pollen_regeneration_time: float = Field(default=240.0, ge=0.0, description="Pollen regeneration time (minutes)")
    
    def calculate_depletion_amount(self, resource_type: ResourceType,
                                 current_amount: float,
                                 collection_amount: float,
                                 visitor_count: int) -> float:
        """Calculate resource depletion from collection"""
        
        # Base depletion from collection
        depletion = collection_amount
        
        # Additional depletion from visitor pressure
        visitor_depletion = current_amount * self.visitor_impact_factor * visitor_count
        
        # Resource-specific depletion
        if resource_type == ResourceType.NECTAR:
            # Nectar depletes faster
            depletion *= 1.2
        elif resource_type == ResourceType.POLLEN:
            # Pollen depletes slower but more permanent
            depletion *= 0.8
        
        total_depletion = depletion + visitor_depletion
        
        return min(current_amount, total_depletion)
    
    def calculate_regeneration_amount(self, resource_type: ResourceType,
                                    current_amount: float,
                                    max_amount: float,
                                    time_since_depletion: float,
                                    environmental_conditions: Dict[str, float]) -> float:
        """Calculate resource regeneration over time"""
        
        if current_amount >= max_amount:
            return 0.0  # Already at maximum
        
        # Get resource-specific regeneration time
        if resource_type == ResourceType.NECTAR:
            regen_time = self.nectar_regeneration_time
        elif resource_type == ResourceType.POLLEN:
            regen_time = self.pollen_regeneration_time
        else:
            regen_time = 60.0  # Default 1 hour
        
        # Calculate regeneration rate
        regeneration_rate = self.base_regeneration_rate
        
        # Environmental factors
        temperature = environmental_conditions.get('temperature', 20.0)
        humidity = environmental_conditions.get('humidity', 50.0)
        
        # Temperature effect (optimal around 20-25°C)
        if 20.0 <= temperature <= 25.0:
            temp_factor = 1.0
        else:
            temp_factor = max(0.3, 1.0 - abs(temperature - 22.5) / 10.0)
        
        # Humidity effect (optimal around 60-70%)
        if 60.0 <= humidity <= 70.0:
            humidity_factor = 1.0
        else:
            humidity_factor = max(0.5, 1.0 - abs(humidity - 65.0) / 20.0)
        
        environmental_multiplier = (temp_factor + humidity_factor) / 2.0
        
        # Time-based regeneration
        time_factor = min(1.0, time_since_depletion / regen_time)
        
        # Calculate regeneration amount
        regeneration_amount = (max_amount - current_amount) * regeneration_rate * time_factor * environmental_multiplier
        
        return regeneration_amount


class DetailedResourceCollector(BaseModel):
    """Comprehensive resource collection system matching NetLogo complexity"""
    
    model_config = {"validate_assignment": True}
    
    # Component systems
    extraction_model: ResourceExtractionModel = Field(default_factory=ResourceExtractionModel)
    crop_manager: CropVolumeManager = Field(default_factory=CropVolumeManager)
    depletion_model: ResourceDepletionModel = Field(default_factory=ResourceDepletionModel)
    
    # Collection parameters
    max_collection_attempts: int = Field(default=10, ge=1, description="Maximum collection attempts per patch")
    success_threshold: float = Field(default=0.3, ge=0.0, le=1.0, description="Minimum efficiency for success")
    
    # Performance tracking
    collection_history: List[Dict[str, Any]] = Field(default_factory=list, description="Collection history")
    
    def attempt_resource_collection(self, bee_physiology: BeePhysiology,
                                  flower: FlowerCharacteristics,
                                  resource_type: ResourceType,
                                  experience_level: float,
                                  environmental_conditions: Dict[str, float]) -> Tuple[CollectionResult, float, Dict[str, Any]]:
        """Attempt to collect resources from a flower"""
        
        # Step 1: Check proboscis-corolla compatibility
        compatibility = self.extraction_model.calculate_proboscis_compatibility(
            bee_physiology.proboscis_length,
            flower.corolla_depth,
            flower.corolla_diameter
        )
        
        if compatibility < self.extraction_model.compatibility_threshold:
            return CollectionResult.INCOMPATIBLE, 0.0, {"compatibility": compatibility}
        
        # Step 2: Calculate extraction efficiency
        efficiency = self.extraction_model.calculate_extraction_efficiency(
            bee_physiology, flower, compatibility, experience_level
        )
        
        if efficiency < self.success_threshold:
            return CollectionResult.FAILURE, 0.0, {"compatibility": compatibility, "efficiency": efficiency}
        
        # Step 3: Determine available resource amount
        if resource_type == ResourceType.NECTAR:
            available_amount = flower.nectar_volume * flower.sugar_concentration
        elif resource_type == ResourceType.POLLEN:
            available_amount = flower.pollen_availability
        else:
            available_amount = 0.0
        
        if available_amount <= 0:
            return CollectionResult.DEPLETED, 0.0, {"compatibility": compatibility, "efficiency": efficiency}
        
        # Step 4: Calculate collection capacity limits
        if resource_type == ResourceType.NECTAR:
            collection_limit = self.crop_manager.calculate_collection_limit(
                bee_physiology.current_crop_volume, resource_type, available_amount
            )
        elif resource_type == ResourceType.POLLEN:
            collection_limit = min(available_amount, 
                                 bee_physiology.pollen_basket_capacity - bee_physiology.current_pollen_load)
        else:
            collection_limit = available_amount
        
        if collection_limit <= 0:
            return CollectionResult.CAPACITY_FULL, 0.0, {"compatibility": compatibility, "efficiency": efficiency}
        
        # Step 5: Calculate actual collection amount
        collection_amount = collection_limit * efficiency
        
        # Step 6: Calculate handling time
        handling_time = self.extraction_model.calculate_handling_time(
            flower, compatibility, efficiency
        )
        
        # Step 7: Apply environmental effects
        weather_effect = self._calculate_weather_effect(environmental_conditions)
        collection_amount *= weather_effect
        
        # Step 8: Determine collection result
        if collection_amount >= collection_limit * 0.8:
            result = CollectionResult.SUCCESS
        elif collection_amount >= collection_limit * 0.3:
            result = CollectionResult.PARTIAL_SUCCESS
        else:
            result = CollectionResult.FAILURE
        
        # Step 9: Record collection attempt
        collection_data = {
            "compatibility": compatibility,
            "efficiency": efficiency,
            "collection_amount": collection_amount,
            "handling_time": handling_time,
            "weather_effect": weather_effect,
            "available_amount": available_amount,
            "collection_limit": collection_limit
        }
        
        self.collection_history.append(collection_data)
        
        return result, collection_amount, collection_data
    
    def _calculate_weather_effect(self, environmental_conditions: Dict[str, float]) -> float:
        """Calculate weather effect on collection efficiency"""
        
        weather_effect = 1.0
        
        # Temperature effect
        temperature = environmental_conditions.get('temperature', 20.0)
        if temperature < 10.0:
            weather_effect *= 0.3  # Very cold reduces efficiency
        elif temperature < 15.0:
            weather_effect *= 0.7  # Cold reduces efficiency
        elif temperature > 35.0:
            weather_effect *= 0.5  # Very hot reduces efficiency
        
        # Wind effect
        wind_speed = environmental_conditions.get('wind_speed', 0.0)
        if wind_speed > 5.0:
            weather_effect *= max(0.2, 1.0 - (wind_speed - 5.0) / 10.0)
        
        # Precipitation effect
        precipitation = environmental_conditions.get('precipitation', 0.0)
        if precipitation > 0:
            weather_effect *= max(0.1, 1.0 - precipitation / 10.0)
        
        return weather_effect
    
    def update_bee_physiology(self, bee_physiology: BeePhysiology,
                            collection_result: CollectionResult,
                            collection_amount: float,
                            resource_type: ResourceType,
                            handling_time: float) -> None:
        """Update bee physiology after collection attempt"""
        
        if collection_result in [CollectionResult.SUCCESS, CollectionResult.PARTIAL_SUCCESS]:
            # Update resource loads
            if resource_type == ResourceType.NECTAR:
                bee_physiology.current_crop_volume += collection_amount
            elif resource_type == ResourceType.POLLEN:
                bee_physiology.current_pollen_load += collection_amount
            
            # Update energy based on handling time and crop load
            energy_cost = handling_time * 0.1  # Base energy cost
            
            # Additional cost for heavy loads
            if resource_type == ResourceType.NECTAR:
                energy_multiplier = self.crop_manager.calculate_energy_cost_multiplier(
                    bee_physiology.current_crop_volume
                )
                energy_cost *= energy_multiplier
            
            bee_physiology.energy_level = max(0, bee_physiology.energy_level - energy_cost)
        
        # Update collection efficiency based on success/failure
        if collection_result == CollectionResult.SUCCESS:
            bee_physiology.collection_efficiency = min(1.0, bee_physiology.collection_efficiency + 0.01)
        elif collection_result == CollectionResult.FAILURE:
            bee_physiology.collection_efficiency = max(0.1, bee_physiology.collection_efficiency - 0.005)
    
    def update_flower_resources(self, flower: FlowerCharacteristics,
                              collection_amount: float,
                              resource_type: ResourceType,
                              visitor_count: int,
                              time_since_last_visit: float,
                              environmental_conditions: Dict[str, float]) -> None:
        """Update flower resource levels after collection"""
        
        # Calculate depletion
        if resource_type == ResourceType.NECTAR:
            current_amount = flower.nectar_volume * flower.sugar_concentration
            depletion = self.depletion_model.calculate_depletion_amount(
                resource_type, current_amount, collection_amount, visitor_count
            )
            new_amount = max(0, current_amount - depletion)
            
            # Update nectar volume (assuming constant sugar concentration)
            if flower.sugar_concentration > 0:
                flower.nectar_volume = new_amount / flower.sugar_concentration
        
        elif resource_type == ResourceType.POLLEN:
            depletion = self.depletion_model.calculate_depletion_amount(
                resource_type, flower.pollen_availability, collection_amount, visitor_count
            )
            flower.pollen_availability = max(0, flower.pollen_availability - depletion)
        
        # Calculate regeneration if time has passed
        if time_since_last_visit > 0:
            self._regenerate_flower_resources(flower, time_since_last_visit, environmental_conditions)
    
    def _regenerate_flower_resources(self, flower: FlowerCharacteristics,
                                   time_passed: float,
                                   environmental_conditions: Dict[str, float]) -> None:
        """Regenerate flower resources over time"""
        
        # Nectar regeneration
        current_nectar = flower.nectar_volume * flower.sugar_concentration
        max_nectar = 10.0  # Assume maximum nectar amount
        
        nectar_regeneration = self.depletion_model.calculate_regeneration_amount(
            ResourceType.NECTAR, current_nectar, max_nectar, time_passed, environmental_conditions
        )
        
        if nectar_regeneration > 0 and flower.sugar_concentration > 0:
            new_nectar = current_nectar + nectar_regeneration
            flower.nectar_volume = new_nectar / flower.sugar_concentration
        
        # Pollen regeneration
        max_pollen = 5.0  # Assume maximum pollen amount
        
        pollen_regeneration = self.depletion_model.calculate_regeneration_amount(
            ResourceType.POLLEN, flower.pollen_availability, max_pollen, time_passed, environmental_conditions
        )
        
        flower.pollen_availability = min(max_pollen, flower.pollen_availability + pollen_regeneration)
    
    def get_collection_efficiency_stats(self) -> Dict[str, float]:
        """Get collection efficiency statistics"""
        
        if not self.collection_history:
            return {}
        
        efficiencies = [record["efficiency"] for record in self.collection_history]
        compatibilities = [record["compatibility"] for record in self.collection_history]
        collection_amounts = [record["collection_amount"] for record in self.collection_history]
        handling_times = [record["handling_time"] for record in self.collection_history]
        
        return {
            "average_efficiency": sum(efficiencies) / len(efficiencies),
            "average_compatibility": sum(compatibilities) / len(compatibilities),
            "total_collected": sum(collection_amounts),
            "average_handling_time": sum(handling_times) / len(handling_times),
            "collection_attempts": len(self.collection_history)
        }
    
    def calculate_optimal_proboscis_length(self, flower_depths: List[float]) -> float:
        """Calculate optimal proboscis length for given flower depths"""
        
        if not flower_depths:
            return 6.0  # Default length
        
        # Calculate weighted average depth
        avg_depth = sum(flower_depths) / len(flower_depths)
        
        # Optimal proboscis length is slightly longer than average depth
        optimal_length = avg_depth * 1.2
        
        return optimal_length
    
    def simulate_foraging_trip(self, bee_physiology: BeePhysiology,
                             flowers: List[FlowerCharacteristics],
                             environmental_conditions: Dict[str, float]) -> Dict[str, Any]:
        """Simulate a complete foraging trip"""
        
        trip_results = {
            "flowers_visited": 0,
            "successful_collections": 0,
            "total_nectar_collected": 0.0,
            "total_pollen_collected": 0.0,
            "total_handling_time": 0.0,
            "trip_efficiency": 0.0,
            "collection_details": []
        }
        
        for flower in flowers:
            # Check if bee can carry more
            if (bee_physiology.current_crop_volume >= self.crop_manager.max_crop_capacity * 0.9 and
                bee_physiology.current_pollen_load >= bee_physiology.pollen_basket_capacity * 0.9):
                break  # Trip complete
            
            # Attempt nectar collection
            if bee_physiology.current_crop_volume < self.crop_manager.max_crop_capacity * 0.9:
                result, amount, details = self.attempt_resource_collection(
                    bee_physiology, flower, ResourceType.NECTAR, 0.5, environmental_conditions
                )
                
                if result in [CollectionResult.SUCCESS, CollectionResult.PARTIAL_SUCCESS]:
                    trip_results["successful_collections"] += 1
                    trip_results["total_nectar_collected"] += amount
                    trip_results["total_handling_time"] += details["handling_time"]
                    
                    self.update_bee_physiology(bee_physiology, result, amount, ResourceType.NECTAR, details["handling_time"])
                
                trip_results["collection_details"].append({
                    "resource_type": "nectar",
                    "result": result.value,
                    "amount": amount,
                    "details": details
                })
            
            # Attempt pollen collection
            if bee_physiology.current_pollen_load < bee_physiology.pollen_basket_capacity * 0.9:
                result, amount, details = self.attempt_resource_collection(
                    bee_physiology, flower, ResourceType.POLLEN, 0.5, environmental_conditions
                )
                
                if result in [CollectionResult.SUCCESS, CollectionResult.PARTIAL_SUCCESS]:
                    trip_results["successful_collections"] += 1
                    trip_results["total_pollen_collected"] += amount
                    trip_results["total_handling_time"] += details["handling_time"]
                    
                    self.update_bee_physiology(bee_physiology, result, amount, ResourceType.POLLEN, details["handling_time"])
                
                trip_results["collection_details"].append({
                    "resource_type": "pollen",
                    "result": result.value,
                    "amount": amount,
                    "details": details
                })
            
            trip_results["flowers_visited"] += 1
        
        # Calculate trip efficiency
        if trip_results["flowers_visited"] > 0:
            trip_results["trip_efficiency"] = trip_results["successful_collections"] / trip_results["flowers_visited"]
        
        return trip_results