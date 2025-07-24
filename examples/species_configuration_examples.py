"""
BSTEW Species Configuration Examples
===================================

This file demonstrates how to configure and use the multi-species
bee system in BSTEW, supporting both honey bees and bumblebees.
"""

from bstew.core.bee_species_config import (
    BeeSpeciesType, BeeSpeciesConfig, BeeSpeciesManager,
    CommunicationType, create_multi_species_simulation
)
from bstew.core.bee_communication import create_communication_system, create_multi_species_communication


# =============================================================================
# 1. BASIC SPECIES SELECTION
# =============================================================================

def basic_species_selection():
    """Example 1: Basic species selection and system creation"""
    
    print("=== Basic Species Selection ===")
    
    # Available species
    available_species = [
        BeeSpeciesType.APIS_MELLIFERA,      # Honey bee
        BeeSpeciesType.BOMBUS_TERRESTRIS,   # Large earth bumblebee
        BeeSpeciesType.BOMBUS_LAPIDARIUS,   # Red-tailed bumblebee
        BeeSpeciesType.BOMBUS_PASCUORUM,    # Common carder bee
    ]
    
    # Create communication systems for each species
    for species in available_species:
        comm_system = create_communication_system(species)
        info = comm_system.get_species_info()
        
        print(f"\n{info['common_name']} ({info['scientific_name']}):")
        print(f"  Communication: {info['communication_type']}")
        print(f"  Colony size: {info['colony_size_range']}")
        print(f"  Max foraging: {info['max_foraging_distance']}m")
        print(f"  Social sharing: {info['social_information_sharing']:.1f}")
        print(f"  Individual weight: {info['individual_decision_weight']:.1f}")


# =============================================================================
# 2. SIMULATION CONFIGURATION
# =============================================================================

def create_single_species_simulation():
    """Example 2: Configure single-species simulation"""
    
    print("\n=== Single Species Simulation ===")
    
    # Choose species for simulation
    species_type = BeeSpeciesType.BOMBUS_TERRESTRIS
    
    # Create communication system
    comm_system = create_communication_system(species_type)
    
    # Example foraging integration
    foraging_result = {
        "patch_quality": 0.8,
        "energy_gained": 45.0,
        "distance": 400.0,
        "patch_id": 123,
        "location": (100.0, 200.0)
    }
    
    # Process communication
    result = comm_system.integrate_foraging_success_with_communication(
        bee_id=456,
        foraging_result=foraging_result,
        colony_state={"energy_level": 0.6},
        environmental_context={"temperature": 18.0}
    )
    
    print(f"Species: {comm_system.get_species_info()['common_name']}")
    print(f"Communication type: {result['communication_type']}")
    print(f"Should communicate: {result['should_communicate']}")
    print(f"Communication vigor: {result['communication_vigor']:.2f}")
    
    return comm_system


def create_multi_species_simulation_example():
    """Example 3: Configure multi-species simulation"""
    
    print("\n=== Multi-Species Simulation ===")
    
    # Define species for the simulation
    species_list = [
        BeeSpeciesType.APIS_MELLIFERA,     # 1 honey bee colony
        BeeSpeciesType.BOMBUS_TERRESTRIS,  # 3 bumblebee colonies
        BeeSpeciesType.BOMBUS_LAPIDARIUS,  # 2 bumblebee colonies
    ]
    
    # Create simulation configuration
    simulation_config = create_multi_species_simulation(species_list)
    
    print("Simulation includes:")
    for species_type, config in simulation_config.items():
        species_config = config["config"]
        print(f"\n{species_config.common_name}:")
        print(f"  Max colony size: {species_config.max_colony_size:,}")
        print(f"  Typical foragers: {species_config.typical_forager_count:,}")
        print(f"  Communication: {species_config.communication_type.value}")
        print(f"  Social vs Individual: {species_config.social_information_sharing:.1f} vs {species_config.individual_decision_weight:.1f}")
    
    return simulation_config


# =============================================================================
# 3. CUSTOM SPECIES CONFIGURATION
# =============================================================================

def create_custom_species():
    """Example 4: Create custom species configuration"""
    
    print("\n=== Custom Species Configuration ===")
    
    # Create a custom species configuration (modify existing species)
    # We'll modify an existing species rather than create a completely new one
    from bstew.core.bee_species_config import species_manager
    
    # Get existing Bombus pascuorum config and modify it
    base_config = species_manager.get_species_config(BeeSpeciesType.BOMBUS_PASCUORUM)
    
    # Create modified version with different parameters
    custom_species = BeeSpeciesConfig(
        species_type=BeeSpeciesType.BOMBUS_PASCUORUM,  # Reuse existing enum
        common_name="Modified Carder Bee (Custom)",
        scientific_name="Bombus pascuorum (modified)",
        communication_type=CommunicationType.SCENT_COMMUNICATION,
        
        # Smaller colonies
        max_colony_size=150,
        min_colony_size=30,
        typical_forager_count=20,
        
        # Specialized for deep flowers - shorter foraging distance
        max_foraging_distance_m=800,
        typical_foraging_distance_m=250,
        min_temperature_c=9.0,
        
        # Communication parameters
        uses_dance_communication=False,
        uses_scent_communication=True,
        social_information_sharing=0.1,  # Very low social sharing
        individual_decision_weight=0.95,  # Very high individual decisions
        
        # Memory and learning
        patch_memory_capacity=6,  # Small memory
        learning_rate=0.05,       # Slow learning
        
        # Physical characteristics - long tongue specialist
        proboscis_length_mm=16.9,  # Very long tongue
        body_size_mm=15.0,
        flight_speed_ms=3.8
    )
    
    # Add to species manager
    species_manager = BeeSpeciesManager()
    species_manager.add_custom_species(custom_species)
    
    # Test the custom species (now that it's added)
    comm_system = create_communication_system(BeeSpeciesType.BOMBUS_PASCUORUM)
    info = comm_system.get_species_info()
    
    print(f"Custom species: {info['common_name']}")
    print(f"Proboscis length: {custom_species.proboscis_length_mm}mm (specialist)")
    print(f"Colony size: {custom_species.min_colony_size}-{custom_species.max_colony_size}")
    print(f"Individual decision weight: {custom_species.individual_decision_weight}")
    
    return custom_species


# =============================================================================
# 4. CONFIGURATION VIA YAML/JSON FILES
# =============================================================================

def load_simulation_from_config():
    """Example 5: Load simulation configuration from file"""
    
    print("\n=== Configuration File Example ===")
    
    # Example configuration structure (would be loaded from YAML/JSON)
    simulation_config = {
        "simulation_name": "Pollinator_Conservation_Study",
        "species": [
            {
                "type": "APIS_MELLIFERA",
                "colonies": 1,
                "initial_population": 45000,
                "location": [0.0, 0.0]
            },
            {
                "type": "BOMBUS_TERRESTRIS", 
                "colonies": 3,
                "initial_population": 250,
                "locations": [[100.0, 100.0], [200.0, 150.0], [150.0, 200.0]]
            },
            {
                "type": "BOMBUS_PASCUORUM",
                "colonies": 2, 
                "initial_population": 120,
                "locations": [[300.0, 100.0], [250.0, 250.0]]
            }
        ],
        "environment": {
            "landscape_size": [1000.0, 1000.0],
            "flower_patches": 50,
            "simulation_days": 180
        }
    }
    
    print("Simulation Configuration:")
    print(f"Name: {simulation_config['simulation_name']}")
    print(f"Species count: {len(simulation_config['species'])}")
    
    # Create communication systems for each species type
    species_systems = {}
    for species_config in simulation_config["species"]:
        species_type = BeeSpeciesType[species_config["type"]]
        if species_type not in species_systems:
            species_systems[species_type] = create_communication_system(species_type)
    
    print("\nCommunication systems created for:")
    for species_type, comm_system in species_systems.items():
        info = comm_system.get_species_info()
        print(f"  - {info['common_name']}: {info['communication_type']}")
    
    return simulation_config, species_systems


# =============================================================================
# 5. RUNTIME SPECIES SWITCHING
# =============================================================================

def demonstrate_runtime_switching():
    """Example 6: Switch between species during runtime"""
    
    print("\n=== Runtime Species Switching ===")
    
    # Create different communication systems
    systems = {
        "honey_bee": create_communication_system(BeeSpeciesType.APIS_MELLIFERA),
        "earth_bumblebee": create_communication_system(BeeSpeciesType.BOMBUS_TERRESTRIS),
        "carder_bee": create_communication_system(BeeSpeciesType.BOMBUS_PASCUORUM)
    }
    
    # Same foraging scenario, different species responses
    foraging_scenario = {
        "patch_quality": 0.7,
        "energy_gained": 35.0,
        "distance": 600.0,
        "patch_id": 789
    }
    
    print("Same foraging scenario, different species responses:")
    print(f"Patch quality: {foraging_scenario['patch_quality']}")
    print(f"Distance: {foraging_scenario['distance']}m")
    
    for name, system in systems.items():
        result = system.integrate_foraging_success_with_communication(
            bee_id=100, 
            foraging_result=foraging_scenario,
            colony_state={},
            environmental_context={}
        )
        
        info = system.get_species_info()
        print(f"\n{info['common_name']}:")
        print(f"  Communication: {result['communication_type']}")
        print(f"  Will communicate: {result['should_communicate']}")
        print(f"  Vigor/Strength: {result['communication_vigor']:.2f}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("BSTEW Species Configuration Examples")
    print("=" * 50)
    
    # Run all examples
    basic_species_selection()
    single_system = create_single_species_simulation()
    multi_config = create_multi_species_simulation_example()
    custom_species = create_custom_species()
    config, systems = load_simulation_from_config()
    demonstrate_runtime_switching()
    
    print("\n" + "=" * 50)
    print("Configuration examples completed!")
    print("\nKey takeaways:")
    print("1. Species selection determines communication behavior")
    print("2. Multi-species simulations are fully supported")
    print("3. Custom species can be easily configured")
    print("4. Runtime switching allows comparative studies")
    print("5. Configuration files enable reproducible research")