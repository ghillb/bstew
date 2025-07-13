"""
Genetic system implementation for BSTEW
=======================================

Implements Complementary Sex Determination (CSD), diploid male detection,
inbreeding effects, and genetic inheritance for realistic bee population genetics.
"""

import numpy as np
import random
from typing import Dict, List, Optional, Any, Set
from pydantic import BaseModel, Field, field_validator
from enum import Enum
import logging
from collections import defaultdict


class Sex(Enum):
    """Bee sex types"""

    MALE = "male"
    FEMALE = "female"
    DIPLOID_MALE = "diploid_male"  # Lethal condition


class Ploidy(Enum):
    """Chromosomal ploidy levels"""

    HAPLOID = 1  # Males (normal)
    DIPLOID = 2  # Females and diploid males


class Allele(BaseModel):
    """Individual allele for sex determination"""

    model_config = {"validate_assignment": True}

    allele_id: int = Field(ge=0, description="Unique allele identifier")
    origin: str = Field(description="Allele origin: 'maternal' or 'paternal'")

    @field_validator("origin")
    @classmethod
    def validate_origin(cls, v: str) -> str:
        if v not in ["maternal", "paternal"]:
            raise ValueError("Origin must be 'maternal' or 'paternal'")
        return v

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Allele) and self.allele_id == other.allele_id

    def __hash__(self) -> int:
        return hash(self.allele_id)

    def __repr__(self) -> str:
        return f"Allele({self.allele_id})"


class Genotype(BaseModel):
    """Individual bee genotype"""

    model_config = {"validate_assignment": True}

    alleles: List[Allele] = Field(description="List of alleles for sex determination")
    ploidy: Ploidy = Field(description="Chromosomal ploidy level")
    sex: Sex = Field(description="Biological sex")
    mtDNA: int = Field(ge=0, description="Mitochondrial DNA identifier")

    @field_validator("alleles")
    @classmethod
    def validate_alleles(cls, v: List[Allele], info: Any) -> List[Allele]:
        if not v:
            raise ValueError("Genotype must have at least one allele")
        if "ploidy" in info.data:
            expected_count = info.data["ploidy"].value
            if len(v) != expected_count:
                raise ValueError(
                    f"Expected {expected_count} alleles for {info.data['ploidy'].name} genotype, got {len(v)}"
                )
        return v

    def __post_init__(self) -> None:
        if self.ploidy == Ploidy.HAPLOID:
            assert len(self.alleles) == 1
            self.sex = Sex.MALE
        else:  # DIPLOID
            assert len(self.alleles) == 2
            if self.alleles[0] == self.alleles[1]:
                self.sex = Sex.DIPLOID_MALE  # Lethal condition
            else:
                self.sex = Sex.FEMALE

    def is_diploid_male(self) -> bool:
        """Check if this genotype represents a diploid male"""
        return self.sex == Sex.DIPLOID_MALE

    def get_allele_ids(self) -> List[int]:
        """Get list of allele IDs"""
        return [allele.allele_id for allele in self.alleles]


class SpermCell(BaseModel):
    """Individual sperm cell in spermatheca"""

    model_config = {"validate_assignment": True}

    allele: Allele = Field(description="Sex determination allele")
    mtDNA: int = Field(ge=0, description="Mitochondrial DNA identifier")
    viability: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Sperm viability (0-1)"
    )
    age: int = Field(default=0, ge=0, description="Days since storage")

    def model_post_init(self, __context: Any) -> None:
        """Update viability based on age after model initialization"""
        # Sperm viability decreases with age
        if self.age > 0:
            self.viability = max(0.0, 1.0 - (self.age * 0.001))  # 0.1% daily decline


class SpermathecaManager:
    """
    Manages sperm storage and utilization for queens.

    Implements realistic sperm storage, aging, and utilization patterns.
    """

    def __init__(self, capacity: int = 6000000):  # ~6 million sperm typical
        self.capacity = capacity
        self.sperm_cells: List[SpermCell] = []
        self.utilization_rate = 0.001  # Fraction used per egg
        self.mixing_efficiency = 0.95  # How well sperm mix

    def store_sperm(self, sperm_cells: List[SpermCell]) -> int:
        """Store sperm from mating, return number successfully stored"""

        available_space = self.capacity - len(self.sperm_cells)
        storable_count = min(len(sperm_cells), available_space)

        # Add sperm to storage
        self.sperm_cells.extend(sperm_cells[:storable_count])

        # Shuffle to simulate mixing
        if self.mixing_efficiency > random.random():
            random.shuffle(self.sperm_cells)

        return storable_count

    def get_sperm_for_fertilization(self) -> Optional[SpermCell]:
        """Get sperm cell for egg fertilization"""

        if not self.sperm_cells:
            return None

        # Select sperm cell (first available with mixing)
        viable_sperm = [sperm for sperm in self.sperm_cells if sperm.viability > 0.5]

        if not viable_sperm:
            return None

        # Remove and return sperm cell
        selected_sperm = viable_sperm[0]
        self.sperm_cells.remove(selected_sperm)

        return selected_sperm

    def age_sperm(self, days: int = 1) -> None:
        """Age all sperm by specified days"""
        for sperm in self.sperm_cells:
            sperm.age += days
            sperm.viability = max(0.0, 1.0 - (sperm.age * 0.001))

        # Remove non-viable sperm
        self.sperm_cells = [
            sperm for sperm in self.sperm_cells if sperm.viability > 0.1
        ]

    def get_sperm_count(self) -> int:
        """Get total viable sperm count"""
        return len([sperm for sperm in self.sperm_cells if sperm.viability > 0.5])

    def get_genetic_diversity(self) -> float:
        """Calculate genetic diversity of stored sperm"""
        if not self.sperm_cells:
            return 0.0

        allele_counts: Dict[int, int] = defaultdict(int)
        for sperm in self.sperm_cells:
            allele_counts[sperm.allele.allele_id] += 1

        # Calculate Shannon diversity
        total = len(self.sperm_cells)
        diversity = 0.0
        for count in allele_counts.values():
            proportion = count / total
            if proportion > 0:
                diversity -= proportion * np.log(proportion)

        return diversity


class GeneticSystem:
    """
    Comprehensive genetic system for bee colonies.

    Implements:
    - Complementary Sex Determination (CSD)
    - Diploid male detection and mortality
    - Inbreeding coefficients
    - Genetic diversity tracking
    - Realistic mating and inheritance
    """

    def __init__(self, initial_allele_count: int = 100):
        self.allele_pool_size = initial_allele_count
        self.next_allele_id = 1
        self.next_mtDNA_id = 1
        self.population_alleles: Set[int] = set()
        self.diploid_male_mortality = 1.0  # 100% mortality for diploid males
        self.inbreeding_effects = True

        self.logger = logging.getLogger(__name__)

        # Initialize allele pool
        self._initialize_allele_pool()

    def _initialize_allele_pool(self) -> None:
        """Initialize population allele pool"""
        for _ in range(self.allele_pool_size):
            self.population_alleles.add(self.next_allele_id)
            self.next_allele_id += 1

    def create_founder_genotype(self, sex: Optional[Sex] = None) -> Genotype:
        """Create genotype for founder individuals"""

        if sex == Sex.MALE or (sex is None and random.random() < 0.5):
            # Create haploid male
            allele_id = random.choice(list(self.population_alleles))
            allele = Allele(allele_id=allele_id, origin="maternal")
            return Genotype(
                alleles=[allele],
                ploidy=Ploidy.HAPLOID,
                sex=Sex.MALE,
                mtDNA=self._get_next_mtDNA_id(),
            )
        else:
            # Create diploid female
            allele_ids = random.sample(list(self.population_alleles), 2)
            alleles = [
                Allele(allele_id=allele_ids[0], origin="maternal"),
                Allele(allele_id=allele_ids[1], origin="paternal"),
            ]
            return Genotype(
                alleles=alleles,
                ploidy=Ploidy.DIPLOID,
                sex=Sex.FEMALE,
                mtDNA=self._get_next_mtDNA_id(),
            )

    def mate_individuals(
        self, female_genotype: Genotype, male_genotypes: List[Genotype]
    ) -> List[SpermCell]:
        """Simulate mating between queen and multiple males"""

        if female_genotype.sex != Sex.FEMALE:
            raise ValueError("Female must have female genotype")

        sperm_cells = []

        for male_genotype in male_genotypes:
            if male_genotype.sex != Sex.MALE:
                continue

            # Each male contributes sperm
            sperm_count = random.randint(100000, 500000)  # Variable sperm contribution

            for _ in range(sperm_count):
                sperm_cell = SpermCell(
                    allele=male_genotype.alleles[0],  # Males are haploid
                    mtDNA=male_genotype.mtDNA,
                )
                sperm_cells.append(sperm_cell)

        return sperm_cells

    def fertilize_egg(
        self, female_genotype: Genotype, sperm_cell: SpermCell
    ) -> Genotype:
        """Fertilize egg to create offspring genotype"""

        # Female contributes one allele randomly
        maternal_allele = random.choice(female_genotype.alleles)
        maternal_allele.origin = "maternal"

        # Sperm contributes paternal allele
        paternal_allele = sperm_cell.allele
        paternal_allele.origin = "paternal"

        # Create offspring genotype
        offspring = Genotype(
            alleles=[maternal_allele, paternal_allele],
            ploidy=Ploidy.DIPLOID,
            sex=Sex.FEMALE,  # Will be corrected in __post_init__
            mtDNA=female_genotype.mtDNA,  # mtDNA inherited maternally
        )

        return offspring

    def create_unfertilized_egg(self, female_genotype: Genotype) -> Genotype:
        """Create unfertilized egg (haploid male)"""

        # Female contributes one allele randomly
        maternal_allele = random.choice(female_genotype.alleles)
        maternal_allele.origin = "maternal"

        return Genotype(
            alleles=[maternal_allele],
            ploidy=Ploidy.HAPLOID,
            sex=Sex.MALE,
            mtDNA=female_genotype.mtDNA,
        )

    def calculate_inbreeding_coefficient(
        self, genotype: Genotype, colony_alleles: Set[int]
    ) -> float:
        """Calculate inbreeding coefficient for individual"""

        if genotype.ploidy == Ploidy.HAPLOID:
            return 0.0  # Males cannot be inbred

        # For diploid individuals, calculate based on allele frequency
        allele_ids = genotype.get_allele_ids()

        # Inbreeding coefficient based on allele rarity
        inbreeding_coeff = 0.0
        for allele_id in allele_ids:
            if allele_id in colony_alleles:
                # More common alleles indicate higher inbreeding
                frequency = list(colony_alleles).count(allele_id) / len(colony_alleles)
                inbreeding_coeff += frequency

        return inbreeding_coeff / len(allele_ids)

    def assess_diploid_male_impact(self, genotype: Genotype) -> Dict[str, Any]:
        """Assess impact of diploid male on colony"""

        if not genotype.is_diploid_male():
            return {"is_diploid_male": False, "mortality_risk": 0.0}

        # Diploid males are lethal to colonies
        impact = {
            "is_diploid_male": True,
            "mortality_risk": self.diploid_male_mortality,
            "energy_cost": 1.5,  # Higher energy cost to produce
            "development_failure": True,
            "colony_stress": 0.8,  # High stress on colony
        }

        self.logger.warning(f"Diploid male detected: {genotype}")
        return impact

    def calculate_genetic_diversity(
        self, genotypes: List[Genotype]
    ) -> Dict[str, float]:
        """Calculate genetic diversity metrics for population"""

        if not genotypes:
            return {"allelic_diversity": 0.0, "heterozygosity": 0.0}

        # Collect all alleles
        all_alleles = []
        diploid_individuals = []

        for genotype in genotypes:
            all_alleles.extend(genotype.get_allele_ids())
            if genotype.ploidy == Ploidy.DIPLOID:
                diploid_individuals.append(genotype)

        # Allelic diversity (number of unique alleles)
        unique_alleles = set(all_alleles)
        allelic_diversity = (
            len(unique_alleles) / len(all_alleles) if all_alleles else 0.0
        )

        # Heterozygosity (proportion of diploid individuals that are heterozygous)
        heterozygous_count = 0
        for genotype in diploid_individuals:
            if len(set(genotype.get_allele_ids())) > 1:
                heterozygous_count += 1

        heterozygosity = (
            heterozygous_count / len(diploid_individuals)
            if diploid_individuals
            else 0.0
        )

        return {
            "allelic_diversity": allelic_diversity,
            "heterozygosity": heterozygosity,
            "unique_alleles": len(unique_alleles),
            "total_alleles": len(all_alleles),
        }

    def simulate_population_genetics(
        self, colony_genotypes: List[Genotype], generations: int = 10
    ) -> Dict[str, Any]:
        """Simulate population genetics over multiple generations"""

        results: Dict[str, Any] = {
            "generation_data": [],
            "diploid_male_frequencies": [],
            "inbreeding_coefficients": [],
            "genetic_diversity": [],
        }

        current_population = colony_genotypes.copy()

        for generation in range(generations):
            # Calculate metrics for current generation
            diversity = self.calculate_genetic_diversity(current_population)
            diploid_male_count = sum(
                1 for g in current_population if g.is_diploid_male()
            )
            diploid_male_frequency = (
                diploid_male_count / len(current_population)
                if current_population
                else 0.0
            )

            # Calculate average inbreeding coefficient
            colony_alleles = set()
            for genotype in current_population:
                colony_alleles.update(genotype.get_allele_ids())

            avg_inbreeding = (
                np.mean(
                    [
                        self.calculate_inbreeding_coefficient(g, colony_alleles)
                        for g in current_population
                        if g.ploidy == Ploidy.DIPLOID
                    ]
                )
                if current_population
                else 0.0
            )

            # Store results
            results["generation_data"].append(
                {
                    "generation": generation,
                    "population_size": len(current_population),
                    "diploid_male_count": diploid_male_count,
                }
            )
            results["diploid_male_frequencies"].append(diploid_male_frequency)
            results["inbreeding_coefficients"].append(avg_inbreeding)
            results["genetic_diversity"].append(diversity)

            # Simulate next generation (simplified)
            if generation < generations - 1:
                current_population = self._simulate_next_generation(current_population)

        return results

    def _simulate_next_generation(
        self, parent_generation: List[Genotype]
    ) -> List[Genotype]:
        """Simulate production of next generation"""

        # Filter out diploid males (they die)
        surviving_parents = [g for g in parent_generation if not g.is_diploid_male()]

        # Simple reproduction simulation
        next_generation = []

        females = [g for g in surviving_parents if g.sex == Sex.FEMALE]
        males = [g for g in surviving_parents if g.sex == Sex.MALE]

        if not females or not males:
            return []  # Population crash

        # Each female produces offspring
        for female in females:
            male_partners = random.sample(males, min(3, len(males)))  # Polyandry
            sperm_cells = self.mate_individuals(female, male_partners)

            # Produce offspring
            for _ in range(random.randint(50, 200)):  # Variable offspring count
                if sperm_cells and random.random() < 0.7:  # 70% fertilized
                    sperm = random.choice(sperm_cells)
                    offspring = self.fertilize_egg(female, sperm)
                else:  # Unfertilized (male)
                    offspring = self.create_unfertilized_egg(female)

                next_generation.append(offspring)

        return next_generation

    def _get_next_mtDNA_id(self) -> int:
        """Get next mitochondrial DNA identifier"""
        mtDNA_id = self.next_mtDNA_id
        self.next_mtDNA_id += 1
        return mtDNA_id

    def get_genetic_summary(self, genotypes: List[Genotype]) -> Dict[str, Any]:
        """Get comprehensive genetic summary for colony"""

        summary = {
            "total_individuals": len(genotypes),
            "males": len([g for g in genotypes if g.sex == Sex.MALE]),
            "females": len([g for g in genotypes if g.sex == Sex.FEMALE]),
            "diploid_males": len([g for g in genotypes if g.is_diploid_male()]),
            "genetic_diversity": self.calculate_genetic_diversity(genotypes),
            "diploid_male_frequency": (
                len([g for g in genotypes if g.is_diploid_male()]) / len(genotypes)
                if genotypes
                else 0.0
            ),
        }

        # Add inbreeding statistics
        colony_alleles = set()
        for genotype in genotypes:
            colony_alleles.update(genotype.get_allele_ids())

        inbreeding_coeffs = [
            self.calculate_inbreeding_coefficient(g, colony_alleles)
            for g in genotypes
            if g.ploidy == Ploidy.DIPLOID
        ]

        summary["inbreeding_statistics"] = {
            "mean_inbreeding_coefficient": (
                np.mean(inbreeding_coeffs) if inbreeding_coeffs else 0.0
            ),
            "max_inbreeding_coefficient": (
                np.max(inbreeding_coeffs) if inbreeding_coeffs else 0.0
            ),
            "inbred_individuals": len(
                [coeff for coeff in inbreeding_coeffs if coeff > 0.1]
            ),
        }

        return summary
