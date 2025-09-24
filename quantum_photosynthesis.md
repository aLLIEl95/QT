"""
Quantum Tunneling in Photosynthesis - Integration Module
Integrates quantum tunneling effects with photosynthesis simulation
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum

class TunnelingProcess(Enum):
    """Types of quantum tunneling processes in photosynthesis"""
    ELECTRON_TRANSPORT = "electron_transport"
    PROTON_TRANSFER = "proton_transfer"
    EXCITATION_TRANSFER = "excitation_transfer"
    CHARGE_SEPARATION = "charge_separation"

@dataclass
class QuantumTunnelingParameters:
    """Parameters for quantum tunneling calculations"""
    barrier_height: float  # eV
    barrier_width: float   # nm
    particle_mass: float   # electron masses
    temperature: float     # K
    reorganization_energy: float  # eV
    driving_force: float   # eV

@dataclass
class PhotosyntheticConditions:
    """Environmental and biological conditions affecting photosynthesis"""
    light_intensity: float     # μmol photons m⁻² s⁻¹
    wavelength: float         # nm
    temperature: float        # K
    co2_concentration: float  # ppm
    water_availability: float # relative units (0-1)
    ph: float                # pH
    chlorophyll_concentration: float  # mg/L

class QuantumPhotosynthesisIntegrator:
    """
    Integrates quantum tunneling effects with photosynthesis processes
    """
    
    def __init__(self):
        # Physical constants
        self.h_bar = 1.055e-34  # J⋅s
        self.k_b = 1.381e-23    # J/K
        self.e = 1.602e-19      # C
        self.m_e = 9.109e-31    # kg
        
        # Photosynthetic complexes and their tunneling parameters
        self.complexes = {
            'photosystem_ii': {
                'p680_pheophytin': QuantumTunnelingParameters(
                    barrier_height=0.8, barrier_width=1.2, particle_mass=1.0,
                    temperature=298, reorganization_energy=0.3, driving_force=-0.5
                ),
                'pheophytin_qa': QuantumTunnelingParameters(
                    barrier_height=0.6, barrier_width=1.5, particle_mass=1.0,
                    temperature=298, reorganization_energy=0.4, driving_force=-0.3
                )
            },
            'cytochrome_b6f': {
                'rieske_cytf': QuantumTunnelingParameters(
                    barrier_height=0.7, barrier_width=1.0, particle_mass=1.0,
                    temperature=298, reorganization_energy=0.2, driving_force=-0.2
                )
            },
            'photosystem_i': {
                'p700_a0': QuantumTunnelingParameters(
                    barrier_height=0.5, barrier_width=0.8, particle_mass=1.0,
                    temperature=298, reorganization_energy=0.25, driving_force=-0.4
                ),
                'a0_a1': QuantumTunnelingParameters(
                    barrier_height=0.4, barrier_width=1.3, particle_mass=1.0,
                    temperature=298, reorganization_energy=0.35, driving_force=-0.15
                )
            },
            'atp_synthase': {
                'proton_tunnel': QuantumTunnelingParameters(
                    barrier_height=0.9, barrier_width=0.5, particle_mass=1836,  # proton mass
                    temperature=298, reorganization_energy=0.1, driving_force=-0.3
                )
            }
        }
    
    def calculate_tunneling_probability(self, params: QuantumTunnelingParameters) -> float:
        """
        Calculate quantum tunneling probability using WKB approximation
        """
        # Convert units
        barrier_height_j = params.barrier_height * self.e
        barrier_width_m = params.barrier_width * 1e-9
        mass_kg = params.particle_mass * self.m_e
        
        # WKB transmission coefficient
        kappa = np.sqrt(2 * mass_kg * barrier_height_j) / self.h_bar
        transmission = np.exp(-2 * kappa * barrier_width_m)
        
        return transmission
    
    def marcus_tunneling_rate(self, params: QuantumTunnelingParameters) -> float:
        """
        Calculate electron transfer rate using Marcus theory with tunneling
        """
        # Temperature factor
        kt = self.k_b * params.temperature / self.e  # in eV
        
        # Marcus activation energy
        delta_g = params.driving_force
        lambda_reorg = params.reorganization_energy
        activation_energy = (lambda_reorg + delta_g)**2 / (4 * lambda_reorg)
        
        # Electronic coupling with tunneling
        tunneling_prob = self.calculate_tunneling_probability(params)
        electronic_coupling = 0.1 * np.sqrt(tunneling_prob)  # eV, typical value scaled by tunneling
        
        # Marcus rate equation
        prefactor = (2 * np.pi / self.h_bar) * electronic_coupling**2
        prefactor /= np.sqrt(4 * np.pi * lambda_reorg * kt)
        
        rate = prefactor * np.exp(-activation_energy / kt)
        return rate * 6.242e18  # Convert to s⁻¹
    
    def calculate_quantum_efficiency(self, conditions: PhotosyntheticConditions) -> Dict[str, float]:
        """
        Calculate quantum efficiency considering tunneling effects
        """
        efficiencies = {}
        
        for complex_name, reactions in self.complexes.items():
            complex_efficiency = 1.0
            
            for reaction_name, params in reactions.items():
                # Update parameters based on conditions
                updated_params = self._update_params_for_conditions(params, conditions)
                
                # Calculate tunneling rate
                rate = self.marcus_tunneling_rate(updated_params)
                
                # Convert rate to efficiency (simplified model)
                # Higher rates generally mean higher efficiency up to a saturation point
                efficiency = min(1.0, rate / 1e12)  # Normalize to reasonable scale
                complex_efficiency *= efficiency
                
            efficiencies[complex_name] = complex_efficiency
            
        return efficiencies
    
    def _update_params_for_conditions(self, params: QuantumTunnelingParameters, 
                                    conditions: PhotosyntheticConditions) -> QuantumTunnelingParameters:
        """
        Update tunneling parameters based on environmental conditions
        """
        updated_params = QuantumTunnelingParameters(
            barrier_height=params.barrier_height,
            barrier_width=params.barrier_width,
            particle_mass=params.particle_mass,
            temperature=conditions.temperature,
            reorganization_energy=params.reorganization_energy,
            driving_force=params.driving_force
        )
        
        # Temperature effects on barrier height (protein conformational changes)
        temp_factor = conditions.temperature / 298.0
        updated_params.barrier_height *= (1 + 0.1 * (temp_factor - 1))
        
        # pH effects on driving force
        ph_factor = (conditions.ph - 7.0) * 0.059  # Nernst equation factor
        updated_params.driving_force += ph_factor * 0.1
        
        # Light intensity effects on reorganization energy
        light_factor = min(1.5, conditions.light_intensity / 1000.0)
        updated_params.reorganization_energy *= (0.8 + 0.2 * light_factor)
        
        return updated_params
    
    def simulate_photosynthetic_pathway(self, conditions: PhotosyntheticConditions) -> Dict:
        """
        Simulate complete photosynthetic pathway with quantum tunneling
        """
        results = {
            'conditions': conditions,
            'quantum_efficiencies': {},
            'tunneling_rates': {},
            'overall_efficiency': 0.0,
            'limiting_factors': []
        }
        
        # Calculate individual complex efficiencies
        efficiencies = self.calculate_quantum_efficiency(conditions)
        results['quantum_efficiencies'] = efficiencies
        
        # Calculate tunneling rates for each step
        for complex_name, reactions in self.complexes.items():
            results['tunneling_rates'][complex_name] = {}
            for reaction_name, params in reactions.items():
                updated_params = self._update_params_for_conditions(params, conditions)
                rate = self.marcus_tunneling_rate(updated_params)
                results['tunneling_rates'][complex_name][reaction_name] = rate
        
        # Overall efficiency is product of individual efficiencies
        overall_eff = 1.0
        for eff in efficiencies.values():
            overall_eff *= eff
        
        # Apply environmental limiting factors
        light_limitation = min(1.0, conditions.light_intensity / 2000.0)
        temp_limitation = np.exp(-((conditions.temperature - 298)**2) / (2 * 15**2))
        co2_limitation = conditions.co2_concentration / (conditions.co2_concentration + 400)
        water_limitation = conditions.water_availability
        
        overall_eff *= light_limitation * temp_limitation * co2_limitation * water_limitation
        results['overall_efficiency'] = overall_eff
        
        # Identify limiting factors
        factors = {
            'light': light_limitation,
            'temperature': temp_limitation,
            'co2': co2_limitation,
            'water': water_limitation
        }
        
        min_factor = min(factors.values())
        results['limiting_factors'] = [name for name, val in factors.items() if val == min_factor]
        
        return results
    
    def parameter_sensitivity_analysis(self, base_conditions: PhotosyntheticConditions,
                                     parameter: str, range_values: List[float]) -> Dict:
        """
        Analyze sensitivity of photosynthesis to parameter changes
        """
        results = {
            'parameter': parameter,
            'values': range_values,
            'efficiencies': [],
            'tunneling_effects': []
        }
        
        for value in range_values:
            # Create modified conditions
            conditions = PhotosyntheticConditions(
                light_intensity=base_conditions.light_intensity,
                wavelength=base_conditions.wavelength,
                temperature=base_conditions.temperature,
                co2_concentration=base_conditions.co2_concentration,
                water_availability=base_conditions.water_availability,
                ph=base_conditions.ph,
                chlorophyll_concentration=base_conditions.chlorophyll_concentration
            )
            
            # Modify the specified parameter
            setattr(conditions, parameter, value)
            
            # Run simulation
            sim_result = self.simulate_photosynthetic_pathway(conditions)
            results['efficiencies'].append(sim_result['overall_efficiency'])
            
            # Calculate average tunneling effect
            avg_tunneling = np.mean([
                np.mean(list(rates.values())) 
                for rates in sim_result['tunneling_rates'].values()
            ])
            results['tunneling_effects'].append(avg_tunneling)
        
        return results
    
    def generate_integration_report(self, conditions: PhotosyntheticConditions) -> str:
        """
        Generate comprehensive report on quantum tunneling effects in photosynthesis
        """
        results = self.simulate_photosynthetic_pathway(conditions)
        
        report = f"""
QUANTUM TUNNELING IN PHOTOSYNTHESIS - INTEGRATION REPORT
========================================================

Environmental Conditions:
- Light Intensity: {conditions.light_intensity:.1f} μmol photons m⁻² s⁻¹
- Wavelength: {conditions.wavelength:.0f} nm
- Temperature: {conditions.temperature:.1f} K
- CO₂ Concentration: {conditions.co2_concentration:.0f} ppm
- Water Availability: {conditions.water_availability:.2f}
- pH: {conditions.ph:.1f}
- Chlorophyll Concentration: {conditions.chlorophyll_concentration:.2f} mg/L

Quantum Tunneling Analysis:
---------------------------
Overall Photosynthetic Efficiency: {results['overall_efficiency']:.3f}

Individual Complex Efficiencies:
"""
        
        for complex_name, efficiency in results['quantum_efficiencies'].items():
            report += f"- {complex_name.replace('_', ' ').title()}: {efficiency:.3f}\n"
        
        report += f"\nTunneling Rates (s⁻¹):\n"
        for complex_name, reactions in results['tunneling_rates'].items():
            report += f"- {complex_name.replace('_', ' ').title()}:\n"
            for reaction, rate in reactions.items():
                report += f"  * {reaction.replace('_', ' → ')}: {rate:.2e}\n"
        
        if results['limiting_factors']:
            report += f"\nLimiting Factors: {', '.join(results['limiting_factors'])}\n"
        
        report += """
Key Quantum Tunneling Effects:
------------------------------
1. Electron Transport: Quantum tunneling enables rapid charge separation
2. Proton Transfer: Facilitates ATP synthesis through membrane barriers
3. Energy Transfer: Coherent excitation transport in light-harvesting complexes
4. Temperature Dependence: Tunneling rates vary with thermal energy
5. Distance Sensitivity: Exponential dependence on barrier width

Integration Notes:
-----------------
- This module can be imported into existing quantum tunneling simulations
- Use simulate_photosynthetic_pathway() for complete analysis
- parameter_sensitivity_analysis() helps identify critical factors
- All rates calculated using Marcus theory with WKB tunneling approximation
        """
        
        return report

# Example usage and testing functions
def create_standard_conditions() -> PhotosyntheticConditions:
    """Create standard photosynthetic conditions for testing"""
    return PhotosyntheticConditions(
        light_intensity=1500.0,  # μmol photons m⁻² s⁻¹
        wavelength=680.0,        # nm (red light)
        temperature=298.0,       # K (25°C)
        co2_concentration=400.0, # ppm
        water_availability=1.0,  # fully available
        ph=7.0,                 # neutral
        chlorophyll_concentration=50.0  # mg/L
    )

def run_example_analysis():
    """Run example analysis showing quantum tunneling integration"""
    integrator = QuantumPhotosynthesisIntegrator()
    conditions = create_standard_conditions()
    
    # Generate full report
    report = integrator.generate_integration_report(conditions)
    print(report)
    
    # Run sensitivity analysis for temperature
    temp_range = np.linspace(273, 323, 20)  # 0°C to 50°C
    temp_analysis = integrator.parameter_sensitivity_analysis(
        conditions, 'temperature', temp_range.tolist()
    )
    
    print(f"\nTemperature Sensitivity Analysis:")
    print(f"Optimal temperature: {temp_range[np.argmax(temp_analysis['efficiencies'])]:.1f} K")
    print(f"Max efficiency: {max(temp_analysis['efficiencies']):.3f}")

if __name__ == "__main__":
    run_example_analysis()
