"""
Streamlit App for Quantum Tunneling in Photosynthesis
Run with: streamlit run streamlit_quantum_photosynthesis.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from quantum_photosynthesis import *

# Configure Streamlit page
st.set_page_config(
    page_title="Quantum Tunneling in Photosynthesis",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("Quantum Tunneling in Photosynthesis Simulator")
    st.markdown("""
    Explore how quantum tunneling effects influence photosynthetic efficiency under different environmental conditions.
    Adjust parameters in the sidebar and see real-time results!
    """)
    
    # Initialize the integrator
    if 'integrator' not in st.session_state:
        st.session_state.integrator = QuantumPhotosynthesisIntegrator()
    
    integrator = st.session_state.integrator
    
    # Sidebar for parameter controls
    st.sidebar.header("Environmental Parameters")
    
    # Environmental condition sliders
    light_intensity = st.sidebar.slider(
        "Light Intensity (μmol photons m⁻² s⁻¹)", 
        min_value=50, max_value=3000, value=1500, step=50,
        help="Photosynthetic photon flux density"
    )
    
    wavelength = st.sidebar.slider(
        "Wavelength (nm)", 
        min_value=400, max_value=750, value=680, step=10,
        help="Peak wavelength of incident light"
    )
    
    temperature = st.sidebar.slider(
        "Temperature (°C)", 
        min_value=-5, max_value=50, value=25, step=1,
        help="Ambient temperature"
    )
    
    co2_concentration = st.sidebar.slider(
        "CO₂ Concentration (ppm)", 
        min_value=200, max_value=1000, value=400, step=25,
        help="Atmospheric CO₂ level"
    )
    
    water_availability = st.sidebar.slider(
        "Water Availability", 
        min_value=0.1, max_value=1.0, value=1.0, step=0.05,
        help="Relative water availability (1.0 = fully hydrated)"
    )
    
    ph = st.sidebar.slider(
        "pH", 
        min_value=5.0, max_value=9.0, value=7.0, step=0.1,
        help="Soil/medium pH level"
    )
    
    chlorophyll_concentration = st.sidebar.slider(
        "Chlorophyll Concentration (mg/L)", 
        min_value=10, max_value=100, value=50, step=5,
        help="Chlorophyll content in leaves"
    )
    
    # Create conditions from sidebar inputs
    conditions = PhotosyntheticConditions(
        light_intensity=float(light_intensity),
        wavelength=float(wavelength),
        temperature=float(temperature + 273.15),  # Convert to Kelvin
        co2_concentration=float(co2_concentration),
        water_availability=float(water_availability),
        ph=float(ph),
        chlorophyll_concentration=float(chlorophyll_concentration)
    )
    
    # Main content area with tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Current Analysis", 
        "Temperature Effects", 
        "Light Response", 
        "Complex Details",
        "Parameter Sweep"
    ])
    
    with tab1:
        st.header("Current Conditions Analysis")
        
        # Run simulation
        with st.spinner("Running quantum tunneling simulation..."):
            results = integrator.simulate_photosynthetic_pathway(conditions)
        
        # Display key metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Overall Efficiency", 
                f"{results['overall_efficiency']:.4f}",
                help="Combined quantum efficiency of all photosynthetic complexes"
            )
        
        with col2:
            limiting_factors = ", ".join(results['limiting_factors']) if results['limiting_factors'] else "None"
            st.metric(
                "Limiting Factor(s)", 
                limiting_factors,
                help="Environmental factors currently limiting photosynthesis"
            )
        
        with col3:
            best_complex = max(results['quantum_efficiencies'], key=results['quantum_efficiencies'].get)
            st.metric(
                "Best Performing Complex", 
                best_complex.replace('_', ' ').title(),
                help="Photosynthetic complex with highest quantum efficiency"
            )
        
        with col4:
            avg_rate = np.mean([
                np.mean(list(rates.values())) 
                for rates in results['tunneling_rates'].values()
            ])
            st.metric(
                "Avg Tunneling Rate", 
                f"{avg_rate:.2e} s⁻¹",
                help="Average quantum tunneling rate across all complexes"
            )
        
        # Complex efficiencies chart
        st.subheader("Individual Complex Efficiencies")
        
        complex_data = pd.DataFrame([
            {"Complex": name.replace('_', ' ').title(), "Efficiency": eff}
            for name, eff in results['quantum_efficiencies'].items()
        ])
        
        fig = px.bar(
            complex_data, x="Complex", y="Efficiency",
            title="Quantum Efficiency by Photosynthetic Complex",
            color="Efficiency",
            color_continuous_scale="Viridis"
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Tunneling rates heatmap
        st.subheader("Quantum Tunneling Rates")
        
        tunneling_data = []
        for complex_name, reactions in results['tunneling_rates'].items():
            for reaction, rate in reactions.items():
                tunneling_data.append({
                    "Complex": complex_name.replace('_', ' ').title(),
                    "Reaction": reaction.replace('_', ' → '),
                    "Rate (log₁₀ s⁻¹)": np.log10(rate)
                })
        
        tunneling_df = pd.DataFrame(tunneling_data)
        
        if not tunneling_df.empty:
            fig = px.scatter(
                tunneling_df, x="Complex", y="Reaction", 
                size="Rate (log₁₀ s⁻¹)", color="Rate (log₁₀ s⁻¹)",
                title="Tunneling Rates Across Photosynthetic Complexes",
                color_continuous_scale="Plasma"
            )
            fig.update_traces(marker=dict(sizemin=10, sizemax=30))
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("Temperature Effects on Quantum Tunneling")
        
        temp_range = np.linspace(-5, 45, 26)  # -5°C to 45°C
        
        with st.spinner("Computing temperature response..."):
            temp_efficiencies = []
            temp_rates = []
            
            for temp_c in temp_range:
                temp_conditions = PhotosyntheticConditions(
                    light_intensity=conditions.light_intensity,
                    wavelength=conditions.wavelength,
                    temperature=temp_c + 273.15,
                    co2_concentration=conditions.co2_concentration,
                    water_availability=conditions.water_availability,
                    ph=conditions.ph,
                    chlorophyll_concentration=conditions.chlorophyll_concentration
                )
                
                temp_result = integrator.simulate_photosynthetic_pathway(temp_conditions)
                temp_efficiencies.append(temp_result['overall_efficiency'])
                
                avg_rate = np.mean([
                    np.mean(list(rates.values())) 
                    for rates in temp_result['tunneling_rates'].values()
                ])
                temp_rates.append(avg_rate)
        
        # Create dual-axis plot
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Efficiency curve
        fig.add_trace(
            go.Scatter(x=temp_range, y=temp_efficiencies, name="Quantum Efficiency", 
                      line=dict(color="green", width=3)),
            secondary_y=False,
        )
        
        # Tunneling rate curve
        fig.add_trace(
            go.Scatter(x=temp_range, y=temp_rates, name="Avg Tunneling Rate", 
                      line=dict(color="red", width=2, dash="dash")),
            secondary_y=True,
        )
        
        # Current temperature indicator
        current_temp = temperature
        current_eff = temp_efficiencies[np.argmin(np.abs(temp_range - current_temp))]
        fig.add_vline(x=current_temp, line_dash="dot", line_color="blue", 
                     annotation_text=f"Current: {current_temp}°C")
        
        fig.update_xaxes(title_text="Temperature (°C)")
        fig.update_yaxes(title_text="Quantum Efficiency", secondary_y=False, color="green")
        fig.update_yaxes(title_text="Average Tunneling Rate (s⁻¹)", secondary_y=True, color="red")
        fig.update_layout(title_text="Temperature Response of Quantum Photosynthesis")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Optimal temperature
        optimal_temp = temp_range[np.argmax(temp_efficiencies)]
        max_efficiency = max(temp_efficiencies)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Optimal Temperature", f"{optimal_temp:.1f}°C")
        with col2:
            st.metric("Maximum Efficiency", f"{max_efficiency:.4f}")
    
    with tab3:
        st.header("Light Intensity Response")
        
        light_range = np.linspace(50, 3000, 30)
        
        with st.spinner("Computing light response curve..."):
            light_efficiencies = []
            
            for light in light_range:
                light_conditions = PhotosyntheticConditions(
                    light_intensity=light,
                    wavelength=conditions.wavelength,
                    temperature=conditions.temperature,
                    co2_concentration=conditions.co2_concentration,
                    water_availability=conditions.water_availability,
                    ph=conditions.ph,
                    chlorophyll_concentration=conditions.chlorophyll_concentration
                )
                
                light_result = integrator.simulate_photosynthetic_pathway(light_conditions)
                light_efficiencies.append(light_result['overall_efficiency'])
        
        # Light response curve
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=light_range, y=light_efficiencies,
            mode='lines+markers',
            name='Quantum Efficiency',
            line=dict(color='orange', width=3),
            marker=dict(size=4)
        ))
        
        # Current light level indicator
        fig.add_vline(x=light_intensity, line_dash="dot", line_color="blue", 
                     annotation_text=f"Current: {light_intensity} μmol m⁻² s⁻¹")
        
        fig.update_layout(
            title="Light Saturation Curve with Quantum Tunneling Effects",
            xaxis_title="Light Intensity (μmol photons m⁻² s⁻¹)",
            yaxis_title="Quantum Efficiency"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Light saturation point
        saturation_idx = np.where(np.diff(light_efficiencies) < 0.0001)[0]
        if len(saturation_idx) > 0:
            saturation_light = light_range[saturation_idx[0]]
            st.info(f"Light saturation begins around {saturation_light:.0f} μmol m⁻² s⁻¹")
    
    with tab4:
        st.header("Photosynthetic Complex Details")
        
        # Complex selection
        complex_names = list(integrator.complexes.keys())
        selected_complex = st.selectbox(
            "Select Complex to Analyze:",
            complex_names,
            format_func=lambda x: x.replace('_', ' ').title()
        )
        
        complex_reactions = integrator.complexes[selected_complex]
        
        st.subheader(f"Analysis: {selected_complex.replace('_', ' ').title()}")
        
        # Create detailed analysis for selected complex
        reaction_data = []
        for reaction_name, params in complex_reactions.items():
            updated_params = integrator._update_params_for_conditions(params, conditions)
            
            tunneling_prob = integrator.calculate_tunneling_probability(updated_params)
            tunneling_rate = integrator.marcus_tunneling_rate(updated_params)
            
            reaction_data.append({
                "Reaction": reaction_name.replace('_', ' → '),
                "Barrier Height (eV)": updated_params.barrier_height,
                "Barrier Width (nm)": updated_params.barrier_width,
                "Tunneling Probability": tunneling_prob,
                "Tunneling Rate (s⁻¹)": tunneling_rate,
                "Reorganization Energy (eV)": updated_params.reorganization_energy,
                "Driving Force (eV)": updated_params.driving_force
            })
        
        reaction_df = pd.DataFrame(reaction_data)
        st.dataframe(reaction_df, use_container_width=True)
        
        # Visualize barrier properties
        if len(reaction_data) > 1:
            fig = px.scatter(
                reaction_df, x="Barrier Width (nm)", y="Barrier Height (eV)",
                size="Tunneling Rate (s⁻¹)", color="Tunneling Probability",
                hover_data=["Reaction"],
                title=f"Barrier Properties in {selected_complex.replace('_', ' ').title()}",
                color_continuous_scale="Viridis"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.header("Parameter Sweep Analysis")
        
        # Parameter selection for sweep
        sweep_param = st.selectbox(
            "Select Parameter for Sweep:",
            ["temperature", "light_intensity", "co2_concentration", "ph", "water_availability"],
            format_func=lambda x: x.replace('_', ' ').title()
        )
        
        # Define parameter ranges
        param_ranges = {
            "temperature": (263, 323, "K"),  # -10°C to 50°C
            "light_intensity": (50, 3000, "μmol m⁻² s⁻¹"),
            "co2_concentration": (200, 1000, "ppm"),
            "ph": (5.0, 9.0, ""),
            "water_availability": (0.1, 1.0, "")
        }
        
        min_val, max_val, unit = param_ranges[sweep_param]
        
        with st.spinner(f"Performing {sweep_param} sweep..."):
            if sweep_param == "temperature":
                param_values = np.linspace(min_val, max_val, 20)
                display_values = param_values - 273.15  # Convert to Celsius for display
                display_unit = "°C"
            else:
                param_values = np.linspace(min_val, max_val, 20)
                display_values = param_values
                display_unit = unit
            
            sweep_results = integrator.parameter_sensitivity_analysis(
                conditions, sweep_param, param_values.tolist()
            )
        
        # Plot results
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Scatter(x=display_values, y=sweep_results['efficiencies'], 
                      name="Quantum Efficiency", mode='lines+markers',
                      line=dict(color="green", width=3)),
            secondary_y=False,
        )
        
        fig.add_trace(
            go.Scatter(x=display_values, y=sweep_results['tunneling_effects'], 
                      name="Avg Tunneling Rate", mode='lines+markers',
                      line=dict(color="red", width=2, dash="dash")),
            secondary_y=True,
        )
        
        fig.update_xaxes(title_text=f"{sweep_param.replace('_', ' ').title()} ({display_unit})")
        fig.update_yaxes(title_text="Quantum Efficiency", secondary_y=False, color="green")
        fig.update_yaxes(title_text="Average Tunneling Rate (s⁻¹)", secondary_y=True, color="red")
        fig.update_layout(title_text=f"Parameter Sensitivity: {sweep_param.replace('_', ' ').title()}")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Optimal value
        optimal_idx = np.argmax(sweep_results['efficiencies'])
        optimal_value = display_values[optimal_idx]
        optimal_efficiency = sweep_results['efficiencies'][optimal_idx]
        
        st.success(f"Optimal {sweep_param.replace('_', ' ')}: {optimal_value:.2f} {display_unit} "
                  f"(Efficiency: {optimal_efficiency:.4f})")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    ### About This Simulator
    
    This application models quantum tunneling effects in photosynthetic electron transport chains. 
    It uses Marcus theory combined with WKB tunneling approximations to calculate realistic 
    tunneling rates and efficiencies for major photosynthetic complexes:
    
    - **Photosystem II**: P680 → Pheophytin → QA
    - **Cytochrome b6f**: Rieske center → Cytochrome f  
    - **Photosystem I**: P700 → A0 → A1
    - **ATP Synthase**: Proton tunneling through c-ring
    
    The model accounts for environmental factors including temperature, light intensity, 
    CO₂ concentration, pH, and water availability.
    """)
    
    # Download report button
    if st.button("Generate Full Report"):
        report = integrator.generate_integration_report(conditions)
        st.download_button(
            label="Download Integration Report",
            data=report,
            file_name=f"quantum_photosynthesis_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

if __name__ == "__main__":
    main()
