import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

# Set page configuration
st.set_page_config(
    page_title="Quantum Tunneling Simulation",
    page_icon="⚛️",
    layout="wide"
)

# Constants
hbar = 1.0545718e-34  # Reduced Planck's constant (J·s)
eV_to_J = 1.602176634e-19  # Conversion factor from eV to Joules

def calculate_wave_numbers(E, V0, m):
    """Calculate wave numbers for different regions"""
    k1 = np.sqrt(2 * m * E) / hbar if E > 0 else 0
    
    if E > V0:
        k2 = np.sqrt(2 * m * (E - V0)) / hbar
    else:
        k2 = 1j * np.sqrt(2 * m * (V0 - E)) / hbar
    
    k3 = k1  # Same as region 1 for x > barrier
    
    return k1, k2, k3

def calculate_transmission_coefficients(E, V0, width, m):
    """Calculate transmission and reflection coefficients"""
    k1, k2, k3 = calculate_wave_numbers(E, V0, m)
    
    if E <= 0:
        return 0, 1
    
    if E > V0:
        # Classical case - particle has enough energy
        # Still some reflection due to potential step
        
        # Account for barrier width using transmission matrix method
        phase = k2 * width
        denominator = 1 + ((k1**2 - k2**2)**2 / (4 * k1**2 * k2**2)) * np.sin(phase)**2
        T = 1 / denominator
        R = 1 - T
    else:
        # Quantum tunneling case
        kappa = np.abs(k2)  # k2 is imaginary in this case
        
        # Avoid division by zero
        if k1 == 0 or kappa == 0:
            T = 0
            R = 1
        else:
            # Transmission coefficient for tunneling
            gamma = (k1**2 + kappa**2) / (2 * k1 * kappa)
            sinh_term = np.sinh(kappa * width)
            T = 1 / (1 + (gamma**2) * sinh_term**2)
            R = 1 - T
    
    return T, R

def wave_function(x, E, V0, barrier_start, barrier_width, m):
    """Calculate the wave function in all regions"""
    k1, k2, k3 = calculate_wave_numbers(E, V0, m)
    
    # Initialize wave function array
    psi = np.zeros(len(x), dtype=complex)
    
    # Transmission and reflection coefficients
    T, R = calculate_transmission_coefficients(E, V0, barrier_width, m)
    
    for i, xi in enumerate(x):
        if xi < barrier_start:
            # Region 1: Before barrier (incident + reflected wave)
            psi[i] = np.exp(1j * k1 * xi) + np.sqrt(R) * np.exp(-1j * k1 * xi)
        elif xi < barrier_start + barrier_width:
            # Region 2: Inside barrier
            if E > V0:
                # Oscillatory solution
                psi[i] = np.sqrt(T) * np.exp(1j * k2 * (xi - barrier_start))
            else:
                # Exponential decay/growth
                psi[i] = np.sqrt(T) * np.exp(-np.abs(k2) * (xi - barrier_start))
        else:
            # Region 3: After barrier (transmitted wave only)
            psi[i] = np.sqrt(T) * np.exp(1j * k3 * xi)
    
    return psi

def create_potential_array(x, V0, barrier_start, barrier_width):
    """Create potential energy array"""
    V = np.zeros(len(x))
    mask = (x >= barrier_start) & (x <= barrier_start + barrier_width)
    V[mask] = V0
    return V

def main():
    st.title("⚛️ Interactive Quantum Tunneling Simulation")
    st.markdown("""
    This simulation demonstrates quantum tunneling, where a particle can pass through 
    a potential barrier even when it doesn't have enough classical energy to go over it.
    """)
    
    # Initialize session state for tracking experiments
    if 'experiment_history' not in st.session_state:
        st.session_state.experiment_history = []
    if 'experiment_counter' not in st.session_state:
        st.session_state.experiment_counter = 1
    
    # Sidebar for parameters
    st.sidebar.header("Simulation Parameters")
    
    # Particle parameters
    st.sidebar.subheader("Particle Properties")
    mass_factor = st.sidebar.slider(
        "Particle Mass (relative to electron mass)",
        min_value=0.1,
        max_value=10.0,
        value=1.0,
        step=0.1,
        help="Mass of the particle relative to electron mass (9.109 × 10⁻³¹ kg)"
    )
    m = mass_factor * 9.10938356e-31
    
    energy_eV = st.sidebar.slider(
        "Particle Energy (eV)",
        min_value=0.1,
        max_value=10.0,
        value=1.0,
        step=0.1,
        help="Kinetic energy of the particle in electron volts"
    )
    E = energy_eV * eV_to_J
    
    # Barrier parameters
    st.sidebar.subheader("Potential Barrier")
    barrier_height_eV = st.sidebar.slider(
        "Barrier Height (eV)",
        min_value=0.5,
        max_value=15.0,
        value=2.0,
        step=0.1,
        help="Height of the potential barrier in electron volts"
    )
    V0 = barrier_height_eV * eV_to_J
    
    barrier_width_nm = st.sidebar.slider(
        "Barrier Width (nm)",
        min_value=0.1,
        max_value=5.0,
        value=1.0,
        step=0.1,
        help="Width of the potential barrier in nanometers"
    )
    barrier_width = barrier_width_nm * 1e-9
    
    barrier_position_nm = st.sidebar.slider(
        "Barrier Position (nm)",
        min_value=-2.0,
        max_value=2.0,
        value=0.0,
        step=0.1,
        help="Position of the barrier center in nanometers"
    )
    barrier_start = (barrier_position_nm * 1e-9) - (barrier_width / 2)
    
    # Simulation range
    x_range_nm = st.sidebar.slider(
        "Simulation Range (±nm)",
        min_value=2.0,
        max_value=10.0,
        value=5.0,
        step=0.5,
        help="Total range of the simulation in nanometers"
    )
    
    # Create position array
    x = np.linspace(-x_range_nm * 1e-9, x_range_nm * 1e-9, 2000)
    
    # Calculate potential and wave function
    V = create_potential_array(x, V0, barrier_start, barrier_width)
    psi = wave_function(x, E, V0, barrier_start, barrier_width, m)
    
    # Calculate transmission and reflection coefficients
    T, R = calculate_transmission_coefficients(E, V0, barrier_width, m)
    
    # Create the main plot
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Potential Barrier and Wave Function', 'Probability Density'),
        vertical_spacing=0.15,
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
    )
    
    # Convert to nanometers for plotting
    x_nm = x * 1e9
    V_eV = V / eV_to_J
    E_eV = E / eV_to_J
    
    # Plot potential barrier
    fig.add_trace(
        go.Scatter(
            x=x_nm, y=V_eV,
            mode='lines',
            name='Potential Barrier',
            line=dict(color='blue', width=3),
            fill='tonexty'
        ),
        row=1, col=1
    )
    
    # Plot energy level
    fig.add_trace(
        go.Scatter(
            x=[x_nm[0], x_nm[-1]], y=[E_eV, E_eV],
            mode='lines',
            name=f'Particle Energy ({E_eV:.2f} eV)',
            line=dict(color='red', width=2, dash='dash')
        ),
        row=1, col=1
    )
    
    # Plot wave function (real part)
    psi_real_scaled = np.real(psi) * E_eV * 0.5 + E_eV
    fig.add_trace(
        go.Scatter(
            x=x_nm, y=psi_real_scaled,
            mode='lines',
            name='Wave Function (Real)',
            line=dict(color='green', width=2)
        ),
        row=1, col=1
    )
    
    # Plot probability density
    prob_density = np.abs(psi)**2
    prob_density_normalized = prob_density / np.max(prob_density) if np.max(prob_density) > 0 else prob_density
    
    fig.add_trace(
        go.Scatter(
            x=x_nm, y=prob_density_normalized,
            mode='lines',
            name='Probability Density |ψ|²',
            line=dict(color='purple', width=3),
            fill='tonexty'
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        title_text="Quantum Tunneling Simulation",
        showlegend=True
    )
    
    # Update x-axis labels
    fig.update_xaxes(title_text="Position (nm)", row=2, col=1)
    fig.update_xaxes(title_text="Position (nm)", row=1, col=1)
    
    # Update y-axis labels
    fig.update_yaxes(title_text="Energy (eV)", row=1, col=1)
    fig.update_yaxes(title_text="Normalized Probability", row=2, col=1)
    
    # Display the plot
    st.plotly_chart(fig, use_container_width=True)
    
    # Display results
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Transmission Probability",
            value=f"{T:.4f}",
            help="Probability that the particle tunnels through the barrier"
        )
    
    with col2:
        st.metric(
            label="Reflection Probability",
            value=f"{R:.4f}",
            help="Probability that the particle is reflected by the barrier"
        )
    
    with col3:
        # Check both energy condition AND actual transmission
        is_quantum_tunneling = (E < V0) and (T > 1e-10)  # Very small threshold to avoid numerical issues
        tunneling_regime = "Yes" if is_quantum_tunneling else "No"
        
        # More descriptive help text
        if E >= V0:
            help_text = "Particle has enough energy - this is classical transmission, not tunneling"
        elif T <= 1e-10:
            help_text = "Energy too low or barrier too thick - no effective tunneling occurring"
        else:
            help_text = "Quantum tunneling is occurring - particle passes through despite insufficient energy"
        
        st.metric(
            label="Quantum Tunneling",
            value=tunneling_regime,
            help=help_text
        )
    
    with col4:
        # Save experiment button
        if st.button("📊 Save Experiment", help="Save current settings to experiment history"):
            experiment_data = {
                'ID': st.session_state.experiment_counter,
                'Energy (eV)': energy_eV,
                'Barrier Height (eV)': barrier_height_eV,
                'Barrier Width (nm)': barrier_width_nm,
                'Mass Factor': mass_factor,
                'Transmission': T,
                'Reflection': R,
                'Tunneling': tunneling_regime,
                'Timestamp': pd.Timestamp.now().strftime('%H:%M:%S')
            }
            st.session_state.experiment_history.append(experiment_data)
            st.session_state.experiment_counter += 1
            st.success(f"Experiment #{experiment_data['ID']} saved!")
        
        # Show number of saved experiments
        st.metric(
            label="Saved Experiments",
            value=len(st.session_state.experiment_history),
            help="Number of experiments you've saved"
        )
    
    # Parameter Variation Analysis
    st.markdown("---")
    st.subheader("Tunneling Probability Analysis")
    
    # Create tabs for different parameter variations
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Barrier Height", "Barrier Width", "Particle Energy", "Parameter Table", "Heatmap View", "Comparison Chart", "Experiment Tracker"])
    
    with tab1:
        st.write("How transmission probability changes with barrier height:")
        heights_eV = np.linspace(0.5, 10.0, 50)
        heights_J = heights_eV * eV_to_J
        transmissions_height = []
        
        for V_test in heights_J:
            T_test, _ = calculate_transmission_coefficients(E, V_test, barrier_width, m)
            transmissions_height.append(T_test)
        
        fig_height = go.Figure()
        fig_height.add_trace(
            go.Scatter(
                x=heights_eV,
                y=transmissions_height,
                mode='lines',
                name='Transmission Probability',
                line=dict(color='orange', width=3)
            )
        )
        
        # Mark current point
        fig_height.add_trace(
            go.Scatter(
                x=[barrier_height_eV],
                y=[T],
                mode='markers',
                name='Current Setting',
                marker=dict(color='red', size=10, symbol='circle')
            )
        )
        
        fig_height.update_layout(
            xaxis_title="Barrier Height (eV)",
            yaxis_title="Transmission Probability",
            title=f"Transmission vs Barrier Height (Energy = {energy_eV:.1f} eV)"
        )
        st.plotly_chart(fig_height, use_container_width=True)
    
    with tab2:
        st.write("How transmission probability changes with barrier width:")
        widths_nm = np.linspace(0.1, 5.0, 50)
        widths_m = widths_nm * 1e-9
        transmissions_width = []
        
        for w_test in widths_m:
            T_test, _ = calculate_transmission_coefficients(E, V0, w_test, m)
            transmissions_width.append(T_test)
        
        fig_width = go.Figure()
        fig_width.add_trace(
            go.Scatter(
                x=widths_nm,
                y=transmissions_width,
                mode='lines',
                name='Transmission Probability',
                line=dict(color='green', width=3)
            )
        )
        
        # Mark current point
        fig_width.add_trace(
            go.Scatter(
                x=[barrier_width_nm],
                y=[T],
                mode='markers',
                name='Current Setting',
                marker=dict(color='red', size=10, symbol='circle')
            )
        )
        
        fig_width.update_layout(
            xaxis_title="Barrier Width (nm)",
            yaxis_title="Transmission Probability",
            title=f"Transmission vs Barrier Width (Height = {barrier_height_eV:.1f} eV)"
        )
        st.plotly_chart(fig_width, use_container_width=True)
    
    with tab3:
        st.write("How transmission probability changes with particle energy:")
        energies_eV = np.linspace(0.1, 10.0, 50)
        energies_J = energies_eV * eV_to_J
        transmissions_energy = []
        
        for E_test in energies_J:
            T_test, _ = calculate_transmission_coefficients(E_test, V0, barrier_width, m)
            transmissions_energy.append(T_test)
        
        fig_energy = go.Figure()
        fig_energy.add_trace(
            go.Scatter(
                x=energies_eV,
                y=transmissions_energy,
                mode='lines',
                name='Transmission Probability',
                line=dict(color='purple', width=3)
            )
        )
        
        # Mark current point
        fig_energy.add_trace(
            go.Scatter(
                x=[energy_eV],
                y=[T],
                mode='markers',
                name='Current Setting',
                marker=dict(color='red', size=10, symbol='circle')
            )
        )
        
        # Mark barrier height
        fig_energy.add_vline(
            x=barrier_height_eV,
            line_dash="dash",
            line_color="blue",
            annotation_text="Barrier Height"
        )
        
        fig_energy.update_layout(
            xaxis_title="Particle Energy (eV)",
            yaxis_title="Transmission Probability",
            title=f"Transmission vs Particle Energy (Barrier = {barrier_height_eV:.1f} eV)"
        )
        st.plotly_chart(fig_energy, use_container_width=True)
    
    with tab4:
        st.write("Parameter combinations and their tunneling probabilities:")
        
        # Create a parameter variation table
        param_data = []
        
        # Test different combinations
        test_heights = [1.0, 2.0, 3.0, 5.0, 8.0]
        test_widths = [0.5, 1.0, 2.0, 3.0]
        test_energies = [0.5, 1.0, 1.5, 2.0]
        
        for h_eV in test_heights[:3]:  # Limit to avoid too many rows
            for w_nm in test_widths[:3]:
                for e_eV in test_energies[:3]:
                    E_test = e_eV * eV_to_J
                    V_test = h_eV * eV_to_J
                    w_test = w_nm * 1e-9
                    
                    T_test, R_test = calculate_transmission_coefficients(E_test, V_test, w_test, m)
                    
                    param_data.append({
                        'Energy (eV)': e_eV,
                        'Barrier Height (eV)': h_eV,
                        'Barrier Width (nm)': w_nm,
                        'Transmission': f"{T_test:.4f}",
                        'Tunneling': 'Yes' if e_eV < h_eV else 'No'
                    })
        
        df = pd.DataFrame(param_data)
        
        # Highlight current parameters
        current_row_style = []
        for _, row in df.iterrows():
            if (abs(float(row['Energy (eV)']) - energy_eV) < 0.1 and 
                abs(float(row['Barrier Height (eV)']) - barrier_height_eV) < 0.1 and 
                abs(float(row['Barrier Width (nm)']) - barrier_width_nm) < 0.1):
                current_row_style.append('background-color: #ffeb3b')
            else:
                current_row_style.append('')
        
        st.dataframe(df, use_container_width=True)
        st.caption("Yellow highlight indicates current parameter settings")
    
    with tab5:
        st.write("2D Heatmap: Transmission probability across parameter combinations")
        
        # Create heatmap data
        height_range = np.linspace(0.5, 5.0, 20)
        width_range = np.linspace(0.2, 3.0, 20)
        heatmap_data = np.zeros((len(height_range), len(width_range)))
        
        for i, h_eV in enumerate(height_range):
            for j, w_nm in enumerate(width_range):
                V_test = h_eV * eV_to_J
                w_test = w_nm * 1e-9
                T_test, _ = calculate_transmission_coefficients(E, V_test, w_test, m)
                heatmap_data[i, j] = T_test
        
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            x=width_range,
            y=height_range,
            colorscale='Viridis',
            colorbar=dict(title="Transmission Probability")
        ))
        
        # Mark current point
        fig_heatmap.add_trace(
            go.Scatter(
                x=[barrier_width_nm],
                y=[barrier_height_eV],
                mode='markers',
                name='Current Setting',
                marker=dict(color='red', size=15, symbol='x', line=dict(width=3, color='white'))
            )
        )
        
        fig_heatmap.update_layout(
            title=f"Transmission Heatmap (Energy = {energy_eV:.1f} eV)",
            xaxis_title="Barrier Width (nm)",
            yaxis_title="Barrier Height (eV)"
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Add percentage breakdown
        st.subheader("Current Settings Breakdown")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if E < V0:
                tunneling_strength = "Strong" if T > 0.1 else "Moderate" if T > 0.01 else "Weak"
                color = "green" if T > 0.1 else "orange" if T > 0.01 else "red"
            else:
                tunneling_strength = "Classical"
                color = "blue"
            
            st.markdown(f"""
            **Tunneling Strength**: <span style="color:{color}; font-weight:bold">{tunneling_strength}</span>
            
            **Transmission**: {T*100:.2f}%  
            **Reflection**: {R*100:.2f}%
            """, unsafe_allow_html=True)
        
        with col2:
            # Energy ratio
            energy_ratio = energy_eV / barrier_height_eV
            ratio_color = "green" if energy_ratio < 1 else "blue"
            
            st.markdown(f"""
            **Energy vs Barrier**:  
            <span style="color:{ratio_color}; font-weight:bold">{energy_ratio:.2f}x</span>
            
            Particle: {energy_eV:.1f} eV  
            Barrier: {barrier_height_eV:.1f} eV
            """, unsafe_allow_html=True)
        
        with col3:
            # Barrier thickness effect
            penetration_depth = hbar / np.sqrt(2 * m * (V0 - E)) if E < V0 else "N/A (Classical)"
            if isinstance(penetration_depth, float):
                depth_nm = penetration_depth * 1e9
                thickness_ratio = barrier_width_nm / depth_nm
                st.markdown(f"""
                **Barrier Effect**:  
                Width: {barrier_width_nm:.1f} nm  
                Penetration: {depth_nm:.2f} nm  
                Ratio: {thickness_ratio:.1f}x
                """)
            else:
                st.markdown(f"""
                **Barrier Effect**:  
                Width: {barrier_width_nm:.1f} nm  
                Mode: Classical  
                (No penetration depth)
                """)
    
    with tab6:
        st.write("Side-by-side comparison of different scenarios")
        
        # Create comparison scenarios
        scenarios = [
            {"name": "Current", "energy": energy_eV, "height": barrier_height_eV, "width": barrier_width_nm},
            {"name": "Thinner Barrier", "energy": energy_eV, "height": barrier_height_eV, "width": barrier_width_nm * 0.5},
            {"name": "Lower Barrier", "energy": energy_eV, "height": barrier_height_eV * 0.7, "width": barrier_width_nm},
            {"name": "Higher Energy", "energy": energy_eV * 1.5, "height": barrier_height_eV, "width": barrier_width_nm}
        ]
        
        comparison_data = []
        for scenario in scenarios:
            E_test = scenario["energy"] * eV_to_J
            V_test = scenario["height"] * eV_to_J
            w_test = scenario["width"] * 1e-9
            
            T_test, R_test = calculate_transmission_coefficients(E_test, V_test, w_test, m)
            
            comparison_data.append({
                "Scenario": scenario["name"],
                "Transmission %": T_test * 100,
                "Reflection %": R_test * 100,
                "Energy (eV)": scenario["energy"],
                "Barrier Height (eV)": scenario["height"],
                "Barrier Width (nm)": scenario["width"]
            })
        
        # Create bar chart
        df_comp = pd.DataFrame(comparison_data)
        
        fig_bar = go.Figure()
        
        fig_bar.add_trace(go.Bar(
            name='Transmission %',
            x=df_comp['Scenario'],
            y=df_comp['Transmission %'],
            marker_color='lightgreen'
        ))
        
        fig_bar.add_trace(go.Bar(
            name='Reflection %',
            x=df_comp['Scenario'],
            y=df_comp['Reflection %'],
            marker_color='lightcoral'
        ))
        
        fig_bar.update_layout(
            title="Transmission vs Reflection Comparison",
            xaxis_title="Scenario",
            yaxis_title="Probability (%)",
            barmode='stack'
        )
        
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # Show detailed comparison table
        st.subheader("Detailed Comparison")
        st.dataframe(df_comp, use_container_width=True)
        
        # Add insights
        best_transmission = df_comp.loc[df_comp['Transmission %'].idxmax()]
        st.success(f"""
        **Best Transmission**: {best_transmission['Scenario']} with {best_transmission['Transmission %']:.2f}%
        
        **Key Insight**: {"Reducing barrier width" if "Thinner" in best_transmission['Scenario'] else 
                         "Lowering barrier height" if "Lower" in best_transmission['Scenario'] else
                         "Increasing particle energy" if "Higher" in best_transmission['Scenario'] else
                         "Current settings are optimal"} gives the highest tunneling probability.
        """)
    
    with tab7:
        st.write("Track and compare all your barrier experiments")
        
        if len(st.session_state.experiment_history) == 0:
            st.info("No experiments saved yet. Use the 'Save Experiment' button above to start tracking your results!")
        else:
            # Create DataFrame from experiment history
            df_experiments = pd.DataFrame(st.session_state.experiment_history)
            
            # Show summary statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                avg_transmission = df_experiments['Transmission'].mean()
                st.metric("Average Transmission", f"{avg_transmission:.4f}")
            
            with col2:
                best_transmission = df_experiments['Transmission'].max()
                st.metric("Best Transmission", f"{best_transmission:.4f}")
            
            with col3:
                tunneling_count = (df_experiments['Tunneling'] == 'Yes').sum()
                st.metric("Tunneling Experiments", f"{tunneling_count}/{len(df_experiments)}")
            
            # Interactive experiment table
            st.subheader("Your Experiment History")
            
            # Add sorting options
            sort_column = st.selectbox("Sort by:", 
                                     ['ID', 'Transmission', 'Energy (eV)', 'Barrier Height (eV)', 'Barrier Width (nm)'])
            sort_ascending = st.checkbox("Ascending order", value=False)
            
            df_sorted = df_experiments.sort_values(sort_column, ascending=sort_ascending)
            st.dataframe(df_sorted, use_container_width=True)
            
            # Plot transmission history
            st.subheader("Transmission Over Time")
            fig_history = go.Figure()
            
            # Color points by tunneling regime
            colors = ['green' if x == 'Yes' else 'blue' for x in df_experiments['Tunneling']]
            
            fig_history.add_trace(go.Scatter(
                x=df_experiments['ID'],
                y=df_experiments['Transmission'],
                mode='markers+lines',
                marker=dict(color=colors, size=10),
                name='Transmission',
                text=[f"Exp {row['ID']}: E={row['Energy (eV)']}eV, H={row['Barrier Height (eV)']}eV, W={row['Barrier Width (nm)']}nm" 
                      for _, row in df_experiments.iterrows()],
                hovertemplate='<b>Experiment %{x}</b><br>' +
                             'Transmission: %{y:.4f}<br>' +
                             '%{text}<br>' +
                             '<extra></extra>'
            ))
            
            fig_history.update_layout(
                title="Transmission Probability Across Your Experiments",
                xaxis_title="Experiment Number",
                yaxis_title="Transmission Probability",
                showlegend=False
            )
            
            # Add color legend
            fig_history.add_trace(go.Scatter(
                x=[None], y=[None], mode='markers',
                marker=dict(color='green', size=10),
                showlegend=True, name='Tunneling (Yes)'
            ))
            fig_history.add_trace(go.Scatter(
                x=[None], y=[None], mode='markers',
                marker=dict(color='blue', size=10),
                showlegend=True, name='Classical (No)'
            ))
            
            st.plotly_chart(fig_history, use_container_width=True)
            
            # Find patterns in successful experiments
            if tunneling_count > 0:
                tunneling_experiments = df_experiments[df_experiments['Tunneling'] == 'Yes'].copy()
                
                st.subheader("Successful Tunneling Patterns")
                col1, col2 = st.columns(2)
                
                with col1:
                    avg_energy = tunneling_experiments['Energy (eV)'].mean()
                    avg_height = tunneling_experiments['Barrier Height (eV)'].mean()
                    avg_width = tunneling_experiments['Barrier Width (nm)'].mean()
                    
                    st.markdown(f"""
                    **Average successful settings:**
                    - Energy: {avg_energy:.1f} eV
                    - Barrier Height: {avg_height:.1f} eV
                    - Barrier Width: {avg_width:.1f} nm
                    """)
                
                with col2:
                    # Use argmax with iloc for numpy array compatibility
                    best_position = np.argmax(tunneling_experiments['Transmission'])
                    best_exp = tunneling_experiments.iloc[best_position]
                    st.markdown(f"""
                    **Best tunneling experiment (#{best_exp['ID']}):**
                    - Energy: {best_exp['Energy (eV)']} eV
                    - Barrier Height: {best_exp['Barrier Height (eV)']} eV
                    - Barrier Width: {best_exp['Barrier Width (nm)']} nm
                    - Transmission: {best_exp['Transmission']:.4f}
                    """)
            
            # Management buttons
            st.subheader("Experiment Management")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📥 Download Experiments", help="Download your experiment data as CSV"):
                    csv = df_experiments.to_csv(index=False)
                    st.download_button(
                        label="💾 Download CSV",
                        data=csv,
                        file_name=f"quantum_tunneling_experiments_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            
            with col2:
                if st.button("🗑️ Clear All Experiments", help="Delete all saved experiments"):
                    st.session_state.experiment_history = []
                    st.session_state.experiment_counter = 1
                    st.success("All experiments cleared!")
                    st.rerun()
            
            with col3:
                if st.button("🔄 Replay Best", help="Load settings from best transmission experiment"):
                    if len(df_experiments) > 0:
                        # Use argmax with iloc for numpy array compatibility
                        best_position = np.argmax(df_experiments['Transmission'])
                        best_exp = df_experiments.iloc[best_position]
                        st.info(f"Best experiment was #{best_exp['ID']} with {best_exp['Transmission']:.4f} transmission. Adjust sliders to match these values:")
                        st.json({
                            "Energy": f"{best_exp['Energy (eV)']} eV",
                            "Barrier Height": f"{best_exp['Barrier Height (eV)']} eV",
                            "Barrier Width": f"{best_exp['Barrier Width (nm)']} nm",
                            "Mass Factor": best_exp['Mass Factor']
                        })
    
    # Educational information
    st.markdown("---")
    st.subheader("Understanding Quantum Tunneling")
    
    if E < V0 and T > 1e-10:
        st.success("""
        **Quantum Tunneling in Action!**
        
        The particle energy ({:.2f} eV) is less than the barrier height ({:.2f} eV). 
        Classically, this particle should not be able to pass through the barrier. 
        However, quantum mechanics allows for a probability of tunneling through!
        
        Key observations:
        - Transmission probability: {:.1%}
        - The wave function decays exponentially inside the barrier
        - Some probability exists on the other side of the barrier
        """.format(energy_eV, barrier_height_eV, T))
    elif E < V0 and T <= 1e-10:
        st.warning("""
        **Quantum Tunneling Blocked**
        
        The particle energy ({:.2f} eV) is less than the barrier height ({:.2f} eV), 
        but the barrier is too thick or the energy too low for effective tunneling.
        
        Key observations:
        - Transmission probability: {:.2e} (essentially zero)
        - Try reducing barrier width or increasing particle energy
        - The wave function decays too rapidly inside the barrier
        """.format(energy_eV, barrier_height_eV, T))
    else:
        st.info("""
        **Classical Regime** ⚡
        
        The particle energy ({:.2f} eV) is greater than the barrier height ({:.2f} eV). 
        The particle has enough energy to classically go over the barrier, but there's 
        still some reflection due to the potential step.
        
        Key observations:
        - Transmission probability: {:.1%}
        - The wave function oscillates inside the barrier
        - Reflection still occurs due to impedance mismatch
        """.format(energy_eV, barrier_height_eV, T))
    
    # Technical details
    with st.expander("Technical Details"):
        st.markdown(f"""
        **Physical Constants:**
        - ℏ (reduced Planck's constant): {hbar:.3e} J·s
        - Particle mass: {m/9.10938356e-31:.1f} × electron mass
        - Barrier width: {barrier_width_nm:.1f} nm
        
        **Wave Numbers:**
        - k₁ (before barrier): {np.sqrt(2 * m * E) / hbar:.3e} m⁻¹
        - k₂ (inside barrier): {"complex" if E < V0 else "real"}
        - k₃ (after barrier): {np.sqrt(2 * m * E) / hbar:.3e} m⁻¹
        
        **Quantum Mechanical Principle:**
        The wave function ψ(x) describes the quantum state of the particle. 
        |ψ(x)|² gives the probability density of finding the particle at position x.
        Even when E < V₀, there's a non-zero probability of transmission due to 
        the wave-like nature of matter at quantum scales.
        """)

if __name__ == "__main__":
    main()
