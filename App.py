"""
================================================================================
TUGAS 10 DESEMBER 2025 - GENETIC ALGORITHM
Implementasi Lengkap dengan Streamlit Interface
================================================================================

Fitur:
1. Data Static (Excel) - Pre-loaded dataset
2. Data Random - Generate random population
3. Interactive Mode - User input parameters
4. Visualisasi Real-time
5. Export hasil ke CSV/JSON

Author: Wildan
Mata Kuliah: Algoritma Genetika
================================================================================
"""

import streamlit as st
import random
import math
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import json

# Konfigurasi harus di paling atas, sebelum widget apapun
st.set_page_config(
    page_title="Genetic Algorithm - Tugas 10 Des 2025",
    page_icon="🧬",
    layout="wide"
)

# ================================================================================
# GENETIC ALGORITHM CLASS
# ================================================================================

class GeneticAlgorithm:
    """Genetic Algorithm Engine"""
    
    def __init__(self, config):
        self.ukuran_populasi = config['ukuran_populasi']
        self.jumlah_generasi = config['jumlah_generasi']
        self.probabilitas_crossover = config['probabilitas_crossover']
        self.probabilitas_mutasi = config['probabilitas_mutasi']
        self.panjang_kromosom = config['panjang_kromosom']
        self.range_min = config['range_min']
        self.range_max = config['range_max']
        self.fungsi_choice = config['fungsi_choice']
        self.tipe_optimasi = config['tipe_optimasi']
        
        self.populasi = []
        self.history = []
        self.best_ever = None
        self.best_ever_fitness = None
        
    def fitness_function(self, x):
        """Hitung fitness berdasarkan pilihan fungsi"""
        if self.fungsi_choice == 1:
            return x ** 2
        elif self.fungsi_choice == 2:
            return -x**2 + 10*x
        elif self.fungsi_choice == 3:
            return math.sin(x) * x
        elif self.fungsi_choice == 4:
            return x**3 - 5*x**2 + 2*x + 10
        else:
            return math.exp(-x**2/10) * math.cos(x)
    
    def binary_to_decimal(self, binary_string):
        """Konversi biner ke desimal"""
        return int(binary_string, 2)
    
    def decimal_to_x(self, decimal):
        """Konversi desimal ke nilai x dalam range"""
        max_decimal = 2 ** self.panjang_kromosom - 1
        return self.range_min + (decimal / max_decimal) * (self.range_max - self.range_min)
    
    def initialize_population(self, initial_pop=None):
        """Inisialisasi populasi"""
        if initial_pop:
            self.populasi = initial_pop
        else:
            self.populasi = []
            for _ in range(self.ukuran_populasi):
                kromosom = ''.join([str(random.randint(0, 1)) for _ in range(self.panjang_kromosom)])
                self.populasi.append(kromosom)
    
    def evaluate_population(self, populasi):
        """Evaluasi fitness populasi"""
        fitness_values = []
        for kromosom in populasi:
            decimal = self.binary_to_decimal(kromosom)
            x = self.decimal_to_x(decimal)
            fitness = self.fitness_function(x)
            fitness_values.append({
                'kromosom': kromosom,
                'decimal': decimal,
                'x': x,
                'fitness': fitness
            })
        return fitness_values
    
    def selection(self, fitness_values):
        """Seleksi dengan Roulette Wheel"""
        fitnesses = [item['fitness'] for item in fitness_values]
        
        if self.tipe_optimasi == 'minimasi':
            min_fit = min(fitnesses)
            adjusted_fitness = [min_fit - f + 1 for f in fitnesses]
        else:
            adjusted_fitness = fitnesses
        
        total_fitness = sum(adjusted_fitness)
        
        selected = []
        for _ in range(self.ukuran_populasi):
            r = random.random() * total_fitness
            cumulative = 0
            for i, fit in enumerate(adjusted_fitness):
                cumulative += fit
                if r <= cumulative:
                    selected.append(fitness_values[i]['kromosom'])
                    break
        
        return selected
    
    def crossover(self, parent1, parent2):
        """Single-point crossover"""
        if random.random() < self.probabilitas_crossover:
            point = random.randint(1, self.panjang_kromosom - 1)
            child1 = parent1[:point] + parent2[point:]
            child2 = parent2[:point] + parent1[point:]
            return child1, child2
        return parent1, parent2
    
    def mutation(self, kromosom):
        """Bit-flip mutation"""
        kromosom_list = list(kromosom)
        for i in range(len(kromosom_list)):
            if random.random() < self.probabilitas_mutasi:
                kromosom_list[i] = '1' if kromosom_list[i] == '0' else '0'
        return ''.join(kromosom_list)
    
    def run(self, progress_callback=None):
        """Jalankan algoritma"""
        if self.tipe_optimasi == 'maksimasi':
            self.best_ever_fitness = -float('inf')
        else:
            self.best_ever_fitness = float('inf')
        
        for gen in range(self.jumlah_generasi):
            # Evaluasi
            fitness_values = self.evaluate_population(self.populasi)
            
            # Cari best
            fitnesses = [item['fitness'] for item in fitness_values]
            if self.tipe_optimasi == 'maksimasi':
                best_fitness = max(fitnesses)
                best_idx = fitnesses.index(best_fitness)
            else:
                best_fitness = min(fitnesses)
                best_idx = fitnesses.index(best_fitness)
            
            best_individual = fitness_values[best_idx]
            avg_fitness = sum(fitnesses) / len(fitnesses)
            
            # Simpan history
            self.history.append({
                'generasi': gen,
                'best_fitness': best_fitness,
                'average_fitness': avg_fitness,
                'best_x': best_individual['x'],
                'best_kromosom': best_individual['kromosom']
            })
            
            # Update best ever
            if self.tipe_optimasi == 'maksimasi':
                if best_fitness > self.best_ever_fitness:
                    self.best_ever_fitness = best_fitness
                    self.best_ever = best_individual
            else:
                if best_fitness < self.best_ever_fitness:
                    self.best_ever_fitness = best_fitness
                    self.best_ever = best_individual
            
            # Progress callback
            if progress_callback:
                progress_callback(gen + 1, self.jumlah_generasi)
            
            # Jika generasi terakhir
            if gen == self.jumlah_generasi - 1:
                break
            
            # Seleksi
            selected = self.selection(fitness_values)
            
            # Crossover
            new_population = []
            for i in range(0, len(selected), 2):
                if i + 1 < len(selected):
                    child1, child2 = self.crossover(selected[i], selected[i+1])
                    new_population.extend([child1, child2])
                else:
                    new_population.append(selected[i])
            
            # Mutasi
            self.populasi = [self.mutation(chrom) for chrom in new_population[:self.ukuran_populasi]]
        
        return self.best_ever, self.history


# ================================================================================
# STREAMLIT APP
# ================================================================================

def main():
    # Header
    st.title("🧬 Genetic Algorithm Interactive")
    st.markdown("**Tugas 10 Desember 2025** - Implementasi Algoritma Genetika")
    st.markdown("---")
    
    # Sidebar - Mode Selection
    st.sidebar.title("⚙️ Pengaturan")
    mode = st.sidebar.radio(
        "Pilih Mode:",
        ["📊 Data Static (Excel)", "🎲 Data Random", "🎮 Interactive Mode"]
    )
    
    st.sidebar.markdown("---")
    
    # ============================================================================
    # MODE 1: DATA STATIC (EXCEL)
    # ============================================================================
    if mode == "📊 Data Static (Excel)":
        st.header("📊 Mode: Data Static (Excel)")
        st.info("Mode ini menggunakan data pre-loaded untuk mendemonstrasikan GA dengan dataset tetap.")
        
        # Create sample static data
        static_data = {
            'No': list(range(1, 11)),
            'Kromosom': [
                '1010101010', '0101010101', '1100110011', '0011001100', '1111000011',
                '0000111111', '1010011001', '0101100110', '1100001111', '0011110000'
            ],
            'x_value': [0.0] * 10,
            'fitness': [0.0] * 10
        }
        
        df_static = pd.DataFrame(static_data)
        
        # Sidebar parameters untuk static data
        st.sidebar.subheader("Parameter Algoritma")
        fungsi_static = st.sidebar.selectbox(
            "Fungsi Fitness:",
            [1, 2, 3, 4, 5],
            format_func=lambda x: {
                1: "f(x) = x²",
                2: "f(x) = -x² + 10x",
                3: "f(x) = sin(x) × x",
                4: "f(x) = x³ - 5x² + 2x + 10",
                5: "f(x) = e^(-x²/10) × cos(x)"
            }[x]
        )
        
        jumlah_gen_static = st.sidebar.slider("Jumlah Generasi:", 5, 50, 20)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📋 Data Input (Static)")
            st.dataframe(df_static, use_container_width=True)
        
        with col2:
            st.subheader("🎯 Pengaturan")
            st.write(f"**Fungsi:** {['x²', '-x² + 10x', 'sin(x) × x', 'x³ - 5x² + 2x + 10', 'e^(-x²/10) × cos(x)'][fungsi_static-1]}")
            st.write(f"**Jumlah Generasi:** {jumlah_gen_static}")
            st.write(f"**Ukuran Populasi:** {len(df_static)}")
        
        if st.button("▶️ Jalankan Algoritma", type="primary", use_container_width=True):
            with st.spinner("Menjalankan Genetic Algorithm..."):
                # Setup config
                config = {
                    'ukuran_populasi': len(df_static),
                    'jumlah_generasi': jumlah_gen_static,
                    'probabilitas_crossover': 0.8,
                    'probabilitas_mutasi': 0.1,
                    'panjang_kromosom': 10,
                    'range_min': -10,
                    'range_max': 10,
                    'fungsi_choice': fungsi_static,
                    'tipe_optimasi': 'maksimasi'
                }
                
                # Run GA
                ga = GeneticAlgorithm(config)
                ga.initialize_population(df_static['Kromosom'].tolist())
                
                progress_bar = st.progress(0)
                def update_progress(current, total):
                    progress_bar.progress(current / total)
                
                best_solution, history = ga.run(update_progress)
                
                # Display results
                st.success("✅ Algoritma selesai!")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Best Fitness", f"{best_solution['fitness']:.6f}")
                with col2:
                    st.metric("Best x", f"{best_solution['x']:.6f}")
                with col3:
                    st.metric("Kromosom", best_solution['kromosom'])
                
                # Plot evolution
                df_history = pd.DataFrame(history)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_history['generasi'], y=df_history['best_fitness'],
                                        mode='lines+markers', name='Best Fitness',
                                        line=dict(color='green', width=2)))
                fig.add_trace(go.Scatter(x=df_history['generasi'], y=df_history['average_fitness'],
                                        mode='lines+markers', name='Average Fitness',
                                        line=dict(color='blue', width=2, dash='dash')))
                fig.update_layout(title="Evolusi Fitness", xaxis_title="Generasi", yaxis_title="Fitness")
                st.plotly_chart(fig, use_container_width=True)
                
                # Evolution table
                st.subheader("📊 Tabel Evolusi")
                st.dataframe(df_history, use_container_width=True)
    
    # ============================================================================
    # MODE 2: DATA RANDOM
    # ============================================================================
    elif mode == "🎲 Data Random":
        st.header("🎲 Mode: Data Random")
        st.info("Mode ini generate populasi awal secara random untuk setiap run.")
        
        # Sidebar parameters
        st.sidebar.subheader("Parameter Algoritma")
        
        ukuran_pop_random = st.sidebar.slider("Ukuran Populasi:", 5, 50, 10)
        jumlah_gen_random = st.sidebar.slider("Jumlah Generasi:", 5, 100, 20)
        
        fungsi_random = st.sidebar.selectbox(
            "Fungsi Fitness:",
            [1, 2, 3, 4, 5],
            format_func=lambda x: {
                1: "f(x) = x²",
                2: "f(x) = -x² + 10x",
                3: "f(x) = sin(x) × x",
                4: "f(x) = x³ - 5x² + 2x + 10",
                5: "f(x) = e^(-x²/10) × cos(x)"
            }[x],
            key="fungsi_random"
        )
        
        optimasi_random = st.sidebar.radio("Tipe Optimasi:", ["maksimasi", "minimasi"])
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("🎯 Konfigurasi")
            st.write(f"**Fungsi:** {['x²', '-x² + 10x', 'sin(x) × x', 'x³ - 5x² + 2x + 10', 'e^(-x²/10) × cos(x)'][fungsi_random-1]}")
            st.write(f"**Ukuran Populasi:** {ukuran_pop_random}")
            st.write(f"**Jumlah Generasi:** {jumlah_gen_random}")
            st.write(f"**Optimasi:** {optimasi_random.upper()}")
        
        with col2:
            st.subheader("ℹ️ Informasi")
            st.write("- Populasi akan di-generate random")
            st.write("- Setiap run menghasilkan hasil berbeda")
            st.write("- Cocok untuk eksperimen dan testing")
        
        if st.button("🎲 Generate & Jalankan", type="primary", use_container_width=True):
            with st.spinner("Generating random population dan menjalankan GA..."):
                config = {
                    'ukuran_populasi': ukuran_pop_random,
                    'jumlah_generasi': jumlah_gen_random,
                    'probabilitas_crossover': 0.8,
                    'probabilitas_mutasi': 0.1,
                    'panjang_kromosom': 10,
                    'range_min': -10,
                    'range_max': 10,
                    'fungsi_choice': fungsi_random,
                    'tipe_optimasi': optimasi_random
                }
                
                ga = GeneticAlgorithm(config)
                ga.initialize_population()
                
                # Show initial population
                initial_pop = ga.evaluate_population(ga.populasi)
                df_initial = pd.DataFrame(initial_pop)
                
                with st.expander("👀 Lihat Populasi Awal (Random)"):
                    st.dataframe(df_initial, use_container_width=True)
                
                progress_bar = st.progress(0)
                def update_progress(current, total):
                    progress_bar.progress(current / total)
                
                best_solution, history = ga.run(update_progress)
                
                st.success("✅ Algoritma selesai!")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Best Fitness", f"{best_solution['fitness']:.6f}")
                with col2:
                    st.metric("Best x", f"{best_solution['x']:.6f}")
                with col3:
                    st.metric("Kromosom", best_solution['kromosom'])
                
                # Visualizations
                df_history = pd.DataFrame(history)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig1 = go.Figure()
                    fig1.add_trace(go.Scatter(x=df_history['generasi'], y=df_history['best_fitness'],
                                            mode='lines+markers', name='Best Fitness',
                                            line=dict(color='green', width=3)))
                    fig1.add_trace(go.Scatter(x=df_history['generasi'], y=df_history['average_fitness'],
                                            mode='lines+markers', name='Average Fitness',
                                            line=dict(color='blue', width=2, dash='dash')))
                    fig1.update_layout(title="Evolusi Fitness", xaxis_title="Generasi", yaxis_title="Fitness")
                    st.plotly_chart(fig1, use_container_width=True)
                
                with col2:
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(x=df_history['generasi'], y=df_history['best_x'],
                                            mode='lines+markers', 
                                            line=dict(color='purple', width=3)))
                    fig2.update_layout(title="Evolusi Best X", xaxis_title="Generasi", yaxis_title="x value")
                    st.plotly_chart(fig2, use_container_width=True)
                
                st.subheader("📊 Tabel Evolusi Lengkap")
                st.dataframe(df_history, use_container_width=True)
    
    # ============================================================================
    # MODE 3: INTERACTIVE MODE
    # ============================================================================
    else:  # Interactive Mode
        st.header("🎮 Mode: Interactive (Full Customization)")
        st.info("Mode ini memberikan kontrol penuh atas semua parameter GA.")
        
        # Sidebar parameters - FULL CONTROL
        st.sidebar.subheader("🎯 Parameter Populasi")
        ukuran_pop = st.sidebar.slider("Ukuran Populasi:", 5, 100, 20)
        jumlah_gen = st.sidebar.slider("Jumlah Generasi:", 5, 200, 30)
        panjang_krom = st.sidebar.slider("Panjang Kromosom (bit):", 8, 16, 10)
        
        st.sidebar.subheader("🧬 Parameter Genetika")
        prob_crossover = st.sidebar.slider("Probabilitas Crossover:", 0.0, 1.0, 0.8, 0.05)
        prob_mutasi = st.sidebar.slider("Probabilitas Mutasi:", 0.0, 0.5, 0.1, 0.01)
        
        st.sidebar.subheader("📊 Fungsi & Range")
        fungsi_interactive = st.sidebar.selectbox(
            "Fungsi Fitness:",
            [1, 2, 3, 4, 5],
            format_func=lambda x: {
                1: "f(x) = x²",
                2: "f(x) = -x² + 10x",
                3: "f(x) = sin(x) × x",
                4: "f(x) = x³ - 5x² + 2x + 10",
                5: "f(x) = e^(-x²/10) × cos(x)"
            }[x],
            key="fungsi_interactive"
        )
        
        range_min = st.sidebar.number_input("Range Minimum:", value=-10.0)
        range_max = st.sidebar.number_input("Range Maximum:", value=10.0)
        
        optimasi_interactive = st.sidebar.radio("Tipe Optimasi:", ["maksimasi", "minimasi"], key="opt_interactive")
        
        # Main area - Summary
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("📋 Parameter Populasi")
            st.write(f"**Ukuran Populasi:** {ukuran_pop}")
            st.write(f"**Jumlah Generasi:** {jumlah_gen}")
            st.write(f"**Panjang Kromosom:** {panjang_krom} bit")
        
        with col2:
            st.subheader("🧬 Parameter Genetika")
            st.write(f"**Crossover (Pc):** {prob_crossover}")
            st.write(f"**Mutasi (Pm):** {prob_mutasi}")
            st.write(f"**Tipe:** {optimasi_interactive.upper()}")
        
        with col3:
            st.subheader("📊 Fungsi & Range")
            fungsi_names = ["x²", "-x² + 10x", "sin(x) × x", "x³ - 5x² + 2x + 10", "e^(-x²/10) × cos(x)"]
            st.write(f"**Fungsi:** {fungsi_names[fungsi_interactive-1]}")
            st.write(f"**Range x:** [{range_min}, {range_max}]")
        
        st.markdown("---")
        
        if st.button("🚀 JALANKAN ALGORITMA", type="primary", use_container_width=True):
            with st.spinner("Menjalankan Genetic Algorithm dengan parameter custom..."):
                config = {
                    'ukuran_populasi': ukuran_pop,
                    'jumlah_generasi': jumlah_gen,
                    'probabilitas_crossover': prob_crossover,
                    'probabilitas_mutasi': prob_mutasi,
                    'panjang_kromosom': panjang_krom,
                    'range_min': range_min,
                    'range_max': range_max,
                    'fungsi_choice': fungsi_interactive,
                    'tipe_optimasi': optimasi_interactive
                }
                
                ga = GeneticAlgorithm(config)
                ga.initialize_population()
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(current, total):
                    progress_bar.progress(current / total)
                    status_text.text(f"Generasi {current}/{total}...")
                
                best_solution, history = ga.run(update_progress)
                
                status_text.empty()
                st.success("🎉 Algoritma selesai dijalankan!")
                
                # Results
                st.subheader("🏆 Solusi Terbaik")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Best Fitness", f"{best_solution['fitness']:.6f}")
                with col2:
                    st.metric("Best x", f"{best_solution['x']:.6f}")
                with col3:
                    st.metric("Decimal Value", f"{best_solution['decimal']}")
                with col4:
                    st.metric("Binary", best_solution['kromosom'][:5] + "...")
                
                st.info(f"**Kromosom Lengkap:** `{best_solution['kromosom']}`")
                
                # Visualizations
                st.subheader("📈 Visualisasi Evolusi")
                
                df_history = pd.DataFrame(history)
                
                tab1, tab2, tab3 = st.tabs(["📊 Fitness Evolution", "📉 X Value Evolution", "📋 Data Table"])
                
                with tab1:
                    fig1 = go.Figure()
                    fig1.add_trace(go.Scatter(
                        x=df_history['generasi'], 
                        y=df_history['best_fitness'],
                        mode='lines+markers', 
                        name='Best Fitness',
                        line=dict(color='green', width=3),
                        marker=dict(size=8)
                    ))
                    fig1.add_trace(go.Scatter(
                        x=df_history['generasi'], 
                        y=df_history['average_fitness'],
                        mode='lines+markers', 
                        name='Average Fitness',
                        line=dict(color='blue', width=2, dash='dash'),
                        marker=dict(size=6)
                    ))
                    fig1.update_layout(
                        title="Evolusi Fitness per Generasi",
                        xaxis_title="Generasi",
                        yaxis_title="Fitness Value",
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig1, use_container_width=True)
                
                with tab2:
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(
                        x=df_history['generasi'], 
                        y=df_history['best_x'],
                        mode='lines+markers',
                        line=dict(color='purple', width=3),
                        marker=dict(size=8)
                    ))
                    fig2.update_layout(
                        title="Evolusi Best X Value per Generasi",
                        xaxis_title="Generasi",
                        yaxis_title="x Value"
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                
                with tab3:
                    st.dataframe(df_history, use_container_width=True)
                
                # Export options
                st.subheader("💾 Export Hasil")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Export CSV
                    csv = df_history.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv,
                        file_name=f"GA_Results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    # Export JSON
                    results = {
                        'parameter': config,
                        'best_solution': best_solution,
                        'history': history
                    }
                    json_str = json.dumps(results, indent=2)
                    st.download_button(
                        label="📥 Download JSON",
                        data=json_str,
                        file_name=f"GA_Results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>🧬 Genetic Algorithm Interactive - Tugas 10 Desember 2025</p>
        <p>Developed with ❤️ using Streamlit</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()