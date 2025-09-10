import uproot
import ROOT
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
import sys
from tensorflow.keras.models import load_model
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

# ===================================================================================
# 1. CONFIGURACIÓN Y CONFIGURACIÓN BÁSICA
# ===================================================================================
# 1.1. Importaciones
# ------------------------------------------------------------------------------------
# Importaciones necesarias para el análisis y visualización
def clear_output(wait=False):
    """Limpia la salida de la consola de forma multiplataforma."""
    # Para Windows
    if os.name == 'nt':
        os.system('cls')
    # Para Unix/Linux/MacOS
    else:
        os.system('clear')
    
    # Si estamos en un notebook de Jupyter, también intentamos usar IPython
    try:
        from IPython.display import clear_output as ipy_clear
        ipy_clear(wait=wait)
    except ImportError:
        pass

# Importaciones adicionales
import joblib
import os
import time
import argparse

# OPCIONAL: Si tienes problemas con la GPU y quieres forzar el uso de CPU
# Descomenta las siguientes dos líneas ANTES de importar tensorflow o keras
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# ===================================================================================
# 2. CONFIGURACIÓN DEL PROGRAMA
# ===================================================================================
# 2.1. Rutas y archivos
# ------------------------------------------------------------------------------------
MODEL_PATH = "particle_classifier_model.keras"  # Ruta al modelo entrenado
SCALER_PATH = "data_scaler.pkl"              # Ruta al scaler usado durante el entrenamiento
INPUT_ROOT_FILE = "clasificable.root"         # Archivo ROOT con los datos a predecir
TREE_NAME = "feature_tree"                    # Nombre del árbol en el archivo ROOT que contiene las características

# 2.2. Características del modelo
FEATURE_COLUMNS = [
    "Ec1",     # Energía del cluster principal
    "Ec2",     # Energía del segundo cluster más grande
    "Qmx",     # Carga máxima
    "QmxP",    # Carga máxima en z > 0
    "QmxM",    # Carga máxima en z < 0
    "tminP",   # Tiempo mínimo en z > 0
    "tminM",   # Tiempo mínimo en z < 0
    "dp",      # Diferencia de tiempo entre z > 0 y z < 0
    "Qmx2_val", # Valor de la carga máxima
    "Qtot_p",  # Carga total en z > 0
    "Qtot_m",  # Carga total en z < 0
    "nhits_fired_sipm"  # Número de SiPMs disparados
]

# ===================================================================================
# 3. CONFIGURACIÓN DE ANÁLISIS DE ENERGÍA
# ===================================================================================
# 3.1. Intervalos de energía en MeV
EK_INTERVALS = [
    (0, 1),           # 0-1 MeV
    (1, 3),           # 1-3 MeV
    (3, float('inf')) # >3 MeV
]

# 3.3. Funciones auxiliares de energía
def get_ek_interval(ek):
    """
    Determina el intervalo de energía al que pertenece un evento.
    
    Args:
        ek: Energía cinética del evento
        
    Returns:
        Índice del intervalo (0-4) o -1 si está fuera de rango
    """
    for i, (low, high) in enumerate(EK_INTERVALS):
        if low <= ek < high:
            return i
    return -1  # Fuera de rango


def get_ek_center(interval_idx):
    """
    Obtiene el valor central de un intervalo de energía.
    
    Args:
        interval_idx: Índice del intervalo (0-4)
        
    Returns:
        Valor central del intervalo o None si el índice es inválido
    """
    if 0 <= interval_idx < len(EK_INTERVALS):
        low, high = EK_INTERVALS[interval_idx]
        return (low + high) / 2
    return None


# ===================================================================================
# 4. MAPEO DE ETIQUETAS
# ===================================================================================
# 4.1. Mapeo de predicciones del modelo
# ------------------------------------------------------------------------------------
LABEL_TO_PARTICLE = {
    0: "Electrón",
    1: "Positrón",
    2: "Gamma"
}

# 4.2. Mapeo de PDG ID para etiquetas reales
PDG_TO_PARTICLE = {
    11: "Electrón",
    -11: "Positrón",
    22: "Gamma"
}

# ===================================================================================
# 5. VISUALIZACIÓN DE RESULTADOS
# ===================================================================================
# ------------------------------------------------------------------------------------
def display_histograms(event_idx, hist_zp, hist_zm, pred_label, true_label, confidence):
    """Muestra los histogramas de carga para un evento."""
    plt.figure(figsize=(14, 6))
    
    # Configurar título general
    title = f'Evento {event_idx} - Predicción: {pred_label} ({confidence*100:.1f}%)'
    if true_label:
        title += f' | Real: {true_label}'
    plt.suptitle(title, fontsize=14, fontweight='bold')
    
    # Mostrar histograma z > 0
    if hist_zp is not None:
        plt.subplot(1, 2, 1)
        plt.imshow(hist_zp.values().T, 
                  extent=[-900, 900, -900, 900], 
                  origin='lower', 
                  norm=LogNorm(vmin=1, vmax=hist_zp.values().max()),
                  cmap='viridis')
        plt.colorbar(label='Carga')
        plt.title('Plano z > 0')
        plt.xlabel('X (mm)')
        plt.ylabel('Y (mm)')
    
    # Mostrar histograma z < 0
    if hist_zm is not None:
        plt.subplot(1, 2, 2)
        plt.imshow(hist_zm.values().T,
                  extent=[-900, 900, -900, 900],
                  origin='lower',
                  norm=LogNorm(vmin=1, vmax=hist_zm.values().max()),
                  cmap='viridis')
        plt.colorbar(label='Carga')
        plt.title('Plano z < 0')
        plt.xlabel('X (mm)')
        plt.ylabel('Y (mm)')
    
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)  # Dar tiempo para que se muestre la figura

# ===================================================================================
# 6. FUNCIÓN PRINCIPAL DE CLASIFICACIÓN
# ===================================================================================

def classify_events(model_path, scaler_path, input_file, tree_name, feature_columns, 
                   max_events=None, show_histograms_flag=True, prob_threshold=0.0):
    """
    Clasifica eventos de partículas usando un modelo de red neuronal entrenado.
    
    Realiza la carga del modelo, preprocesamiento de datos, predicción y análisis
    de resultados, incluyendo métricas de rendimiento y visualizaciones.
    
    Args:
        model_path (str): Ruta al archivo del modelo Keras guardado (.keras o .h5)
        scaler_path (str): Ruta al archivo del escalador guardado (.pkl)
        input_file (str): Ruta al archivo ROOT de entrada con los datos
        tree_name (str): Nombre del árbol en el archivo ROOT que contiene los datos
        feature_columns (list): Lista de nombres de características a utilizar
        max_events (int, opcional): Número máximo de eventos a procesar
        show_histograms_flag (bool): Si es True, muestra histogramas de eventos
        prob_threshold (float): Umbral de probabilidad [0-1] para filtrar predicciones
        
    Returns:
        None: Los resultados se muestran por consola y en gráficos
    """
    
    # ===========================================================================
    # 6.1. INICIALIZACIÓN Y CARGA DE RECURSOS
    # ===========================================================================
    file = None  # Variable para el archivo ROOT
    
    try:
        # -------------------------------------------------------------------
        # 6.1.1. Cargar el modelo de red neuronal
        # -------------------------------------------------------------------
        try:
            model = load_model(model_path)
            print(f"✅ Modelo cargado exitosamente: {os.path.basename(model_path)}")
        except Exception as e:
            print(f"❌ Error al cargar el modelo desde '{model_path}': {e}")
            return

        # -------------------------------------------------------------------
        # 6.1.2. Cargar el escalador de características
        # -------------------------------------------------------------------
        try:
            scaler = joblib.load(scaler_path)
            print(f"✅ Scaler cargado exitosamente: {os.path.basename(scaler_path)}")
        except Exception as e:
            print(f"❌ Error al cargar el scaler desde '{scaler_path}': {e}")
            return

        # ===================================================================
        # 6.2. CARGA Y PREPARACIÓN DE DATOS
        # ===================================================================
        try:
            # -------------------------------------------------------------------
            # 6.2.1. Cargar datos del archivo ROOT
            # -------------------------------------------------------------------
            print(f"\n📂 Cargando datos desde: {input_file}")
            file = uproot.open(input_file)
            
            # Verificar si el árbol existe
            if tree_name not in file:
                print(f"❌ Error: No se encontró el árbol '{tree_name}' en el archivo")
                return
                
            tree = file[tree_name]
            
            # -------------------------------------------------------------------
            # 6.2.2. Verificar y cargar histogramas si es necesario
            # -------------------------------------------------------------------
            show_histograms = False
            if show_histograms_flag:
                histo_dir = file.get("histogramas")
                if histo_dir is not None:
                    show_histograms = True
                    print("✅ Histogramas disponibles para visualización")
                else:
                    print("⚠️  Advertencia: No se encontró el directorio 'histogramas'")
            
            # -------------------------------------------------------------------
            # 6.2.3. Cargar características y etiquetas
            # -------------------------------------------------------------------
            print(f"📊 Cargando características: {', '.join(feature_columns)}")
            
            # Cargar características principales del feature_tree
            # Cargar datos en un DataFrame de pandas
            print(f"\n📥 Cargando datos desde {input_file}...")
            df = tree.arrays(feature_columns + ["mcpdg", "event_number"], library="pd")
            print(f"   - Total eventos cargados: {len(df)} (event_number: {df['event_number'].min()} a {df['event_number'].max()})")
            
            # Cargar mcke del árbol output si existe
            if 'output' in file:
                output_tree = file['output']
                try:
                    mcke_data = output_tree.arrays(["mcke"], library="pd")['mcke']
                    # Asegurarse de que las longitudes coincidan
                    if len(mcke_data) >= len(df):
                        df['mcke'] = mcke_data.values[:len(df)]
                        print("✅ mcke cargado correctamente del árbol 'output'")
                    else:
                        print("⚠️  Advertencia: El árbol 'output' tiene menos eventos que 'feature_tree'")
                        df = df.iloc[:len(mcke_data)]
                        print(f"   - Eventos después de igualar con mcke_data: {len(df)}")
                        df['mcke'] = mcke_data.values
                except Exception as e:
                    print(f"⚠️  No se pudo cargar 'mcke' del árbol 'output': {e}")
                    df['mcke'] = 0.0  # Valor por defecto
            else:
                print("⚠️  No se encontró el árbol 'output' para cargar 'mcke'")
                df['mcke'] = 0.0  # Valor por defecto

            # -------------------------------------------------------------------
            # 6.2.4. Limitar el número de eventos manualmente si es necesario
            # -------------------------------------------------------------------
            if max_events is not None and max_events > 0 and len(df) > max_events:
                df = df.head(max_events)

                print(f"🔢 Se procesarán {max_events} eventos (de {len(df)} disponibles)")
            else:
                print(f"🔢 Se procesarán todos los eventos: {len(df)}")
                
        except Exception as e:
            print(f"❌ Error al cargar el archivo ROOT: {e}")
            return
        
        # ===================================================================
        # 6.3. PREPROCESAMIENTO DE DATOS
        # ===================================================================
        print("\n🔧 Preprocesando datos...")
        
        # -------------------------------------------------------------------
        # 6.3.1. Extraer y escalar características
        # -------------------------------------------------------------------
        X = df[feature_columns].values
        X_scaled = scaler.transform(X)
        
        # ===================================================================
        # 6.4. PREDICCIÓN
        # ===================================================================
        print("\n🧠 Realizando predicciones...")
        
        # -------------------------------------------------------------------
        # 6.4.1. Realizar predicciones y normalizar probabilidades
        # -------------------------------------------------------------------
        probabilities = model.predict(X_scaled, verbose=0)
        probabilities = probabilities / np.sum(probabilities, axis=1, keepdims=True)
        
        # -------------------------------------------------------------------
        # 6.4.2. Procesar resultados de predicción
        # -------------------------------------------------------------------
        predicted_labels = np.argmax(probabilities, axis=1)
        confidence_scores = np.max(probabilities, axis=1)
        
        # ===================================================================
        # 6.5. ANÁLISIS DE RESULTADOS
        # ===================================================================

        # Crear carpeta Plots para guardar las figuras (si no existe)
        plots_dir = 'Plots'
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
            print(f"\nCreada carpeta '{plots_dir}' para guardar las figuras")
                
        if 'mcpdg' in df.columns:
            print("\n📊 Analizando resultados...")
            
            # -------------------------------------------------------------------
            # 6.5.1. Mapear códigos PDG a etiquetas numéricas
            # -------------------------------------------------------------------
            def map_pdg_to_label(pdg_code):
                """
                Convierte códigos PDG a etiquetas numéricas.
                
                Args:
                    pdg_code: Código PDG de la partícula
                    
                Returns:
                    int: 0 (e-), 1 (e+), 2 (gamma) o -1 (desconocido)
                """
                if pdg_code == 11: return 0    # Electrón
                if pdg_code == -11: return 1   # Positrón
                if pdg_code == 22: return 2    # Gamma
                return -1  # PDG no reconocido
            
            # -------------------------------------------------------------------
            # 6.5.2. Filtrar eventos con PDG reconocido
            # -------------------------------------------------------------------
            # Mapear códigos PDG y filtrar solo los reconocidos
            true_labels = df['mcpdg'].apply(map_pdg_to_label)
            valid_pdg_mask = true_labels != -1
            
            # Obtener los event_numbers originales
            event_numbers = df['event_number'].values
            
            # Aplicar máscara de PDG válido a todos los arrays
            true_labels = true_labels[valid_pdg_mask].copy()
            predicted_labels = predicted_labels[valid_pdg_mask]
            confidence_scores = confidence_scores[valid_pdg_mask]
            probabilities = probabilities[valid_pdg_mask]
            mcke_values = df['mcke'].values[valid_pdg_mask].copy()
            filtered_event_numbers = event_numbers[valid_pdg_mask].copy()
            
            # Inicializar versiones filtradas (sin umbral aplicado)
            filtered_true_labels = true_labels.copy()
            filtered_predicted_labels = predicted_labels.copy()
            filtered_confidence_scores = confidence_scores.copy()
            filtered_probabilities = probabilities.copy()
            filtered_mcke_values = mcke_values.copy()
            
            # -------------------------------------------------------------------
            # 6.5.3. Aplicar umbral de probabilidad si es necesario
            # -------------------------------------------------------------------
            if prob_threshold > 0.0:
                print(f"\n[DEBUG] Antes del filtrado - Eventos totales: {len(true_labels)}")
                print(f"\n[DEBUG] Aplicando umbral de {prob_threshold*100}%...")
                # Crear máscara para eventos que superan el umbral
                above_threshold = confidence_scores >= prob_threshold
                
                if np.any(above_threshold):
                    # Actualizar versiones filtradas
                    filtered_true_labels = true_labels[above_threshold].copy()
                    filtered_predicted_labels = predicted_labels[above_threshold].copy()
                    filtered_confidence_scores = confidence_scores[above_threshold].copy()
                    filtered_probabilities = probabilities[above_threshold].copy()
                    filtered_mcke_values = mcke_values[above_threshold].copy()
                    filtered_event_numbers = filtered_event_numbers[above_threshold].copy()
                    
                    # Estadísticas
                    n_filtered = len(filtered_true_labels)
                    n_total = len(true_labels)
                    print(f"[DEBUG] Eventos que superan el umbral: {n_filtered}/{n_total} ({n_filtered/n_total*100:.1f}%)")
                    print(f"[DEBUG] Eventos descartados: {n_total - n_filtered}/{n_total} ({(n_total - n_filtered)/n_total*100:.1f}%)")
                else:
                    print("\n¡Advertencia! Ningún evento superó el umbral de confianza.")
                    return
            
            # -------------------------------------------------------------------
            # 6.5.4. Mostrar estadísticas de confianza
            # -------------------------------------------------------------------
            if len(filtered_confidence_scores) > 0:
                print("\nEstadísticas de confianza (solo eventos clasificados):")
                print(f"  - Mínima: {np.min(filtered_confidence_scores):.4f}")
                print(f"  - Máxima: {np.max(filtered_confidence_scores):.4f}")
                print(f"  - Media: {np.mean(filtered_confidence_scores):.4f}")
            else:
                print("\nNo hay eventos que cumplan con el umbral de confianza.")
                return
            
            # -------------------------------------------------------------------
            # 6.5.5. Validar conjunto de datos
            # -------------------------------------------------------------------
            if len(filtered_true_labels) == 0:
                print("\nNo quedan eventos válidos después de aplicar el umbral de probabilidad.")
                return
            
            # Verificar el número de clases únicas
            unique_classes = set(filtered_true_labels)
            if len(unique_classes) < 2:
                # Si solo hay una clase, mostrar estadísticas básicas
                unique_label = list(unique_classes)[0]
                particle_name = LABEL_TO_PARTICLE.get(unique_label, 'Desconocida')
                total_events = len(filtered_true_labels)
                print(f"\n[DEBUG] total_events (filtered_true_labels) = {total_events}")
                correct_predictions = np.sum(filtered_true_labels == filtered_predicted_labels)
                accuracy = correct_predictions / total_events * 100
                
                print(f"\n¡Advertencia! El dataset contiene solo una clase de partícula: {particle_name}")
                print(f"Total de eventos: {total_events}")
                print(f"Predicciones correctas: {correct_predictions}")
                print(f"Precisión global: {accuracy:.2f}%")
                return
            
            # -------------------------------------------------------------------
            # 6.5.6. Calcular matriz de confusión sin filtrar
            # -------------------------------------------------------------------
            print("\n🔍 Generando matriz de confusión sin filtrar...")
            # Usar las etiquetas originales sin filtrar
            predicted_unfiltered = np.argmax(probabilities, axis=1)
            cm_unfiltered = confusion_matrix(true_labels, predicted_unfiltered)
            
            # Calcular precisión global sin filtrar
            total_correct_unfiltered = np.sum(cm_unfiltered.diagonal())
            total_samples_unfiltered = np.sum(cm_unfiltered)
            accuracy_unfiltered = total_correct_unfiltered / total_samples_unfiltered if total_samples_unfiltered > 0 else 0
            
            # Mostrar resumen de precisión global sin filtrar
            print(f"Precisión global (sin filtrar): {accuracy_unfiltered:.4f} ({accuracy_unfiltered*100:.2f}%)")
            
            # Generar visualización de la matriz de confusión sin filtrar
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm_unfiltered, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Electrón', 'Positrón', 'Gamma'],
                       yticklabels=['Electrón', 'Positrón', 'Gamma'])
            plt.title('Matriz de Confusión')
            plt.xlabel('Predicción')
            plt.ylabel('Valor Real')
            plt.savefig(f'{plots_dir}/matriz_confusion_sin_filtrar.png', dpi=300, bbox_inches='tight')
            plt.show()

            # -------------------------------------------------------------------
            # 6.5.7. Análisis de clasificación por partícula (sin filtrar)
            # -------------------------------------------------------------------
            print("\n" + "="*75)
            print("📊 ANÁLISIS DE CLASIFICACIÓN POR PARTÍCULA (SIN FILTRAR)".center(75))
            print("="*75)
            
            # Obtener predicciones sin filtrar
            predicted_unfiltered = np.argmax(probabilities, axis=1)
            
            # Análisis para cada tipo de partícula (sin filtrar)
            for i, (label, particle) in enumerate(LABEL_TO_PARTICLE.items()):
                total_particles = np.sum(true_labels == i)
                if total_particles == 0:
                    print(f"\n⚠️ No hay eventos para la partícula {particle}")
                    continue
                
                correct = cm_unfiltered[i, i]
                total_predicted = np.sum(predicted_unfiltered == i)
                accuracy = correct/total_particles*100 if total_particles > 0 else 0
                
                # Emoji para la partícula
                emoji = {
                    'electrón': '🔵',
                    'positrón': '🔴',
                    'gamma': '🟢'
                }.get(particle.lower(), '•')
                
                # Barra de progreso para la precisión
                bar_length = 20
                filled_length = int(bar_length * correct / total_particles) if total_particles > 0 else 0
                bar = '█' * filled_length + '░' * (bar_length - filled_length)
                
                print(f"\n{emoji} {particle.upper():<10} {'─' * (75 - len(particle) - 2)}")
                print(f"   • Total eventos: {total_particles:>6}")
                print(f"   • Correctos:     {correct:>6} ({accuracy:5.1f}%) {bar}")
                
                # Mostrar errores de clasificación si hay más de una clase
                total_errors = np.sum(cm_unfiltered[i]) - correct if len(cm_unfiltered) > i else 0
                if len(LABEL_TO_PARTICLE) > 1 and total_errors > 0:
                    print(f"   • Errores:       {total_errors:>6} ({(100-accuracy):5.1f}%)")
                    print("     └─" + "─" * 68)
                    for j, (_, other_particle) in enumerate(LABEL_TO_PARTICLE.items()):
                        if i != j and i < len(cm_unfiltered) and j < len(cm_unfiltered[i]) and cm_unfiltered[i, j] > 0:
                            count = cm_unfiltered[i, j]
                            error_percentage = count / total_errors * 100 if total_errors > 0 else 0
                            error_bar = '│' * (int(error_percentage/5) or 1)
                            print(f"       • Como {other_particle:<9}: {count:>6} ({error_percentage:5.1f}%) {error_bar}")
            
            print("\n" + "="*75 + "\n")
            
            
            # -------------------------------------------------------------------
            # 6.5.8. Calcular matriz de confusión filtrada
            # -------------------------------------------------------------------
            print("\n🔍 Generando matriz de confusión con filtrado...")
            cm = confusion_matrix(filtered_true_labels, filtered_predicted_labels)
            
            # Calcular precisión global filtrada
            total_correct = np.sum(cm.diagonal()) if len(cm) > 0 else 0
            total_samples = np.sum(cm) if len(cm) > 0 else 0
            accuracy = total_correct / total_samples if total_samples > 0 else 0
            total_accuracy = total_correct / len(df)
            
            # Mostrar resumen de precisión global filtrada
            print(f"Precisión (filtrada): {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"Precisión global (filtrada): {total_accuracy:.4f} ({total_accuracy*100:.2f}%)")
            if prob_threshold > 0.0:
                print(f"Umbral de probabilidad aplicado: {prob_threshold*100:.0f}%")
                print(f"Eventos que superan el umbral: {len(filtered_true_labels)}/{len(true_labels)} ({(len(filtered_true_labels)/len(true_labels))*100:.1f}%)")
            
            # Generar visualización de la matriz de confusión filtrada
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Electrón', 'Positrón', 'Gamma'],
                       yticklabels=['Electrón', 'Positrón', 'Gamma'])
            plt.title('Matriz de Confusión (filtrada)')
            plt.xlabel('Predicción')
            plt.ylabel('Valor Real')
            plt.savefig(f'{plots_dir}/matriz_confusion_filtrada.png', dpi=300, bbox_inches='tight')
            plt.show()
    
            # -------------------------------------------------------------------
            # 6.5.9. Análisis de clasificación por partícula (filtrado)
            # -------------------------------------------------------------------
            print("\n" + "="*75)
            print("📊 ANÁLISIS DE CLASIFICACIÓN POR PARTÍCULA (FILTRADO)".center(75))
            print("="*75)
            
            # Análisis para cada tipo de partícula (filtrado)
            for i, (label, particle) in enumerate(LABEL_TO_PARTICLE.items()):
                total_particles = np.sum(filtered_true_labels == i) if len(filtered_true_labels) > 0 else 0
                if total_particles == 0:
                    print(f"\n⚠️ No hay eventos para la partícula {particle}")
                    continue
                
                correct = cm[i, i] if i < len(cm) and i < len(cm[i]) else 0
                total_predicted = np.sum(filtered_predicted_labels == i) if len(filtered_predicted_labels) > 0 else 0
                accuracy = correct/total_particles*100 if total_particles > 0 else 0
                
                # Emoji para la partícula
                emoji = {
                    'electrón': '🔵',
                    'positrón': '🔴',
                    'gamma': '🟢'
                }.get(particle.lower(), '•')
                
                # Barra de progreso para la precisión
                bar_length = 20
                filled_length = int(bar_length * correct / total_particles) if total_particles > 0 else 0
                bar = '█' * filled_length + '░' * (bar_length - filled_length)
                
                print(f"\n{emoji} {particle.upper():<10} {'─' * (75 - len(particle) - 2)}")
                print(f"   • Total eventos: {total_particles:>6}")
                print(f"   • Correctos:     {correct:>6} ({accuracy:5.1f}%) {bar}")
                
                # Mostrar errores de clasificación si hay más de una clase
                total_errors = np.sum(cm[i]) - correct if i < len(cm) else 0
                if len(LABEL_TO_PARTICLE) > 1 and total_errors > 0:
                    print(f"   • Errores:       {total_errors:>6} ({(100-accuracy):5.1f}%)")
                    print("     └─" + "─" * 68)
                    for j, (_, other_particle) in enumerate(LABEL_TO_PARTICLE.items()):
                        if i != j and i < len(cm) and j < len(cm[i]) and cm[i, j] > 0:
                            count = cm[i, j]
                            error_percentage = count / total_errors * 100 if total_errors > 0 else 0
                            error_bar = '│' * (int(error_percentage/5) or 1)
                            print(f"       • Como {other_particle:<9}: {count:>6} ({error_percentage:5.1f}%) {error_bar}")
            
            print("\n" + "="*75 + "\n")

            """
            # -------------------------------------------------------------------
            # 6.5.10. Histogramas de probabilidades de clasificación (sin filtrar)
            # -------------------------------------------------------------------
            print("\nGenerando histogramas de probabilidades de clasificación (datos sin filtrar)...")
            
            # Crear figura para los histogramas
            plt.figure(figsize=(15, 5))
            
            # Para cada tipo de partícula (e-, e+, gamma)
            for i, (particle_name, color) in enumerate(zip(
                ['Electrones', 'Positrones', 'Gammas'],
                ['blue', 'red', 'green']
            )):
                plt.subplot(1, 3, i+1)
                mask = (true_labels == i)  # Usar all_true_labels sin filtrar
                
                if np.sum(mask) > 0:  # Si hay eventos de este tipo
                    # Graficar histograma de las probabilidades de la clase correcta
                    probs = probabilities[mask, i]  # Usar probabilities sin filtrar
                    plt.hist(probs, 
                            bins=20, 
                            alpha=0.7, 
                            color=color,
                            edgecolor='black')
                    plt.title(f'{particle_name} reales (n={np.sum(mask)})')
                    plt.xlabel(f'Prob. de ser {particle_name.lower().rstrip("s")}')
                    plt.ylabel('Número de eventos')
                    plt.xlim(0, 1)
                    plt.grid(True, alpha=0.3)
                else:
                    plt.text(0.5, 0.5, f'No hay {particle_name.lower()} reales', 
                            ha='center', va='center')
                    plt.xlim(0, 1)
                    plt.ylim(0, 1)
            
            plt.tight_layout()
            plt.suptitle(f'Distribución de probabilidades (sin filtrar, {len(true_labels)} eventos)', y=1.05)
            # Guardar la figura
            plt.savefig(f'{plots_dir}/histogramas_probabilidades_sin_filtrar.png', dpi=300, bbox_inches='tight')
            plt.show()  # Mostrar la figura de los histogramas
            """
            # -------------------------------------------------------------------
            # 6.5.10 Histogramas de probabilidades detallados por clase real y predicha (sin filtrar)
            # -------------------------------------------------------------------
            
            print("\nGenerando histogramas detallados de probabilidades por clase real y predicha (sin filtrar)...")
            
            # Nombres de las partículas para los títulos
            particle_names = ['Electrones', 'Positrones', 'Gammas']
            particle_short = ['e-', 'e+', 'γ']
            particle_root = ['electron', 'positron', 'gamma']  # Nombres para los histogramas ROOT
            colors = ['blue', 'red', 'green']
            
            # Lista para guardar los histogramas de ROOT
            root_histos = []
            
            # Crear una figura grande para todos los histogramas
            plt.figure(figsize=(20, 15))
            
            # Índice para el subplot
            plot_idx = 1
            
            # Para cada tipo de partícula REAL (filas)
            for true_idx, (true_name, true_short) in enumerate(zip(particle_names, particle_short)):
                # Para cada tipo de partícula PREDICHA (columnas)
                for pred_idx, (pred_name, pred_short, color) in enumerate(zip(particle_names, particle_short, colors)):
                    plt.subplot(3, 3, plot_idx)
                    
                    # Obtener máscara para la partícula real actual
                    mask = (true_labels == true_idx)
                    
                    if np.sum(mask) > 0:  # Si hay eventos de este tipo
                        # Obtener probabilidades para la clase predicha actual
                        probs = probabilities[mask, pred_idx]
                        
                        # Crear histograma de matplotlib
                        n, bins, _ = plt.hist(probs, 
                                        bins=20, 
                                        alpha=0.7, 
                                        color=color,
                                        edgecolor='black')  # Mostrar número de eventos absolutos
                                        
                        # Crear histograma de ROOT con estilo de solo barras
                        hist_name = f'h_prob_{particle_root[true_idx]}_as_{particle_root[pred_idx]}'
                        hist_title = f'{particle_root[true_idx]} as {particle_root[pred_idx]};Probability;Counts'
                        nbins = 20
                        xmin, xmax = 0.0, 1.0
                        
                        # Crear y configurar histograma
                        root_hist = ROOT.TH1F(hist_name, hist_title, nbins, xmin, xmax)
                        root_hist.SetStats(0)  # Desactivar caja de estadísticas
                        root_hist.SetLineColor(ROOT.kBlack)
                        root_hist.SetLineWidth(1)
                        root_hist.SetFillColor(ROOT.kBlue)  # Color de relleno
                        root_hist.SetFillStyle(1001)  # Relleno sólido
                        
                        # Llenar el histograma
                        for prob in probs:
                            root_hist.Fill(prob)
                        
                        # Añadir a la lista de histogramas
                        root_histos.append(root_hist)
                        
                        # Configurar título y etiquetas
                        plt.title(f'Reales: {true_name}\nPredichos como: {pred_name}')
                        plt.xlabel(f'Prob. de ser {pred_short}')
                        plt.ylabel('Número de eventos')
                        plt.xlim(0, 1)
                        plt.grid(True, alpha=0.3)
                        
                        # Añadir el número de eventos en la esquina superior derecha
                        plt.text(0.98, 0.95, f'N = {np.sum(mask):,}', 
                                transform=plt.gca().transAxes,
                                ha='right', va='top',
                                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
                    else:
                        plt.text(0.5, 0.5, f'No hay {true_name.lower()} reales', 
                                ha='center', va='center')
                        plt.xlim(0, 1)
                        plt.ylim(0, 1)
                    
                    # Incrementar el índice del subplot
                    plot_idx += 1
            
            # Ajustar el diseño y guardar la figura
            plt.tight_layout()
            plt.suptitle(f'Distribuciones de probabilidad por clase real y predicha\n(Total eventos: {len(true_labels):,})', y=1.02, fontsize=14)
            plt.savefig(f'{plots_dir}/histogramas_probabilidades_detallados.png', dpi=300, bbox_inches='tight')
            plt.show()
            print("Histogramas detallados guardados en:", f'{plots_dir}/histogramas_probabilidades_detallados.png')
            
            # Guardar los histogramas en el archivo ROOT
            try:
                # Obtener el nombre del archivo ROOT de entrada
                if hasattr(file, '_file') and hasattr(file._file, 'fFilePath'):
                    root_file_path = file._file.fFilePath
                else:
                    root_file_path = input_file
                
                # Abrir el archivo ROOT en modo actualización
                root_file = ROOT.TFile(root_file_path, 'UPDATE')
                
                # Crear o acceder al directorio prob_histo
                root_dir = root_file.GetDirectory('prob_histo')
                if not root_dir:
                    root_dir = root_file.mkdir('prob_histo')
                root_dir.cd()
                
                # Guardar cada histograma
                for hist in root_histos:
                    hist.Write(hist.GetName(), ROOT.TObject.kOverwrite)
                
                # Cerrar el archivo
                root_file.Close()
                print(f"\nHistogramas guardados en el archivo ROOT: {root_file_path}:/prob_histo/")
                
            except Exception as e:
                print(f"\n⚠️  Error al guardar los histogramas en el archivo ROOT: {e}")
                print("Los histogramas se han guardado correctamente en formato PNG, pero no en el archivo ROOT.")
            
            """
            # -------------------------------------------------------------------
            # 6.5.11. Histogramas de probabilidades de clasificación (filtrados por umbral)
            # -------------------------------------------------------------------
            print("\nGenerando histogramas de probabilidades de clasificación (datos filtrados)...")
            
            # Crear figura para los histogramas
            plt.figure(figsize=(15, 5))
            
            # Para cada tipo de partícula (e-, e+, gamma)
            for i, (particle_name, color) in enumerate(zip(
                ['Electrones', 'Positrones', 'Gammas'],
                ['blue', 'red', 'green']
            )):
                plt.subplot(1, 3, i+1)
                mask = (filtered_true_labels == i)  # Usar filtered_true_labels
                
                if np.sum(mask) > 0:  # Si hay eventos de este tipo
                    # Graficar histograma de las probabilidades de la clase correcta
                    probs = filtered_probabilities[mask, i]  # Usar filtered_probabilities
                    plt.hist(probs, 
                            bins=20, 
                            alpha=0.7, 
                            color=color,
                            edgecolor='black')
                    plt.title(f'{particle_name} reales (n={np.sum(mask)})')
                    plt.xlabel(f'Prob. de ser {particle_name.lower().rstrip("s")}')
                    plt.ylabel('Número de eventos')
                    plt.xlim(0, 1)
                    plt.grid(True, alpha=0.3)
                else:
                    plt.text(0.5, 0.5, f'No hay {particle_name.lower()} reales', 
                            ha='center', va='center')
                    plt.xlim(0, 1)
                    plt.ylim(0, 1)
            
            plt.tight_layout()
            plt.suptitle(f'Distribución de probabilidades (filtrado, {len(filtered_true_labels)} eventos)', y=1.05)
            plt.savefig(f'{plots_dir}/histogramas_probabilidades_filtrados.png', dpi=300, bbox_inches='tight')
            plt.show()  # Mostrar la figura de los histogramas
            """
            # -------------------------------------------------------------------
            # 6.5.11. Histogramas de probabilidades detallados por clase real y predicha (filtrados)
            # -------------------------------------------------------------------
            
            print("\nGenerando histogramas detallados de probabilidades por clase real y predicha (filtrados)...")
            
            # Lista para guardar los histogramas de ROOT filtrados
            root_histos_filtered = []
            
            # Crear una figura grande para todos los histogramas
            plt.figure(figsize=(20, 15))
            
            # Índice para el subplot
            plot_idx = 1
            
            # Para cada tipo de partícula REAL (filas)
            for true_idx, (true_name, true_short) in enumerate(zip(particle_names, particle_short)):
                # Para cada tipo de partícula PREDICHA (columnas)
                for pred_idx, (pred_name, pred_short, color) in enumerate(zip(particle_names, particle_short, colors)):
                    plt.subplot(3, 3, plot_idx)
                    
                    # Obtener máscara para la partícula real actual en los datos filtrados
                    mask = (filtered_true_labels == true_idx)
                    
                    if np.sum(mask) > 0:  # Si hay eventos de este tipo
                        # Obtener probabilidades para la clase predicha actual
                        probs = filtered_probabilities[mask, pred_idx]
                        
                        # Crear histograma de matplotlib
                        n, bins, _ = plt.hist(probs, 
                                        bins=20, 
                                        alpha=0.7, 
                                        color=color,
                                        edgecolor='black')  # Mostrar número de eventos absolutos
                                                
                        # Crear histograma de ROOT con estilo de solo barras
                        hist_name = f'h_prob_{particle_root[true_idx]}_as_{particle_root[pred_idx]}_filtered'
                        hist_title = f'{particle_root[true_idx]} as {particle_root[pred_idx]} (filtered);Probability;Counts'
                        nbins = 20
                        xmin, xmax = 0.0, 1.0
                        
                        # Crear y configurar histograma
                        root_hist = ROOT.TH1F(hist_name, hist_title, nbins, xmin, xmax)
                        root_hist.SetStats(0)  # Desactivar caja de estadísticas
                        root_hist.SetLineColor(ROOT.kBlack)
                        root_hist.SetLineWidth(1)
                        root_hist.SetFillColor(ROOT.kBlue)  # Color de relleno
                        root_hist.SetFillStyle(1001)  # Relleno sólido
                        
                        # Llenar el histograma
                        for prob in probs:
                            root_hist.Fill(prob)
                        
                        # Añadir a la lista de histogramas
                        root_histos_filtered.append(root_hist)
                        
                        # Configurar título y etiquetas
                        plt.title(f'Reales: {true_name}\nPredichos como: {pred_name}\n(Eventos filtrados)')
                        plt.xlabel(f'Prob. de ser {pred_short}')
                        plt.ylabel('Número de eventos')
                        plt.xlim(0, 1)
                        plt.grid(True, alpha=0.3)
                        
                        # Añadir el número de eventos en la esquina superior derecha
                        plt.text(0.98, 0.95, f'N = {np.sum(mask):,}', 
                                transform=plt.gca().transAxes,
                                ha='right', va='top',
                                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
                    else:
                        plt.text(0.5, 0.5, f'No hay {true_name.lower()} reales\n(filtrados)', 
                                ha='center', va='center')
                        plt.xlim(0, 1)
                        plt.ylim(0, 1)
                    
                    # Incrementar el índice del subplot
                    plot_idx += 1
            
            # Ajustar el diseño y guardar la figura
            plt.tight_layout()
            plt.suptitle(f'Distribuciones de probabilidad por clase real y predicha\n(Eventos filtrados, {len(filtered_true_labels):,} eventos)', y=1.02, fontsize=14)
            plt.savefig(f'{plots_dir}/histogramas_probabilidades_detallados_filtrados.png', dpi=300, bbox_inches='tight')
            plt.show()
            print("Histogramas detallados (filtrados) guardados en:", f'{plots_dir}/histogramas_probabilidades_detallados_filtrados.png')
            
            # Guardar los histogramas en el archivo ROOT
            try:
                # Obtener el nombre del archivo ROOT de entrada
                if hasattr(file, '_file') and hasattr(file._file, 'fFilePath'):
                    root_file_path = file._file.fFilePath
                else:
                    root_file_path = input_file
                
                # Abrir el archivo ROOT en modo actualización
                root_file = ROOT.TFile(root_file_path, 'UPDATE')
                
                # Crear o acceder al directorio prob_histo_filtered
                root_dir = root_file.GetDirectory('prob_histo_filtered')
                if not root_dir:
                    root_dir = root_file.mkdir('prob_histo_filtered')
                root_dir.cd()
                
                # Guardar cada histograma filtrado
                for hist in root_histos_filtered:
                    hist.Write(hist.GetName(), ROOT.TObject.kOverwrite)
                
                # Cerrar el archivo
                root_file.Close()
                print(f"\nHistogramas filtrados guardados en el archivo ROOT: {root_file_path}:/prob_histo_filtered/")
                
            except Exception as e:
                print(f"\n⚠️  Error al guardar los histogramas filtrados en el archivo ROOT: {e}")
                print("Los histogramas filtrados se han guardado correctamente en formato PNG, pero no en el archivo ROOT.")
            
            # -------------------------------------------------------------------
            # 6.5.12.1 Curva ROC para positrones vs gammas
            # -------------------------------------------------------------------
            
            print("\nGenerando curva ROC para positrones vs gammas...")
            
            # Obtener máscaras para positrones reales
            true_positrons = (true_labels == 1)  # Índice 1 para positrones
            
            # Obtener probabilidades de ser positrones y gammas para todos los eventos
            probs_positron = probabilities[:, 1]  # Probabilidad de ser positrón
            probs_gamma = probabilities[:, 2]      # Probabilidad de ser gamma
            
            # Umbrales para la curva ROC (de 0 a 1 con pasos de 0.01)
            thresholds = np.linspace(0, 1, 101)
            
            # Arrays para almacenar TPR y FPR
            tpr = []  # True Positive Rate
            fpr = []  # False Positive Rate
            
            for thresh in thresholds:
                # Predicción de positrón (positivo) cuando la probabilidad > umbral
                pred_positron = probs_positron > thresh
                
                # Calcular TPR: VP / (VP + FN)
                vp = np.sum(true_positrons & pred_positron)
                fn = np.sum(true_positrons & ~pred_positron)
                tpr_val = vp / (vp + fn) if (vp + fn) > 0 else 0
                
                # Calcular FPR: FP / (FP + TN)
                true_gammas = (true_labels == 2)  # Índice 2 para gammas
                fp = np.sum(true_gammas & pred_positron)
                tn = np.sum(true_gammas & ~pred_positron)
                fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0
                
                tpr.append(tpr_val)
                fpr.append(fpr_val)
                
                # Mostrar TPR y FPR para umbrales 0.1, 0.2, ..., 0.9
                if round(thresh*10) == thresh*10 and 0 < thresh < 1.0:
                    print(f"Umbral {thresh:.1f}: TPR = {tpr_val:.4f}, FPR = {fpr_val:.4f}")
            
            # Convertir a arrays de numpy
            tpr = np.array(tpr)
            fpr = np.array(fpr)
            
            # Encontrar puntos donde TPR está más cercano a 0.5 y 0.8
            idx_05 = np.argmin(np.abs(tpr - 0.5))
            idx_08 = np.argmin(np.abs(tpr - 0.8))
            fpr_at_tpr_05 = fpr[idx_05]
            fpr_at_tpr_08 = fpr[idx_08]
            print(f"\n[Positrones vs Gammas]")
            print(f"Cuando TPR ≈ 0.5, FPR = {fpr_at_tpr_05:.4f}")
            print(f"Cuando TPR ≈ 0.8, FPR = {fpr_at_tpr_08:.4f}")
            
            # Calcular el área bajo la curva ROC (AUC)
            # Las métricas ya están importadas al inicio del archivo
            roc_auc = auc(fpr, tpr)
            
            # Crear la figura
            plt.figure(figsize=(10, 8))
            plt.semilogy(tpr, fpr, color='darkorange', lw=2, 
                        label=f'Curva ROC (AUC = {roc_auc:.3f})')
            plt.xlim([0.5, 1.0])  # Rango del eje X ajustado a [0.5, 1.0]
            plt.ylim([1e-4, 1.05])  # Límite inferior pequeño pero no cero para escala log
            plt.xlabel('TPR (True Positive Rate)')
            plt.ylabel('FPR (False Positive Rate, escala log)')
            plt.title('Curva ROC: Positrones vs Gammas')
            plt.legend(loc="lower right")
            plt.grid(True, which="both", alpha=0.3)
            
            # Guardar la figura
            plt.savefig(f'{plots_dir}/roc_curve_positron_vs_gamma.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print("Curva ROC guardada en:", f'{plots_dir}/roc_curve_positron_vs_gamma.png')
            
            # -------------------------------------------------------------------
            # 6.5.12.2 Curva ROC para Gammas vs Electrones
            # -------------------------------------------------------------------
            
            print("\nGenerando curva ROC para gammas vs electrones...")
            
            # Obtener máscaras para gammas reales
            true_gammas = (true_labels == 2)  # Índice 2 para gammas
            
            # Obtener probabilidades de ser gammas y electrones para todos los eventos
            probs_gamma = probabilities[:, 2]     # Probabilidad de ser gamma
            probs_electron = probabilities[:, 0]   # Probabilidad de ser electrón
            
            # Umbrales para la curva ROC (de 0 a 1 con pasos de 0.01)
            thresholds = np.linspace(0, 1, 101)
            
            # Arrays para almacenar TPR y FPR
            tpr_g = []  # True Positive Rate para gammas
            fpr_g = []  # False Positive Rate (clasificar electrones como gammas)
            
            for thresh in thresholds:
                # Predicción de gamma (positivo) cuando la probabilidad > umbral
                pred_gamma = probs_gamma > thresh
                
                # Calcular TPR: VP / (VP + FN)
                vp = np.sum(true_gammas & pred_gamma)
                fn = np.sum(true_gammas & ~pred_gamma)
                tpr_val = vp / (vp + fn) if (vp + fn) > 0 else 0
                
                # Calcular FPR: FP / (FP + TN)
                true_electrons = (true_labels == 0)  # Índice 0 para electrones
                fp = np.sum(true_electrons & pred_gamma)
                tn = np.sum(true_electrons & ~pred_gamma)
                fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0
                
                tpr_g.append(tpr_val)
                fpr_g.append(fpr_val)
            
            # Convertir a arrays de numpy
            tpr_g = np.array(tpr_g)
            fpr_g = np.array(fpr_g)
            
            # Calcular el área bajo la curva ROC (AUC)
            roc_auc_g = auc(fpr_g, tpr_g)
            
            # Crear la figura
            plt.figure(figsize=(10, 8))
            plt.semilogy(tpr_g, fpr_g, color='green', lw=2, 
                        label=f'Curva ROC (AUC = {roc_auc_g:.3f})')
            plt.xlim([0.5, 1.0])  # Mismo rango que para positrones
            plt.ylim([1e-4, 1.05])
            plt.xlabel('TPR (True Positive Rate)')
            plt.ylabel('FPR (False Positive Rate, escala log)')
            plt.title('Curva ROC: Gammas vs Electrones')
            plt.legend(loc="lower right")
            plt.grid(True, which="both", alpha=0.3)
            
            # Guardar la figura
            plt.savefig(f'{plots_dir}/roc_curve_gamma_vs_electron.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print("Curva ROC guardada en:", f'{plots_dir}/roc_curve_gamma_vs_electron.png')
            
            # -------------------------------------------------------------------
            # 6.5.12.3 Curva ROC para Electrones vs Gammas
            # -------------------------------------------------------------------
            
            print("\nGenerando curva ROC para electrones vs gammas...")
            
            # Obtener máscaras para electrones reales
            true_electrons = (true_labels == 0)  # Índice 0 para electrones
            
            # Obtener probabilidades de ser electrones y gammas para todos los eventos
            probs_electron = probabilities[:, 0]  # Probabilidad de ser electrón
            probs_gamma = probabilities[:, 2]     # Probabilidad de ser gamma
            
            # Umbrales para la curva ROC (de 0 a 1 con pasos de 0.01)
            thresholds = np.linspace(0, 1, 101)
            
            # Arrays para almacenar TPR y FPR
            tpr_e = []  # True Positive Rate para electrones
            fpr_e = []  # False Positive Rate (clasificar gammas como electrones)
            
            for thresh in thresholds:
                # Predicción de electrón (positivo) cuando la probabilidad > umbral
                pred_electron = probs_electron > thresh
                
                # Calcular TPR: VP / (VP + FN)
                vp = np.sum(true_electrons & pred_electron)
                fn = np.sum(true_electrons & ~pred_electron)
                tpr_val = vp / (vp + fn) if (vp + fn) > 0 else 0
                
                # Calcular FPR: FP / (FP + TN)
                true_gammas = (true_labels == 2)  # Índice 2 para gammas
                fp = np.sum(true_gammas & pred_electron)    # Gammas mal clasificados como electrones
                tn = np.sum(true_gammas & ~pred_electron)   # Gammas correctamente identificados
                fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0
                
                tpr_e.append(tpr_val)
                fpr_e.append(fpr_val)
            
            # Convertir a arrays de numpy
            tpr_e = np.array(tpr_e)
            fpr_e = np.array(fpr_e)
            
            # Calcular el área bajo la curva ROC (AUC)
            roc_auc_e = auc(fpr_e, tpr_e)
            
            # Crear la figura
            plt.figure(figsize=(10, 8))
            plt.semilogy(tpr_e, fpr_e, color='blue', lw=2, 
                        label=f'Curva ROC (AUC = {roc_auc_e:.3f})')
            plt.xlim([0.5, 1.0])  # Mismo rango que para las otras curvas
            plt.ylim([1e-4, 1.05])  # Límite inferior pequeño pero no cero para escala log
            plt.xlabel('TPR (True Positive Rate)')
            plt.ylabel('FPR (False Positive Rate, escala log)')
            plt.title('Curva ROC: Electrones vs Gammas')
            plt.legend(loc="lower right")
            plt.grid(True, which="both", alpha=0.3)
            
            # Guardar la figura
            plt.savefig(f'{plots_dir}/roc_curve_electron_vs_gamma.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print("Curva ROC guardada en:", f'{plots_dir}/roc_curve_electron_vs_gamma.png')
            
            # -------------------------------------------------------------------
            # 6.5.12.2 Curva ROC: Electrones vs Positrones
            # -------------------------------------------------------------------
            print("\nGenerando curva ROC para electrones vs positrones...")
            
            # Obtener máscaras para electrones y positrones reales
            true_electrons = (true_labels == 0)  # Índice 0 para electrones
            true_positrons = (true_labels == 1)  # Índice 1 para positrones
            
            # Obtener probabilidades de ser electrones para todos los eventos
            probs_electron = probabilities[:, 0]  # Probabilidad de ser electrón
            
            # Umbrales para la curva ROC (de 0 a 1 con pasos de 0.01)
            thresholds = np.linspace(0, 1, 101)
            
            # Arrays para almacenar TPR y FPR
            tpr_e_pos = []  # True Positive Rate para electrones
            fpr_e_pos = []  # False Positive Rate (clasificar positrones como electrones)
            
            for thresh in thresholds:
                # Predicción de electrón (positivo) cuando la probabilidad > umbral
                pred_electron = probs_electron > thresh
                
                # Calcular TPR: VP / (VP + FN) para electrones
                vp = np.sum(true_electrons & pred_electron)
                fn = np.sum(true_electrons & ~pred_electron)
                tpr_val = vp / (vp + fn) if (vp + fn) > 0 else 0
                
                # Calcular FPR: FP / (FP + TN) para positrones
                fp = np.sum(true_positrons & pred_electron)    # Positrones mal clasificados como electrones
                tn = np.sum(true_positrons & ~pred_electron)   # Positrones correctamente identificados
                fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0
                
                tpr_e_pos.append(tpr_val)
                fpr_e_pos.append(fpr_val)
            
            # Convertir a arrays de numpy
            tpr_e_pos = np.array(tpr_e_pos)
            fpr_e_pos = np.array(fpr_e_pos)
            
            # Calcular el área bajo la curva ROC (AUC)
            roc_auc_e_pos = auc(fpr_e_pos, tpr_e_pos)
            
            # Crear la figura
            plt.figure(figsize=(10, 8))
            plt.semilogy(tpr_e_pos, fpr_e_pos, color='green', lw=2, 
                        label=f'Curva ROC (AUC = {roc_auc_e_pos:.3f})')
            plt.xlim([0.5, 1.0])  # Mismo rango que para las otras curvas
            plt.ylim([1e-4, 1.05])  # Límite inferior pequeño pero no cero para escala log
            plt.xlabel('TPR (True Positive Rate)')
            plt.ylabel('FPR (False Positive Rate, escala log)')
            plt.title('Curva ROC: Electrones vs Positrones')
            plt.legend(loc="lower right")
            plt.grid(True, which="both", alpha=0.3)
            
            # Guardar la figura
            plt.savefig(f'{plots_dir}/roc_curve_electron_vs_positron.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print("Curva ROC guardada en:", f'{plots_dir}/roc_curve_electron_vs_positron.png')
            
            # -------------------------------------------------------------------
            # 6.5.12.3 Curva ROC: Positrones vs Electrones
            # -------------------------------------------------------------------
            print("\nGenerando curva ROC para positrones vs electrones...")
            
            # Obtener máscaras para positrones y electrones reales
            true_positrons = (true_labels == 1)  # Índice 1 para positrones
            true_electrons = (true_labels == 0)  # Índice 0 para electrones
            
            # Obtener probabilidades de ser positrones para todos los eventos
            probs_positron = probabilities[:, 1]  # Probabilidad de ser positrón
            
            # Umbrales para la curva ROC (de 0 a 1 con pasos de 0.01)
            thresholds = np.linspace(0, 1, 101)
            
            # Arrays para almacenar TPR y FPR
            tpr_pos_e = []  # True Positive Rate para positrones
            fpr_pos_e = []  # False Positive Rate (clasificar electrones como positrones)
            
            for thresh in thresholds:
                # Predicción de positrón (positivo) cuando la probabilidad > umbral
                pred_positron = probs_positron > thresh
                
                # Calcular TPR: VP / (VP + FN) para positrones
                vp = np.sum(true_positrons & pred_positron)
                fn = np.sum(true_positrons & ~pred_positron)
                tpr_val = vp / (vp + fn) if (vp + fn) > 0 else 0
                
                # Calcular FPR: FP / (FP + TN) para electrones
                fp = np.sum(true_electrons & pred_positron)    # Electrones mal clasificados como positrones
                tn = np.sum(true_electrons & ~pred_positron)   # Electrones correctamente identificados
                fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0
                
                tpr_pos_e.append(tpr_val)
                fpr_pos_e.append(fpr_val)

                # Mostrar TPR y FPR para umbrales 0.1, 0.2, ..., 0.9
                if round(thresh*10) == thresh*10 and 0 < thresh < 1.0:
                    print(f"Umbral {thresh:.1f}: TPR = {tpr_val:.4f}, FPR = {fpr_val:.4f}")
            
            # Convertir a arrays de numpy
            tpr_pos_e = np.array(tpr_pos_e)
            fpr_pos_e = np.array(fpr_pos_e)
            
            # Encontrar puntos donde TPR está más cercano a 0.5 y 0.8
            idx_05 = np.argmin(np.abs(tpr_pos_e - 0.5))
            idx_08 = np.argmin(np.abs(tpr_pos_e - 0.8))
            fpr_at_tpr_05 = fpr_pos_e[idx_05]
            fpr_at_tpr_08 = fpr_pos_e[idx_08]
            print(f"\n[Positrones vs Electrones]")
            print(f"Cuando TPR ≈ 0.5, FPR = {fpr_at_tpr_05:.4f}")
            print(f"Cuando TPR ≈ 0.8, FPR = {fpr_at_tpr_08:.4f}")
            
            # Calcular el área bajo la curva ROC (AUC)
            roc_auc_pos_e = auc(fpr_pos_e, tpr_pos_e)
            
            # Crear la figura
            plt.figure(figsize=(10, 8))
            plt.semilogy(tpr_pos_e, fpr_pos_e, color='purple', lw=2, 
                        label=f'Curva ROC (AUC = {roc_auc_pos_e:.3f})')
            plt.xlim([0.5, 1.0])  # Mismo rango que para las otras curvas
            plt.ylim([1e-4, 1.05])  # Límite inferior pequeño pero no cero para escala log
            plt.xlabel('TPR (True Positive Rate)')
            plt.ylabel('FPR (False Positive Rate, escala log)')
            plt.title('Curva ROC: Positrones vs Electrones')
            plt.legend(loc="lower right")
            plt.grid(True, which="both", alpha=0.3)
            
            # Guardar la figura
            plt.savefig(f'{plots_dir}/roc_curve_positron_vs_electron.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print("Curva ROC guardada en:", f'{plots_dir}/roc_curve_positron_vs_electron.png')
            
            
            # -------------------------------------------------------------------
            # 6.5.12.4 Análisis de umbrales para e+ vs γ y e+ vs e- (ALGORITMO EN PRUEBAS)
            # -------------------------------------------------------------------
            print("\n🔍 Analizando umbrales para clasificación de positrones...")
            
            # Obtener probabilidades de ser positrón y ser gamma
            probs_positron = probabilities[:, 1]  # Probabilidad de ser e+
            probs_gamma = probabilities[:, 2]     # Probabilidad de ser gamma
            
            # Umbrales a evaluar
            thresholds = np.linspace(0.1, 0.9, 9)
            
            # Resultados
            print("\nUmbral | TPR (e+ vs γ) | FPR (e+ vs γ) | TPR (e+ vs e-) | FPR (e+ vs e-)")
            print("-" * 75)
            
            for thresh in thresholds:
                # Clasificación de positrones vs gamma
                pred_positron_vs_gamma = (probs_positron >= thresh)
                
                # Calcular métricas para e+ vs γ
                tp_gamma = np.sum(true_positrons & pred_positron_vs_gamma)
                fn_gamma = np.sum(true_positrons & ~pred_positron_vs_gamma)
                fp_gamma = np.sum((true_labels == 2) & pred_positron_vs_gamma)
                tn_gamma = np.sum((true_labels == 2) & ~pred_positron_vs_gamma)
                
                tpr_vs_gamma = tp_gamma / (tp_gamma + fn_gamma) if (tp_gamma + fn_gamma) > 0 else 0
                fpr_vs_gamma = fp_gamma / (fp_gamma + tn_gamma) if (fp_gamma + tn_gamma) > 0 else 0
                
                # Calcular métricas para e+ vs e-
                fp_eminus = np.sum((true_labels == 0) & pred_positron_vs_gamma)
                tn_eminus = np.sum((true_labels == 0) & ~pred_positron_vs_gamma)
                fpr_vs_eminus = fp_eminus / (fp_eminus + tn_eminus) if (fp_eminus + tn_eminus) > 0 else 0
                
                # Imprimir resultados
                print(f"{thresh:.2f}   | {tpr_vs_gamma:^12.3f}  | {fpr_vs_gamma:^12.3f}  | {tpr_vs_gamma:^12.3f}  | {fpr_vs_eminus:^12.3f}")
            
            # Guardar resultados en un archivo
            threshold_file = f'{plots_dir}/umbrales_positrones.txt'
            with open(threshold_file, 'w') as f:
                f.write("Análisis de umbrales para clasificación de positrones\n")
                f.write("="*60 + "\n\n")
                f.write("Umbral | TPR (e+ vs γ) | FPR (e+ vs γ) | TPR (e+ vs e-) | FPR (e+ vs e-)\n")
                f.write("-"*75 + "\n")
                
                for thresh in thresholds:
                    pred_positron_vs_gamma = (probs_positron >= thresh)
                    
                    # Métricas para e+ vs γ
                    tp_gamma = np.sum(true_positrons & pred_positron_vs_gamma)
                    fn_gamma = np.sum(true_positrons & ~pred_positron_vs_gamma)
                    fp_gamma = np.sum((true_labels == 2) & pred_positron_vs_gamma)
                    tn_gamma = np.sum((true_labels == 2) & ~pred_positron_vs_gamma)
                    
                    tpr_vs_gamma = tp_gamma / (tp_gamma + fn_gamma) if (tp_gamma + fn_gamma) > 0 else 0
                    fpr_vs_gamma = fp_gamma / (fp_gamma + tn_gamma) if (fp_gamma + tn_gamma) > 0 else 0
                    
                    # Métricas para e+ vs e-
                    fp_eminus = np.sum((true_labels == 0) & pred_positron_vs_gamma)
                    tn_eminus = np.sum((true_labels == 0) & ~pred_positron_vs_gamma)
                    fpr_vs_eminus = fp_eminus / (fp_eminus + tn_eminus) if (fp_eminus + tn_eminus) > 0 else 0
                    
                    f.write(f"{thresh:.2f}   | {tpr_vs_gamma:^12.3f}  | {fpr_vs_gamma:^12.3f}  | {tpr_vs_gamma:^12.3f}  | {fpr_vs_eminus:^12.3f}\n")
            
            print(f"\n📄 Resultados detallados guardados en: {threshold_file}")

            # -------------------------------------------------------------------
            # 6.5.13. Mapas de calor
            # -------------------------------------------------------------------
            bins = 20
            range_ = [[0, 1], [0, 1]]

            # Crear máscaras para cada tipo de partícula real (usando datos filtrados)
            is_electron = (filtered_true_labels == 0)
            is_positron = (filtered_true_labels == 1)
            is_gamma = (filtered_true_labels == 2)
            
            # Crear una nueva figura para los mapas de calor
            plt.figure(figsize=(15, 6))
            fig_heatmap = plt.gcf()  # Obtener la figura actual
            
            # -------------------------------------------------------------------
            # 6.5.13.1 Mapa de calor: Probabilidades para positrones reales
            # -------------------------------------------------------------------
            
            plt.subplot(1, 2, 1)
            if np.any(is_positron):
                plt.hist2d(filtered_probabilities[is_positron, 1],  # Prob. de ser e+
                           filtered_probabilities[is_positron, 2],  # Prob. de ser gamma
                           bins=bins, range=range_, 
                           cmap='Reds', norm=LogNorm())
                plt.colorbar(label='Número de eventos')
                plt.title('e+ reales')
                plt.xlabel('Prob. de ser e+')
                plt.ylabel('Prob. de ser gamma')
            else:
                plt.text(0.5, 0.5, 'No hay e+ reales', 
                         ha='center', va='center')
            
            # -------------------------------------------------------------------
            # 6.5.13.2 Mapa de calor: Probabilidades para gammas reales
            # -------------------------------------------------------------------
            plt.subplot(1, 2, 2)
            if np.any(is_gamma):
                plt.hist2d(filtered_probabilities[is_gamma, 1],  # Prob. de ser e+
                          filtered_probabilities[is_gamma, 2],    # Prob. de ser gamma
                          bins=bins, range=range_,
                          cmap='Blues', norm=LogNorm())
                plt.colorbar(label='Número de eventos')
                plt.title('Gammas reales')
                plt.xlabel('Prob. de ser e+')
                plt.ylabel('Prob. de ser gamma')
            else:
                plt.text(0.5, 0.5, 'No hay gammas reales',
                         ha='center', va='center')
            
            # Ajustar el diseño de la figura
            plt.tight_layout()
            
            # Guardar la figura de los mapas de calor
            output_file = f'{plots_dir}/mapas_calor.png'
            fig_heatmap.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f'   - Gráfico guardado: {output_file}')
            plt.show()
            
            # ===================================================================
            # 6.6. ANÁLISIS POR ENERGÍA CINÉTICA
            # ===================================================================
            print("\n⚡ Analizando rendimiento por energía cinética...")
            
            # -------------------------------------------------------------------
            # 6.6.1. Validación y preparación de datos
            # -------------------------------------------------------------------
            print("   - Validando y preparando datos...")
            
            # Asegurar consistencia en la longitud de los arrays
            min_length = min(len(filtered_true_labels), len(filtered_predicted_labels), len(filtered_mcke_values))
            
            if min_length == 0:
                print("❌ Error: No hay datos válidos para analizar.")
                return
                
            # Ajustar longitudes si es necesario
            if min_length < len(filtered_true_labels):
                print(f"   - Ajustando longitud de los datos a {min_length} eventos.")
                filtered_true_labels = filtered_true_labels[:min_length]
                filtered_predicted_labels = filtered_predicted_labels[:min_length]
                filtered_confidence_scores = filtered_confidence_scores[:min_length]
                filtered_probabilities = filtered_probabilities[:min_length]
                filtered_mcke_values = filtered_mcke_values[:min_length]
            
            # Usar los valores de mcke ya filtrados
            # Versión original (comentada):
            # ek_values = filtered_mcke_values
            
            # Versión usando valores originales:
            # Asegurarse de que todos los arrays tengan la misma longitud
            min_length = min(len(df['mcke'].values), len(filtered_mcke_values))
            ek_values = df['mcke'].values[:min_length]  # Usar los valores originales de mcke
            
            # Verificación final de longitudes
            if len(ek_values) != len(filtered_true_labels) or len(ek_values) != len(filtered_predicted_labels):
                print("❌ Error: Inconsistencia en las longitudes de los arrays.")
                print(f"   - Valores de energía: {len(ek_values)}")
                print(f"   - Etiquetas reales: {len(filtered_true_labels)}")
                print(f"   - Predicciones: {len(filtered_predicted_labels)}")
                return
            
            # -------------------------------------------------------------------
            # 6.6.2. Inicialización de estructuras de datos
            # -------------------------------------------------------------------
            print("   - Inicializando estructuras de datos...")
            
            # Diccionario para almacenar métricas por tipo de partícula
            metrics = {
                'e+': {'true': [], 'pred_e+': [], 'pred_e-': [], 'pred_gamma': [], 'ek_centers': []},
                'e-': {'true': [], 'pred_e+': [], 'pred_e-': [], 'pred_gamma': [], 'ek_centers': []},
                'gamma': {'true': [], 'pred_e+': [], 'pred_e-': [], 'pred_gamma': [], 'ek_centers': []}
            }
            
            # Mapeo de etiquetas a nombres de partículas
            LABEL_TO_NAME = {0: 'e-', 1: 'e+', 2: 'gamma'}
            
            # -------------------------------------------------------------------
            # 6.6.3. Análisis por intervalos de energía
            # -------------------------------------------------------------------
            print("   - Procesando intervalos de energía...")
            
            # Diccionario para almacenar métricas ROC por intervalo
            roc_metrics = {}
            
            for i, (low, high) in enumerate(EK_INTERVALS):
                # Determinar máscara para el intervalo de energía actual en MeV
                interval_str = f"{low}-{high} MeV" if high != float('inf') else f">{low} MeV"
                print(f"      - Procesando intervalo {interval_str}...")
                
                if high == float('inf'):
                    ek_mask = (ek_values >= low)  # Último intervalo (sin límite superior)
                else:
                    ek_mask = (ek_values >= low) & (ek_values < high)
                
                # Saltar si no hay eventos en este intervalo
                if ek_mask.sum() == 0:
                    print(f"         No hay eventos en el intervalo {interval_str}")
                    continue
                
                # Extraer datos para este intervalo de energía
                # Versión original (comentada):
                # true_ek = filtered_true_labels[ek_mask]      # Etiquetas reales
                # pred_ek = filtered_predicted_labels[ek_mask]  # Predicciones
                # prob_ek = filtered_probabilities[ek_mask]     # Probabilidades
                
                # Versión usando valores originales:
                # Asegurarse de que la máscara tenga la longitud correcta
                min_length = min(len(true_labels), len(ek_mask))
                true_ek = true_labels[:min_length][ek_mask[:min_length]]
                
                min_length = min(len(predicted_labels), len(ek_mask))
                pred_ek = predicted_labels[:min_length][ek_mask[:min_length]]
                
                min_length = min(probabilities.shape[0], len(ek_mask))
                prob_ek = probabilities[:min_length][ek_mask[:min_length]]
                
                # Almacenar métricas ROC para este intervalo
                roc_metrics[interval_str] = {
                    'true': true_ek,
                    'pred': pred_ek,
                    'prob': prob_ek,
                    'n_events': len(true_ek)
                }
                
                # Calcular el centro del intervalo para el eje X
                ek_center = (low + high) / 2 if high != float('inf') else low * 1.5
                
                # Calcular métricas para cada tipo de partícula
                for true_label, true_name in LABEL_TO_NAME.items():
                    # Crear máscara para la partícula actual
                    part_mask = (true_ek == true_label)
                    total = part_mask.sum()
                    
                    if total > 0:  # Solo si hay partículas de este tipo en el intervalo
                        # Calcular porcentajes de clasificación
                        pred_counts = {
                            'e+': ((pred_ek[part_mask] == 1).sum() / total) * 100,
                            'e-': ((pred_ek[part_mask] == 0).sum() / total) * 100,
                            'gamma': ((pred_ek[part_mask] == 2).sum() / total) * 100
                        }
                        
                        # Almacenar métricas
                        metrics[true_name]['true'].append(total)
                        metrics[true_name]['pred_e+'].append(pred_counts['e+'])
                        metrics[true_name]['pred_e-'].append(pred_counts['e-'])
                        metrics[true_name]['pred_gamma'].append(pred_counts['gamma'])
                        metrics[true_name]['ek_centers'].append(ek_center)
            
            # -------------------------------------------------------------------
            # 6.6.4. Generación de curvas ROC por intervalo de energía
            # -------------------------------------------------------------------
            print("\n   - Generando curvas ROC por intervalo de energía...")
            
            # Configuración de colores para los intervalos
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
            
            # Tipos de partículas para las curvas ROC
            particle_pairs = [
                ('e+', 'gamma', 'Positrones vs Gammas'),
                ('gamma', 'e-', 'Gammas vs Electrones'),
                ('e-', 'gamma', 'Electrones vs Gammas'),
                ('e+', 'e-', 'Positrones vs Electrones')  
            ]
            
            for true_particle, false_particle, title in particle_pairs:
                plt.figure(figsize=(10, 8))
                
                # Mapeo de etiquetas
                true_label = {'e+': 1, 'e-': 0, 'gamma': 2}[true_particle]
                false_label = {'e+': 1, 'e-': 0, 'gamma': 2}[false_particle]
                
                # Para cada intervalo de energía
                for i, (interval_str, data) in enumerate(roc_metrics.items()):
                    if data['n_events'] < 10:  # Mínimo de eventos para calcular ROC
                        continue
                        
                    # Filtrar solo las partículas de interés
                    mask = (data['true'] == true_label) | (data['true'] == false_label)
                    if mask.sum() == 0:
                        continue
                        
                    true_labels = (data['true'][mask] == true_label).astype(int)
                    prob = data['prob'][mask, true_label if true_particle != 'gamma' else 2]
                    
                    # Método 1: Cálculo manual con 100 umbrales (más suave)
                    thresholds = np.linspace(0, 1, 101)
                    tpr_vals = []
                    fpr_vals = []
                    
                    for thresh in thresholds:
                        # Predicción según el umbral
                        pred = (prob >= thresh).astype(int)
                        
                        # Calcular VP, VN, FP, FN
                        vp = np.sum((true_labels == 1) & (pred == 1))
                        fp = np.sum((true_labels == 0) & (pred == 1))
                        vn = np.sum((true_labels == 0) & (pred == 0))
                        fn = np.sum((true_labels == 1) & (pred == 0))
                        
                        # Calcular TPR y FPR
                        tpr_val = vp / (vp + fn) if (vp + fn) > 0 else 0
                        fpr_val = fp / (fp + vn) if (fp + vn) > 0 else 0
                        
                        tpr_vals.append(tpr_val)
                        fpr_vals.append(fpr_val)
                    
                    # Método alternativo: Usando roc_curve de scikit-learn (menos suave)
                    # fpr, tpr, _ = roc_curve(true_labels, prob)
                    # tpr_vals, fpr_vals = tpr, fpr
                    
                    # Calcular AUC
                    roc_auc = auc(fpr_vals, tpr_vals)
                    
                    # Contar partículas reales en el intervalo actual
                    n_true_pos = (data['true'][mask] == true_label).sum()
                    n_true_neg = (data['true'][mask] == false_label).sum()
                    
                    # Graficar con escala logarítmica en el eje Y
                    plt.semilogy(tpr_vals, fpr_vals, color=colors[i % len(colors)], lw=2,
                               label=f'{interval_str} (AUC={roc_auc:.3f}, {true_particle}:{n_true_pos}, {false_particle}:{n_true_neg})')
                
                # Configuración del gráfico
                plt.xlim([0.5, 1.0])  # Rango de 0.5 a 1.0 en el eje X
                plt.ylim([1e-4, 1.05])  # Escala logarítmica en el eje Y
                plt.xlabel('TPR (True Positive Rate)')
                plt.ylabel('FPR (False Positive Rate, escala log)')
                plt.title(f'Curvas ROC por intervalo de energía: {title}')
                plt.legend(loc="lower right")
                plt.grid(True, which="both", alpha=0.3)
                
                # Guardar la figura y mostrarla
                filename = f'roc_{true_particle}_vs_{false_particle}_by_energy.png'.replace('+', 'p')
                plt.tight_layout()
                plt.savefig(f'{plots_dir}/{filename}', dpi=300, bbox_inches='tight')
                print(f"      - Gráfico guardado: {filename}")
                plt.show()
                plt.close()
            
            # -------------------------------------------------------------------
            # 6.6.5. Generación de gráficos de eficiencia
            # -------------------------------------------------------------------
            print("   - Generando gráficos de eficiencia...")
            
            for particle in ['e+', 'e-', 'gamma']:
                if not metrics[particle]['ek_centers']:  # Saltar si no hay datos
                    continue
                    
                # Configurar figura
                plt.figure(figsize=(14, 8))
                
                # Usar los intervalos de energía ya definidos en EK_INTERVALS
                # Reemplazamos float('inf') por 10.0 para la visualización
                energy_intervals = [(low, high if high != float('inf') else 10.0) 
                                 for low, high in EK_INTERVALS]
                
                # Colores más vibrantes para los intervalos
                interval_colors = ['#ffcdd2', '#c8e6c9', '#bbdefb']
                
                # Etiquetas personalizadas para los intervalos
                interval_labels = ['0-1 MeV', '1-3 MeV', '>3 MeV']
                
                # Dibujar áreas sombreadas para cada intervalo de energía
                for i, (e_min, e_max) in enumerate(energy_intervals):
                    plt.axvspan(e_min, e_max, alpha=0.3, color=interval_colors[i], 
                              label=interval_labels[i])
                
                # Ordenar datos por energía cinética
                sorted_idx = np.argsort(metrics[particle]['ek_centers'])
                ek_sorted = np.array(metrics[particle]['ek_centers'])[sorted_idx]
                
                # Graficar tasa de aciertos (identificaciones correctas)
                particle_name = {'e+': 'Positrón', 'e-': 'Electrón', 'gamma': 'Gamma'}[particle]
                plt.plot(ek_sorted, 
                        np.array(metrics[particle][f'pred_{particle}'])[sorted_idx], 
                        'o-', color='#2ecc71', linewidth=2.5, markersize=8,
                        label=f'Correcto: {particle_name}')
                
                # Graficar tasas de falsas identificaciones
                for other in ['e+', 'e-', 'gamma']:
                    if other != particle and metrics[particle][f'pred_{other}']:
                        style = 'o--'
                        color = None
                        particle_names = {'e+': 'Positrón', 'e-': 'Electrón', 'gamma': 'Gamma'}
                        label = f'Error: {particle_names[other]}'
                        
                        if other == 'e+':
                            color = '#e74c3c'  # Rojo
                        elif other == 'e-':
                            color = '#3498db'  # Azul
                        else:  # gamma
                            color = '#9b59b6'  # Púrpura
                        
                        plt.plot(ek_sorted, 
                                np.array(metrics[particle][f'pred_{other}'])[sorted_idx],
                                style, color=color, linewidth=1.5, markersize=5,
                                label=label, alpha=0.8)
                
                # Configuración del gráfico
                particle_title = {'e+': 'Positrones', 'e-': 'Electrones', 'gamma': 'Gammas'}[particle]
                plt.title(f'Identificación de {particle_title} por intervalo de energía', 
                         fontsize=14, pad=15, fontweight='bold')
                plt.xlabel('Energía cinética (MeV)', fontsize=14, labelpad=12)
                plt.ylabel('Porcentaje de eventos (%)', fontsize=14, labelpad=12)
                plt.grid(True, alpha=0.2, linestyle='--', which='both')
                
                # Ajustar límites del eje Y y X
                plt.ylim(-2, 102)
                plt.xlim(0, max(ek_sorted) * 1.1)  # 10% de margen en el eje X
                
                # Añadir leyenda dentro del gráfico (posición vertical entre 40% y 60%)
                plt.legend(loc='center right', frameon=True, framealpha=0.9,
                         fontsize=10, bbox_to_anchor=(0.98, 0.5))
                
                # Mejorar la apariencia general
                plt.gca().spines['top'].set_visible(False)
                plt.gca().spines['right'].set_visible(False)
                plt.tight_layout()
                
                # Guardar la figura con metadatos en la carpeta Plots
                output_file = f'{plots_dir}/eficiencia_por_energia_{particle}.png'
                plt.savefig(output_file, dpi=300, bbox_inches='tight', 
                            metadata={'CreationDate': None, 'Software': 'Python/Matplotlib'})
                print(f'   - Gráfico guardado: {output_file}')
                plt.show()
                plt.close()
            
            # -------------------------------------------------------------------
            # 6.6.5. ANÁLISIS DE PRECISIÓN POR INTERVALO DE ENERGÍA
            # -------------------------------------------------------------------
            print("\n📊 Analizando precisión por intervalo de energía...")
            
            # Diccionario para almacenar métricas por intervalo
            interval_metrics = {}
            
            for i, (low, high) in enumerate(EK_INTERVALS):
                # -------------------------------------------------------------------
                # 6.6.5.1. Filtrar eventos por intervalo de energía
                # -------------------------------------------------------------------
                if high == float('inf'):
                    ek_mask = (ek_values >= low)  # Último intervalo (sin límite superior)
                    interval_str = f">{low:.1f} MeV"
                else:
                    ek_mask = (ek_values >= low) & (ek_values < high)
                    interval_str = f"{low:.1f}-{high:.1f} MeV"
                
                # Saltar si no hay eventos en este intervalo
                if ek_mask.sum() == 0:
                    print(f"\n⚠️  No hay eventos en el intervalo {interval_str}")
                    continue
                
                # -------------------------------------------------------------------
                # 6.6.5.2. Calcular métricas para el intervalo actual
                # -------------------------------------------------------------------
                true_ek = filtered_true_labels[ek_mask]      # Etiquetas reales filtradas
                pred_ek = filtered_predicted_labels[ek_mask]  # Predicciones filtradas
                total_events = len(true_ek)         # Total de eventos
                print(f"\n[DEBUG] total_events (true_ek) = {total_events}")
                
                # Calcular matriz de confusión
                cm_ek = confusion_matrix(true_ek, pred_ek, labels=[0, 1, 2])
                correct = np.sum(cm_ek.diagonal())
                accuracy = correct / total_events
                
                # Almacenar métricas para resumen final
                interval_metrics[interval_str] = {
                    'total': total_events,
                    'accuracy': accuracy,
                    'cm': cm_ek
                }
                
                # -------------------------------------------------------------------
                # 6.6.5.3. Mostrar resultados detallados por consola
                # -------------------------------------------------------------------
                print(f"\n🔍 Intervalo de energía {interval_str}:")
                print(f"   - Total de eventos: {total_events}")
                print(f"   - Precisión global: {accuracy:.4f} ({accuracy*100:.2f}%)")
                
                # Función para calcular y mostrar precisión por partícula
                def print_particle_metrics(particle_name, particle_idx):
                    total_particle = np.sum(cm_ek[particle_idx, :])
                    if total_particle > 0:
                        acc = cm_ek[particle_idx, particle_idx] / total_particle
                        print(f"   - Precisión para {particle_name}: {acc:.4f} ({acc*100:.2f}%)")
                    else:
                        print(f"   - No hay {particle_name} en este intervalo")
                
                # Mostrar métricas para cada tipo de partícula
                print_particle_metrics("electrones", 0)
                print_particle_metrics("positrones", 1)
                print_particle_metrics("gammas", 2)
                
                # Mostrar matriz de confusión reducida
                if total_events > 0:
                    print("\n   Matriz de confusión (fila: real, col: pred):")
                    print(f"   {'':<10} {'e-':<8} {'e+':<8} {'gamma':<8}")
                    for i, (true_name, true_idx) in enumerate([('e-', 0), ('e+', 1), ('gamma', 2)]):
                        row = [f"{cm_ek[true_idx, pred_idx]:<8}" for pred_idx in range(3)]
                        print(f"   {true_name:<10} {' '.join(row)}")
            
            # -------------------------------------------------------------------
            # 6.6.5.4. Mostrar resumen de métricas por intervalo
            # -------------------------------------------------------------------
            if interval_metrics:
                print("\n" + "="*80)
                print("📋 RESUMEN DE PRECISIÓN POR INTERVALO DE ENERGÍA")
                print("="*80)
                print(f"{'Intervalo de energía':<40} {'Eventos':<10} {'Precisión':<15}")
                print("-"*80)
                
                # Mostrar métricas para cada intervalo de energía
                for interval, metrics in interval_metrics.items():
                    print(f"{interval:<40} {metrics['total']:<10} {metrics['accuracy']*100:>6.2f}%")
                
                print("="*80 + "\n")
            
            # ===================================================================
            # 6.7. ANÁLISIS DETALLADO POR EVENTO
            # ===================================================================
            print(f"\n{'='*80}")
            print(f"📊 ANÁLISIS DETALLADO POR EVENTO - {os.path.basename(input_file)}")
            print(f"{'='*80}")
            
            # Inicializar lista para almacenar resultados detallados
            results = []
            
            # Crear máscara para eventos que superan el umbral
            if prob_threshold > 0.0:
                above_threshold = np.max(filtered_probabilities, axis=1) >= prob_threshold
                valid_indices = np.where(above_threshold)[0]
                print(f"\n🔍 Aplicando umbral de confianza del {prob_threshold*100:.0f}%")
                print(f"   - Eventos que superan el umbral: {len(valid_indices)}/{len(df)} ({(len(valid_indices)/len(df))*100:.1f}%)")
                print(f"   - Rango de event_number en eventos seleccionados: {df['event_number'].iloc[valid_indices].min()} a {df['event_number'].iloc[valid_indices].max()}")
            else:
                valid_indices = range(len(df))
                print(f"\n🔍 Procesando todos los {len(df)} eventos (sin umbral de confianza)")
                print(f"   - Rango de event_number: {df['event_number'].min()} a {df['event_number'].max()}")
            
            # -------------------------------------------------------------------
            # 6.7.1. PROCESAMIENTO DE EVENTOS VÁLIDOS
            # -------------------------------------------------------------------
            print(f"\n🔍 Procesando {len(valid_indices)} eventos...")
            start_time = time.time()
            
            # Recorrer solo los eventos que superan el umbral
            for i, idx in enumerate(valid_indices):
                # Obtener información básica del evento
                event_idx = i + 1  # Índices empiezan en 1 para mejor legibilidad
                original_id = int(filtered_event_numbers[i])  # Usar el event_number filtrado
                true_pdg = int(df['mcpdg'].iloc[idx]) if 'mcpdg' in df.columns else None
                
                # Mostrar los valores en cada iteración
                #print(f"event_idx: {event_idx}, original_id: {original_id}")
                
                # Obtener probabilidades y predicción
                event_probs = filtered_probabilities[idx]
                max_prob_idx = np.argmax(event_probs)
                confidence = event_probs[max_prob_idx]
                predicted_particle = LABEL_TO_PARTICLE.get(max_prob_idx, "Desconocido")
                
                # Mapear etiquetas verdaderas
                true_particle = PDG_TO_PARTICLE.get(true_pdg, f"PDG {true_pdg}") if true_pdg is not None else "Desconocido"
                
                # Validar predicción si hay etiqueta verdadera
                is_correct = False
                if true_pdg is not None:
                    true_label_num = None
                    if true_pdg == 11: true_label_num = 0    # Electrón
                    elif true_pdg == -11: true_label_num = 1  # Positrón
                    elif true_pdg == 22: true_label_num = 2   # Gamma
                    
                    is_correct = (max_prob_idx == true_label_num) if true_label_num is not None else False
                    # La variable is_correct se mantiene en el diccionario de resultados
                
                # Crear diccionario de probabilidades
                prob_dict = {p: prob for p, prob in zip(LABEL_TO_PARTICLE.values(), event_probs)}
                
                # Almacenar resultados
                results.append({
                    'event_idx': event_idx,  # Índice después del filtrado
                    'original_id': original_id,  # ID original del evento
                    'predicted': predicted_particle,
                    'confidence': confidence,
                    'true': true_particle,
                    'true_pdg': true_pdg,
                    'is_correct': is_correct,
                    'probabilities': prob_dict
                })
        
            # -------------------------------------------------------------------
            # 6.7.2 PRESENTACIÓN DE RESULTADOS
            # -------------------------------------------------------------------
            print("\n" + "="*80)
            print("📊 RESUMEN DE RESULTADOS".center(80))
            print("="*80)
            
            # Configuración de paginación
            events_per_page = 5
            current_page = 0
            total_events = len(results)
            total_pages = (total_events + events_per_page - 1) // events_per_page
            
            # Mostrar resumen de la primera página
            print(f"\n📄 Mostrando página 1 de {total_pages} "
                f"(eventos 1 a {min(events_per_page, total_events)} de {total_events})")
            
            # Mostrar resultados por páginas
            while current_page < total_pages:
                # Calcular índices de la página actual
                start_idx = current_page * events_per_page
                end_idx = min(start_idx + events_per_page, total_events)
                current_results = results[start_idx:end_idx]
                
                # Mostrar encabezado de página mejorado
                page_info = f"PÁGINA {current_page + 1}/{total_pages} - EVENTOS {start_idx + 1}-{end_idx} de {total_events}"
                print("\n" + "═" * 80)
                print(f"📊 {page_info:^76} 📊")
                print("")
                print("")
                print("")
                print("")
                print("")
                print("")
                
                # Mostrar eventos de la página actual en formato compacto
                for result in current_results:
                    # Encabezado del evento con información básica
                    status = f"✅" if result['is_correct'] else ("❌" if result['true_pdg'] is not None else "")
                    print(f"\n╔{'═'*78}╗")
                    print(f"║ 📌 EVENTO #{result['event_idx']:<3} (ID original: {result['original_id']:<5}) | {result['predicted']:<15} | "
                        f"Conf: {result['confidence']*100:5.1f}% {status:>3} ║")
                    
                    # Información detallada en una sola línea
                    if result['true_pdg'] is not None:
                        print(f"║ {'Real: ' + result['true'] + ' (PDG:' + str(result['true_pdg']) + ')':<76} ║")
                    
                    # Barras de probabilidad compactas
                    print(f"╠{'═'*78}╣")
                    probs = [(p, prob) for p, prob in result['probabilities'].items()]
                    print("║ ", end="")
                    for i, (particle, prob) in enumerate(probs):
                        bar = '█' * int(round(prob * 10)) + ' ' * (10 - int(round(prob * 10)))
                        print(f"{particle[0]}:{prob*100:3.0f}%|{bar}| ", end="" if i < len(probs)-1 else "║\n")
                    print(f"╚{'═'*78}╝")
                
                # Opciones de navegación
                print("\n" + "="*80)
                print("OPCIONES:".center(80))
                print("  • Presiona Enter para ver la siguiente página")
                print("  • 'a' + Enter para ver la página anterior" if current_page > 0 else "")
                print("  • 's' + Enter para salir")
                print(f"  • Número de evento (1-{total_events}) para ver detalles")
                print("="*80)
                
                user_input = input("\n¿Qué deseas hacer? ").strip().lower()
                
                # Procesar entrada del usuario
                if user_input == 's':
                    break
                elif user_input == 'a' and current_page > 0:
                    current_page -= 2  # Se incrementará 1 después
                elif user_input.isdigit():
                    event_num = int(user_input) - 1
                    if 0 <= event_num < total_events:
                        # Mostrar evento específico
                        result = results[event_num]
                        clear_output(wait=True)
                        print("\n" + "="*80)
                        print(f"📋 DETALLES DEL EVENTO #{result['event_idx']} (ID original: {result['original_id']})".center(80))
                        print("="*80)
                        
                        # Mostrar información detallada
                        print(f"\n🔍 CLASIFICACIÓN:")
                        print(f"  {'Predicción:':<18} {result['predicted']}")
                        print(f"  {'Confianza:':<18} {result['confidence']*100:.1f}%")
                        
                        if result['true_pdg'] is not None:
                            print(f"\n✅ VERIFICACIÓN:")
                            print(f"  {'Partícula real:':<18} {result['true']}")
                            print(f"  {'PDG:':<18} {result['true_pdg']}")
                            status = "✅ CORRECTO" if result['is_correct'] else "❌ INCORRECTO"
                            print(f"  {'Estado:':<18} {status}")
                        
                        print("\n📊 DISTRIBUCIÓN DE PROBABILIDADES:")
                        for p, prob in result['probabilities'].items():
                            bar_len = int(prob * 20)
                            print(f"  {p + ':':<12} {prob*100:5.1f}% |{'█'*bar_len}{'░'*(20-bar_len)}|")
                        
                        # Mostrar histogramas si están disponibles
                        if show_histograms and histo_dir is not None:
                            print("\n📊 VISUALIZACIÓN DE HISTOGRAMAS:")
                            try:
                                # Usar el ID original para cargar los histogramas
                                original_id = result['original_id']
                                event_dir = histo_dir.get(f"event_{original_id}")
                                
                                if event_dir:
                                    # Obtener histogramas desde el directorio del evento
                                    hist_zp = event_dir.get("hr_zp")
                                    hist_zm = event_dir.get("hr_zm")
                                else:
                                    # Mantener compatibilidad con el formato antiguo
                                    hist_zp = histo_dir.get(f"hr_zp_evt{original_id}")
                                    hist_zm = histo_dir.get(f"hr_zm_evt{original_id}")
                                
                                if hist_zp is not None and hist_zm is not None:
                                    # Mostrar los histogramas con el original_id
                                    display_histograms(
                                        original_id,
                                        hist_zp,
                                        hist_zm,
                                        result['predicted'],
                                        result['true'] if result['true_pdg'] is not None else None,
                                        result['confidence']
                                    )
                                else:
                                    print("  ⚠️  No se encontraron los histogramas para este evento")
                            except Exception as e:
                                print(f"  ⚠️  Error al cargar los histogramas: {str(e)}")
                        
                        input("\n⏎ Presiona Enter para continuar...")
                        continue
                
                # Avanzar a la siguiente página
                current_page += 1
            
            # -------------------------------------------------------------------
            # 6.9. FINALIZACIÓN
            # -------------------------------------------------------------------
            print("\n" + "="*80)
            print("ANÁLISIS COMPLETADO EXITOSAMENTE".center(80))
            print("="*80)
            print(f"\n✓ Procesamiento finalizado a las: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*80 + "\n")
        
    except Exception as e:
        print(f"Ocurrió un error inesperado al procesar el archivo ROOT: {e}")
    finally:
        # Asegurarse de cerrar el archivo ROOT
        if file is not None:
            file.close()

if __name__ == "__main__":
    # Configuración del análisis
    CONFIG = {
        'show_histograms': True,  # Mostrar histogramas interactivos
        'max_events': None,       # Número máximo de eventos a procesar (None = todos)
        'prob_threshold': 0.6     # Umbral de probabilidad (0.0 = desactivado, 1.0 = máximo)
    }
    
    # Validar umbral
    if not (0.0 <= CONFIG['prob_threshold'] < 1.0):
        print("Error: El umbral de probabilidad debe estar entre 0.0 y 0.99")
        sys.exit(1)
    
    # Verificar si los archivos de modelo y scaler existen
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Archivo de modelo '{MODEL_PATH}' no encontrado. Asegúrate de que el modelo esté entrenado y guardado.")
    elif not os.path.exists(SCALER_PATH):
        print(f"Error: Archivo de scaler '{SCALER_PATH}' no encontrado. Asegúrate de que el scaler se haya guardado durante el entrenamiento.")
    elif not os.path.exists(INPUT_ROOT_FILE):
        print(f"Error: Archivo de entrada '{INPUT_ROOT_FILE}' no encontrado. Verifica la ruta y el nombre del archivo.")
    else:
        # Llamar a la función con la configuración
        classify_events(
            model_path=MODEL_PATH,
            scaler_path=SCALER_PATH,
            input_file=INPUT_ROOT_FILE,
            tree_name=TREE_NAME,
            feature_columns=FEATURE_COLUMNS,
            max_events=CONFIG['max_events'],
            show_histograms_flag=CONFIG['show_histograms'],
            prob_threshold=CONFIG['prob_threshold']
        )