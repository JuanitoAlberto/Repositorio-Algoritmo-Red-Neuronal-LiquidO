# PASO 1: Importar las herramientas necesarias
import pandas as pd        # Para manejar tablas de datos (DataFrames)
import uproot              # Para leer archivos .root de forma sencilla
import numpy as np         # Para operaciones numéricas
from sklearn.model_selection import train_test_split # Para dividir los datos
from sklearn.preprocessing import StandardScaler     # Para escalar los datos
from tensorflow.keras.utils import to_categorical    # Para codificar etiquetas
from tensorflow.keras.models import Sequential       # Para construir la red neuronal
from tensorflow.keras.layers import Dense, Dropout   # Capas de la red (Dense) y regularización (Dropout)
from tensorflow.keras.callbacks import EarlyStopping # Para detener el entrenamiento si no mejora
import matplotlib.pyplot as plt                      # Para graficar resultados
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# PASO 2: Definir dónde están tus archivos con las CARACTERÍSTICAS ya calculadas
# ¡¡¡IMPORTANTE!!! Asegúrate de que estas rutas sean correctas y que los archivos existan.
# Las etiquetas (0, 1, 2) se asignarán basadas en el archivo del que provienen.
file_paths_and_labels = {
    0: "extracted_features_electrons.root", # Etiqueta 0 para electrones
    1: "extracted_features_positrons.root", # Etiqueta 1 para positrones
    2: "extracted_features_gammas.root"     # Etiqueta 2 para gammas
}
# Si tus archivos están en otra carpeta, ajusta la ruta, por ejemplo:
# 0: "ruta/a/tus/datos/features_electrons_nn.root",

# Usamos 'feature_tree' que es el nombre del árbol que creamos en extract_features_completo
tree_name_in_files = "feature_tree"

# Lista de las columnas (features) que queremos usar de esos archivos.
# Estas deben ser los nombres EXACTOS de las ramas en tu FeatureTree.
# La columna 'mcpdg' también se leerá (si la guardaste con ese nombre) y se usará para verificar la etiqueta.
# pero no se usará como feature de entrada a la red si la excluimos de esta lista.
# Lista de características a utilizar (todas las del feature_tree excepto mcpdg)
feature_columns_to_use = [
    "Ec1",
    "Ec2",
    "Qmx",
    "QmxP",
    "QmxM",
    "tminP",
    "tminM",
    "dp",
    "Qmx2_val",
    "Qtot_p",
    "Qtot_m",
    "nhits_fired_sipm"
    # mcpdg se usa solo como etiqueta
]

# PASO 3: Cargar los datos en una tabla (DataFrame de Pandas)
all_data_list = []
for assigned_label, file_path in file_paths_and_labels.items():
    try:
        # Uproot puede leer directamente las ramas que necesitas más 'mcpdg' para la etiqueta
        branches_to_read = feature_columns_to_use + ["mcpdg"] # Leer 'mcpdg' para verificar
        with uproot.open(f"{file_path}:{tree_name_in_files}") as tree:
            df_particle = tree.arrays(branches_to_read, library="pd")
            
            # Verificación y asignación de etiqueta
            # Mapear PDG ID a etiquetas: electrón (11) -> 0, positrón (-11) -> 1, gamma (22) -> 2
            def map_pdg_to_label(pdg_code):
                if pdg_code == 11: return 0    # Electrón
                if pdg_code == -11: return 1    # Positrón
                if pdg_code == 22: return 2     # Gamma
                return -1  # Descartar PDGs inesperados

            df_particle['label_from_pdg'] = df_particle['mcpdg'].apply(map_pdg_to_label)
            
            # Filtrar eventos que no coincidan con la etiqueta esperada del archivo (opcional, pero buena práctica)
            # O si el PDG es inesperado
            df_particle = df_particle[df_particle['label_from_pdg'] == assigned_label]
            
            if df_particle.empty:
                print(f"Advertencia: No se encontraron eventos con mcpdg esperado para {file_path} (etiqueta {assigned_label}).")
                continue

            df_particle['label'] = assigned_label # Usar la etiqueta asignada basada en el archivo
            
            # Seleccionar solo las columnas de features y la etiqueta final
            columns_for_final_df = feature_columns_to_use + ['label']
            all_data_list.append(df_particle[columns_for_final_df])
            print(f"Cargado {file_path} para etiqueta {assigned_label}, {len(df_particle)} eventos válidos.")

    except Exception as e:
        print(f"Error al cargar o procesar {file_path}: {e}.")
        print("Asegúrate de que el archivo exista, la ruta sea correcta, y el TTree/ramas también.")

if not all_data_list:
    print("\n¡No se cargaron datos! Verifica las rutas de los archivos y si los archivos existen y contienen datos.")
    exit()

# Combinar los datos de todos los archivos en una sola tabla
data = pd.concat(all_data_list, ignore_index=True)

# Limpieza básica: eliminar filas donde alguna feature sea NaN (Not a Number) o infinita
data.replace([np.inf, -np.inf], np.nan, inplace=True) # Reemplazar infinitos con NaN
data.dropna(inplace=True)                              # Eliminar filas con NaN

print(f"\nTotal de eventos cargados y combinados después de limpieza: {len(data)}")
if data.empty:
    print("El DataFrame está vacío después de combinar o limpiar. Revisa los datos de origen o los filtros.")
    exit()

print("\nPrimeras filas de la tabla de datos:")
print(data.head())
print("\nDistribución de etiquetas (0=e-, 1=e+, 2=gamma):")
print(data['label'].value_counts())

# PASO 4: Separar las CARACTERÍSTICAS (X) de las ETIQUETAS (y)
X = data[feature_columns_to_use]  # Todas las columnas de features
y = data['label']                 # Solo la columna 'label'

# PASO 5: Dividir los datos en entrenamiento y prueba
# test_size=0.2 significa que el 20% de los datos serán para prueba, 80% para entrenamiento
# random_state es para que la división sea siempre la misma (reproducibilidad)
# stratify=y intenta que la proporción de cada clase sea similar en train y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nEventos para entrenamiento: {len(X_train)}")
print(f"Eventos para prueba: {len(X_test)}")

# PASO 6: Escalar las características
# Esto ayuda a que la red neuronal aprenda mejor
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) # Aprende la media/std de los datos de ENTRENAMIENTO y los transforma
X_test_scaled = scaler.transform(X_test)     # USA la media/std aprendida para transformar los datos de PRUEBA

# PASO 7: Codificar las etiquetas a formato "one-hot"
# Ejemplo: si y_train es [0, 1, 2, 0], y_train_one_hot será:
# [[1,0,0], [0,1,0], [0,0,1], [1,0,0]]
num_classes = 3 # Tenemos 3 tipos de partículas (electrón, positrón, gamma)
y_train_one_hot = to_categorical(y_train, num_classes=num_classes)
y_test_one_hot = to_categorical(y_test, num_classes=num_classes)

print("\nEjemplo de una etiqueta original y su versión one-hot:")
if len(y_train) > 0:
    print(f"Original: {y_train.iloc[0]}, One-hot: {y_train_one_hot[0]}")
else:
    print("No hay datos de entrenamiento para mostrar ejemplo de etiqueta.")

# --- HASTA AQUÍ HEMOS PREPARADO LOS DATOS ---
# --- AHORA VAMOS A CONSTRUIR Y ENTRENAR LA IA ---

# PASO 8: Construir el modelo de Red Neuronal
model = Sequential()

# Capa de entrada y primera capa oculta:
model.add(Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],))) # Más neuronas
model.add(Dropout(0.3)) # Dropout para regularización

# Segunda capa oculta
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3)) # Dropout

# Tercera capa oculta (opcional, para mayor profundidad)
model.add(Dense(32, activation='relu'))

# Capa de salida:
model.add(Dense(num_classes, activation='softmax')) # Softmax para clasificación multiclase

# PASO 9: Compilar el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Mostrar un resumen del modelo
print("\nResumen del modelo de Red Neuronal:")
model.summary()

# PASO 10: Entrenar el modelo
print("\n--- Empezando el entrenamiento de la Red Neuronal ---")
# EarlyStopping: detiene el entrenamiento si la 'val_loss' no mejora después de 'patience' epochs
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train_scaled,
    y_train_one_hot,
    epochs=50,             # Puedes empezar con 50-100 epochs
    batch_size=64,         # Un batch size común
    validation_split=0.2,  # Usar el 20% de los datos de entrenamiento para validación durante el entrenamiento
    callbacks=[early_stopping],
    verbose=1
)
print("--- Entrenamiento completado ---")

# PASO 11: Evaluar el modelo con los datos de PRUEBA
loss, accuracy = model.evaluate(X_test_scaled, y_test_one_hot, verbose=0)
print(f"\nResultados en el conjunto de PRUEBA:")
print(f"  Pérdida (Loss): {loss:.4f}")
print(f"  Precisión (Accuracy): {accuracy:.4f} (esto es {accuracy*100:.2f}%)")

# PASO 11.5: Generar y mostrar la matriz de confusión
print("\nGenerando matriz de confusión...")
y_pred = model.predict(X_test_scaled)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test_one_hot, axis=1)

# Calcular la matriz de confusión
cm = confusion_matrix(y_true, y_pred_classes)

# Crear un mapa de calor de la matriz de confusión
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Electrón', 'Positrón', 'Gamma'],
            yticklabels=['Electrón', 'Positrón', 'Gamma'])
plt.title('Matriz de Confusión')
plt.xlabel('Predicción')
plt.ylabel('Valor Real')
plt.tight_layout()
plt.savefig('matriz_confusion_NN.png')
print("Matriz de confusión guardada como 'matriz_confusion_NN.png'")
plt.show()

# Calcular métricas adicionales
total_correct = np.sum(cm.diagonal())
total_samples = np.sum(cm)
accuracy = total_correct / total_samples
print(f"\nPrecisión global: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Imprimir métricas por clase
print("\nPrecisión por clase:")
for i, particle in enumerate(['Electrón', 'Positrón', 'Gamma']):
    correct = cm[i, i]
    total = np.sum(cm[i, :])
    if total > 0:
        class_acc = correct / total
        print(f"{particle}: {class_acc:.4f} ({class_acc*100:.2f}%)")

# PASO 12: Graficar historial de entrenamiento (Accuracy y Loss)
if history is not None:
    
    # Crear figura para los gráficos de entrenamiento
    plt.figure(figsize=(12, 4))
    
    # Gráfica de Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Precisión (entrenamiento)')
    plt.plot(history.history['val_accuracy'], label='Precisión (validación)')
    plt.title('Precisión del Modelo')
    plt.xlabel('Epoch')
    plt.ylabel('Precisión')
    plt.legend()

    # Gráfica de Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Pérdida (entrenamiento)')
    plt.plot(history.history['val_loss'], label='Pérdida (validación)')
    plt.title('Pérdida del Modelo')
    plt.xlabel('Epoch')
    plt.ylabel('Pérdida')
    plt.legend()

    plt.tight_layout()
    
    # Primero guardar el gráfico
    plt.savefig('training_history.png', bbox_inches='tight')
    print("Gráfico de entrenamiento guardado como 'training_history.png'")
    
    # Luego mostrarlo
    plt.show()
    
    # Cerrar la figura para liberar memoria
    plt.close()

# PASO 13: Hacer algunas predicciones (opcional, para ver cómo funciona)
if len(X_test_scaled) > 0:
    print("\nEjemplo de predicciones en algunos datos de prueba:")
    num_predictions_to_show = min(5, len(X_test_scaled)) # Mostrar hasta 5 o menos si hay pocos datos de prueba
    predictions_probabilities = model.predict(X_test_scaled[:num_predictions_to_show])
    predicted_classes = np.argmax(predictions_probabilities, axis=1)

    for i in range(num_predictions_to_show):
        print(f"  Muestra {i+1}:")
        # Formatear las probabilidades para que sean más legibles
        prob_str = ", ".join([f"{p:.3f}" for p in predictions_probabilities[i]])
        print(f"    Probabilidades (e-, e+, gamma): [{prob_str}]")
        print(f"    Clase predicha: {predicted_classes[i]} (0=e-, 1=e+, 2=gamma)")
        print(f"    Clase real:    {y_test.iloc[i]}")
        print("-" * 30)
else:
    print("\nNo hay datos de prueba para mostrar predicciones.")

# PASO 14: Guardar el modelo entrenado
try:
    model.save("particle_classifier_model.keras")
    print("\nModelo guardado exitosamente como particle_classifier_model.keras")
except Exception as e:
    print(f"\nError al guardar el modelo: {e}")

# También es buena idea guardar el 'scaler' que usaste para preprocesar los datos,
# ya que necesitarás aplicar EXACTAMENTE la misma transformación a cualquier dato nuevo
# antes de pasarlo al modelo cargado.
import joblib
try:
    joblib.dump(scaler, 'data_scaler.pkl')
    print("Scaler guardado exitosamente como data_scaler.pkl")
except Exception as e:
    print(f"\nError al guardar el scaler: {e}")
