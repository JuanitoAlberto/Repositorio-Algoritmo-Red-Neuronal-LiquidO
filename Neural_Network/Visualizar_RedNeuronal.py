from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
import netron
import os

# Ruta al modelo original guardado
ruta_modelo_keras = "particle_classifier_model.keras"

# Cargar el modelo ya entrenado
if not os.path.exists(ruta_modelo_keras):
    raise FileNotFoundError(f"No se encontró el modelo: {ruta_modelo_keras}")

modelo = load_model(ruta_modelo_keras)

# 1️⃣ Visualización clásica con plot_model
plot_model(modelo,
           to_file="modelo_red_neuronal.png",
           show_shapes=True,
           show_layer_names=True,
           dpi=100)
print("✅ Imagen .png generada con plot_model (modelo_red_neuronal.png)")

# 2️⃣ Guardar en formato .h5 para usar con Netron
modelo.save("modelo_red_neuronal_netron.h5")

# Visualización interactiva con Netron
netron.start("modelo_red_neuronal_netron.h5")
