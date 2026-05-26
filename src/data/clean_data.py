import os
import sys
data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tools'))
sys.path.append(data_path)
import helpers

##TODO Poner esto en un archivo de configuracion 
current_dir = os.path.dirname(__file__)
carpeta_transcripciones=os.path.join(current_dir, '../../audio_files_processed/text_files1')
carpeta_destino=os.path.join(current_dir, '../../audio_files_processed/text_files1/transcripts')
carpeta_archivos_procesados=os.path.join(current_dir, '../../audio_files_processed/text_files1/procesados')
prompt_inicial = (
    "En adelante te enviaré transcripciones de conversaciones entre un asesor de un car dealer "
    "y un cliente que desea comprar un vehículo. Necesito que identifiques en las transcripciones "
    "al asesor y al cliente, y los etiquetes en cada conversación que le corresponde a cada uno. "
    "Adicionalmente, debes poner el token <soc> al comienzo de la etiqueta del asesor y del cliente. "
    "Asegúrate de revisar la coherencia en las conversaciones y usar buena ortografía. Además, en los casos "
    "en los que se esté hablando de un vehículo o modelo de vehículo, agrega la marca y modelo. Esta es la transcripción:"
)
gptize_switch=False


def gptize_transcripts(prompt,carpeta_origen,carpeta_destino, carpeta_archivos_procesados):
    for archivo in os.listdir(carpeta_origen):
        ruta_completa = os.path.join(carpeta_origen, archivo)
        
        # Verificar si es un archivo
        if os.path.isfile(ruta_completa):
            old_transcription=helpers.load_text_file(ruta_completa)
            new_transcription=helpers.GPTorganizar_transcripcion(prompt,old_transcription)
            carpeta_destino_transcripcion=os.path.join(carpeta_destino, archivo)
            helpers.save_string_to_file(carpeta_destino_transcripcion,new_transcription)
            helpers.move_file(ruta_completa, carpeta_archivos_procesados)



def main():
    gptize_transcripts(prompt_inicial,carpeta_transcripciones,carpeta_destino,carpeta_archivos_procesados) if gptize_switch==True else None
    


if __name__ == "__main__":
    main()