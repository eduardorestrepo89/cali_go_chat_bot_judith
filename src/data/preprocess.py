import os
import sys
data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tools'))
sys.path.append(data_path)
import helpers

current_dir = os.path.dirname(__file__)
ruta_transcripciones=os.path.join(current_dir, '../../audio_files_processed/text_files1/transcripts')

def question_answer_prep_data(ruta_transcripciones):
    pregunta_respuesta_list=[]
    for archivo in os.listdir(ruta_transcripciones):
        ruta_completa = os.path.join(ruta_transcripciones, archivo)
        # Verificar si es un archivo
        if os.path.isfile(ruta_completa):
            transcripcion=helpers.load_text_file(ruta_completa,encoder='utf-8')
            transcript_list=transcripcion.split('<soc>')
            ##print(transcript_list)
            transcript_list=helpers.remove_empty_strings(transcript_list)
            cliente_line=''
            asesor_line=''
            prev_actor=''
            for line in transcript_list:
                if 'Cliente' in line:
                    #if prev is asesor entonces guardo el par pregunta respuesta  y reinicio cliente line y el asesor line
                    if prev_actor=='Asesor':
                        pregunta_respuesta_list.append({
                            "question": cliente_line,
                            "answer": asesor_line
                            })
                        cliente_line=''
                        asesor_line=''
                    
                    cliente_line=cliente_line+line.replace('Cliente:','')
                    #pongo que el prev es cliente  
                    prev_actor='Cliente' 
                else:
                    asesor_line=asesor_line+line.replace('Asesor:','')
                    #pongo el prev en asesor
                    prev_actor='Asesor'

            pregunta_respuesta_list.append({
                "question": cliente_line,
                "answer": asesor_line
                })
    helpers.save_to_json(pregunta_respuesta_list)


def main():
    question_answer_prep_data(ruta_transcripciones) 
    

if __name__ == "__main__":
    main()