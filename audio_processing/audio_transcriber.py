import os
import assemblyai as aai
import shutil

aai.settings.api_key = os.environ["ASSEMBLYAI_API_KEY"]

# Specify the path to your folder
audio_folder_path = "audio_files"
text_folder_path = "text_files"
audio_processed_path = "audio_files_processed"

config = aai.TranscriptionConfig(speaker_labels=True, language_code='es')

transcriber = aai.Transcriber()

# Iterate over all files in the folder
for index,filename in enumerate(os.listdir(audio_folder_path)):
    audio_file_path = os.path.join(audio_folder_path, filename)
    text_file_path=f"{text_folder_path}/audio #{index}.txt"
    try:
        transcript = transcriber.transcribe( 
        audio_file_path, 
        config=config 
        )
        if transcript.audio_duration>30:
            with open(text_file_path, 'w') as file:
                for utterance in transcript.utterances:
                    file.write(f"Speaker {utterance.speaker}: {utterance.text}\n")
                shutil.move(audio_file_path, audio_processed_path)
    except:
        print(f'Fallo el archivo {audio_file_path}')
        print(transcript.error)