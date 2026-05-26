from  audio_processing.plain_text import PlainText
import os
import shutil

answer_machine_sentence=["Please leave your message","Por favor deje su mensaje","Por favor, deje su mensaje"]
text_folder_path = "audio_files_processed\\text_files1"
answer_machine_files_path="answer_machine_files"

for index,filename in enumerate(os.listdir(text_folder_path)):
    text_file_path = os.path.join(text_folder_path, filename)
    call_text=PlainText(file_path=text_file_path)
    #print(call_text())
    if(call_text.sentence_lookup(answer_machine_sentence) is True):
        shutil.move(text_file_path, answer_machine_files_path)