
from openai import OpenAI
import os
import shutil
import json

OPENAI_KEY = os.environ["OPENAI_API_KEY"]

def GPTorganizar_transcripcion(prompt,transcripcion):
    
    # Configura tu clave API
    client = OpenAI(
        # This is the default and can be omitted
        api_key=OPENAI_KEY,
    )

    # Crear el prompt completo
    prompt_completo = f"{prompt} \n\n{transcripcion}"

    # Llamada a la API de OpenAI
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "assistant", "content": prompt},
                {"role": "user", "content": prompt_completo}
            ],
            max_tokens=1500,
            temperature=0.5,
            model="gpt-4o-mini",
        )

        # Obtenemos la respuesta generada
        respuesta = chat_completion.choices[0].message.content
        return respuesta

    except Exception as e:
        raise e 

def GPT4o_mini(prompt):
    
    # Configura tu clave API
    client = OpenAI(
        # This is the default and can be omitted
        api_key=OPENAI_KEY,
    )
    # Llamada a la API de OpenAI
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "assistant", "content": prompt}
            ],
            max_tokens=1500,
            temperature=0.5,
            model="gpt-4o-mini",
        )
        # Obtenemos la respuesta generada
        respuesta = chat_completion.choices[0].message.content
        return respuesta

    except Exception as e:
        raise e 

def GPT4o(prompt):
    
    # Configura tu clave API
    client = OpenAI(
        # This is the default and can be omitted
        api_key=OPENAI_KEY,
    )
    # Llamada a la API de OpenAI
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "assistant", "content": prompt}
            ],
            max_tokens=1500,
            temperature=0.5,
            model="gpt-4o",
        )
        # Obtenemos la respuesta generada
        respuesta = chat_completion.choices[0].message.content
        return respuesta

    except Exception as e:
        raise e 


def load_text_file(file_path, encoder='latin-1'):
    """
    Load text data from a directory containing multiple .txt files.

    Args:
        data_dir (str): Path to the directory containing .txt files.

    Returns:
        list: A list of strings, where each string contains the content of a text file.
    """
    try:
        with open(file_path, 'r', encoding=encoder) as file:
            text = file.read()
    except:
        with open(file_path, 'r', encoding='latin-1') as file:
            text = file.read()

    return text

def save_string_to_file(file_path, string):
    """
    Save a string into a file.
    
    Args:
    file_path (str): The path where the file will be saved.
    string (str): The data as a string.
    
    Returns:
    None
    """
    with open(file_path, 'w',encoding='latin-1') as file:
        file.write(string)

def move_file(source_path, destination_folder):
    """
    Moves a file from the source path to the destination folder.
    
    Args:
    source_path (str): The full path to the file you want to move.
    destination_folder (str): The folder where the file should be moved.
    
    Returns:
    None
    """
    # Check if the destination folder exists, if not, create it
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    # Move the file
    shutil.move(source_path, destination_folder)
    print(f"File moved to {destination_folder}")

def remove_empty_strings(string_list):
    return [s for s in string_list if s]

def save_to_json(data, filename='data.json'):
    """Save a list of dictionaries to a JSON file.
    
    Args:
        data (list): A list of dictionaries to save.
        filename (str): The name of the file to save the JSON data. Default is 'data.json'.
    """
    with open(filename, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file,ensure_ascii=False, indent=4)  # Use indent for pretty printing
