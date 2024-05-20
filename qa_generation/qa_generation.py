import os
import csv
import torch
import random
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_llm():
    try:
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
        model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
        #if torch.cuda.is_available():
        #    model = model.to("cuda")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading the model: {e}")
        exit(1)

def generate_questions(text, model, tokenizer, num_questions=10):
    prompt = f"""
    Eres un experto en crear preguntas basadas en materiales de codificación y documentación.
    Tu objetivo es preparar a un programador o desarrollador para sus exámenes y pruebas de codificación.
    Lo haces preguntando sobre el texto a continuación:
    ------------
    {text}
    ------------
    Crea {num_questions} preguntas que preparen a los programadores para sus pruebas.
    Asegúrate de no perder ninguna información importante.
    PREGUNTAS:"""

    inputs = tokenizer.encode(prompt, return_tensors="pt")
    device = "cpu"
    inputs = inputs.to(device)
    model = model.to(device)
    outputs = model.generate(inputs, max_new_tokens=1000)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text.split('\n')[:num_questions]

def select_random_files(directory, num_files=10):
    files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.txt')]
    if not files:
        print("No text files found in the directory.")
        exit(1)
    return random.sample(files, min(num_files, len(files)))

def get_csv(directory, model, tokenizer, num_files=10, num_questions=10):
    files = select_random_files(directory, num_files)
    output_file = "QA.csv"
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Archivo", "Pregunta"])  # Escribiendo la fila del encabezado
        for i, file_path in enumerate(files):
            print(f"Processing file {i+1}/{len(files)}: {os.path.basename(file_path)}")
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    text_content = file.read()
                questions = generate_questions(text_content, model, tokenizer, num_questions)
                for question in questions:
                    csv_writer.writerow([os.path.basename(file_path), question])
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
    return output_file

def main():
    directory = input("Introduce el directorio que contiene los archivos de texto: ")
    num_files = int(input("Introduce el número de archivos a procesar: "))
    num_questions = int(input("Introduce el número de preguntas a generar por archivo: "))
    model, tokenizer = load_llm()
    output_csv = get_csv(directory, model, tokenizer, num_files, num_questions)
    print("CSV generado en:", output_csv)

if __name__ == "__main__":
    main()
