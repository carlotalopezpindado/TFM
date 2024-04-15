from langdetect import detect

def validate_response(response):
    if detect(response) != 'es':
        # Aquí podrías implementar una traducción automática o re-prompt al modelo
        return translate_to_spanish(response)
    return response

# Función ficticia para ilustrar la traducción
def translate_to_spanish(english_text):
    # Implementa una traducción automática aquí -> copiar del doc de zeroshot
    return "Texto traducido al español."
