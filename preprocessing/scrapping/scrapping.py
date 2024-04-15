import pandas as pd
import os
from bs4 import BeautifulSoup, NavigableString, Comment
import re
import glob  

import configparser
config = configparser.ConfigParser()
config.read('config.ini')

data_path = config['scrapping']['data_path']

def open_html(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    return html_content

def erase_tags(soup):
    try:
        for script in soup(['footer','form','script','style','nav','img','i']):
            if script is not None:
                script.decompose()
        clean_text=soup
        return clean_text
    
    except Exception:
        return soup

def erase_comments(soup):
    for element in soup(text=lambda text: isinstance(text, Comment)):
        element.extract()
    return soup

def remove_isolated_links(soup):
    for a_tag in soup.findAll('a'):
        previous_sibling = a_tag.find_previous_sibling(string=True)
        next_sibling = a_tag.find_next_sibling(string=True)

        if not (previous_sibling and next_sibling):
            a_tag.replace_with(NavigableString(a_tag.text))
        else:
            a_tag.decompose()
            
    return soup

def href_for_text(soup):
    a_tags = soup.find_all('a')
    for tag in a_tags:
        if '#0' in tag.get('href', ''):
            tag.decompose()
        else:
            tag.replace_with(tag.text)
    return soup

def remove_specific_classes(soup, classes_to_remove):
    for class_ in classes_to_remove:
        for tag in soup.find_all(True, {'class': class_}):
            tag.decompose()
    return soup

def remove_empty_structures(soup,parametros):
    for param in parametros:
        div_tags = soup.find_all(param)
        for tag in div_tags:
            if not tag.get_text(strip=True):
                tag.decompose()
    return soup

def clear_blankspaces_after_href(soup):
    rgx=r'\r\n\s*\.'
    new_str_soup=re.sub(rgx,'',str(soup))
    soup= BeautifulSoup(new_str_soup, 'html.parser')
    return soup

def erase_empty_lines(clean_text):
    lines = (line.strip() for line in clean_text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    clean_text = '\n'.join(chunk for chunk in chunks if chunk)
    return clean_text

def html_to_text(soup):
    texto=soup.get_text()
    texto_normalizado=erase_empty_lines(texto)
    return texto_normalizado

def main_html_scrapping(soup):  
    content_l = erase_tags(soup)
    content_l = remove_isolated_links(content_l)
    content_sin_comments = erase_comments(content_l)
    content_l = href_for_text(content_sin_comments)
    content_l = remove_specific_classes(content_l, ['idioma'])
    content_l = remove_empty_structures(content_l, ['span','p','div'])
    soup = clear_blankspaces_after_href(content_l)
    
    text_to_remove = "Agencia Estatal Boletín Oficial del Estado\nIr a contenido\nConsultar el diario oficial BOE\nPuede seleccionar otro idioma:\nCastellanoes\nesCastellano\ncaCatalà\nglGalego\neuEuskara\nvaValencià\nenEnglish\nfrFrançais\nMenú\nMenú\nDiarios Oficiales\nBOE\nBORME\nOtros diarios oficiales\nInformación Jurídica\nTodo el Derecho\nBiblioteca Jurídica Digital\nOtros servicios\nNotificaciones\nEdictos judiciales\nPortal de subastas\nAnunciantes\nEstá Vd. en\nInicio\nBuscar Documento "
    text_to_remove2 = "subir\nContactar\nSobre esta sede electrónica\nMapa\nAviso legal\nAccesibilidad\nProtección de datos\nSistema Interno de Información\nTutoriales\nAgencia Estatal Boletín Oficial del Estado\nAvda. de Manoteras, 54 - 28050 Madrid"
    
    texto = html_to_text(soup)   
    texto = texto.replace(text_to_remove, '')
    texto = texto.replace(text_to_remove2, '')

    return texto

def process_html_content(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    texto = main_html_scrapping(soup)  
    return texto

def process_html_from_directory(directory_path):
    processed_texts = []
    i = 0
    for file_path in glob.glob(os.path.join(directory_path, '*.html')):
        i = i + 1
        print(i)
        html_content = open_html(file_path)  
        processed_text = process_html_content(html_content)
        if processed_text != '': processed_texts.append(processed_text)

    return processed_texts

def save_to_parquet(processed_texts, output_file_name):
    df = pd.DataFrame(processed_texts, columns=['text'])
    df.to_parquet(output_file_name, index=False)

def main(directory_path, output_file_name):
    processed_texts = process_html_from_directory(directory_path)
    save_to_parquet(processed_texts, output_file_name)

main('data', 'clean_data.parquet')
