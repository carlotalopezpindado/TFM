from factscore.factscorer import FactScorer
import json
import time
import os
import sqlite3
import numpy as np
import pickle as pkl
from rank_bm25 import BM25Okapi

SPECIAL_SEPARATOR = "####SPECIAL####SEPARATOR####"
MAX_LENGTH = 256

class DocDB(object):
    """Sqlite backed document storage.

    Implements get_doc_text(doc_id).
    """

    def __init__(self, db_path=None, data_path=None):
        self.db_path = db_path
        self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self.bm25 = None

        cursor = self.connection.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        
        if len(cursor.fetchall())==0:
            assert data_path is not None, f"{self.db_path} is empty. Specify `data_path` in order to create a DB."
            print (f"{self.db_path} is empty. start building DB from {data_path}...")
            self.build_db(self.db_path, data_path)
        
        # Load documents into BM25 index
        self.load_bm25_index()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def path(self):
        """Return the path to the file that backs this database."""
        return self.path

    def close(self):
        """Close the connection to the database."""
        self.connection.close()

    def build_db(self, db_path, data_path):
        from transformers import RobertaTokenizer
        tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
        
        titles = set()
        output_lines = []
        tot = 0
        start_time = time.time()
        c = self.connection.cursor()
        c.execute("CREATE TABLE documents (title PRIMARY KEY, text);")

        with open(data_path, "r") as f:
            for line in f:
                dp = json.loads(line)
                title = dp["title"]
                text = dp["text"]
                if title in titles:
                    continue
                titles.add(title)
                if type(text)==str:
                    text = [text]
                passages = [[]]
                for sent_idx, sent in enumerate(text):
                    assert len(sent.strip())>0
                    tokens = tokenizer(sent)["input_ids"]
                    max_length = MAX_LENGTH - len(passages[-1])
                    if len(tokens) <= max_length:
                        passages[-1].extend(tokens)
                    else:
                        passages[-1].extend(tokens[:max_length])
                        offset = max_length
                        while offset < len(tokens):
                            passages.append(tokens[offset:offset+MAX_LENGTH])
                            offset += MAX_LENGTH
                
                psgs = [tokenizer.decode(tokens) for tokens in passages if np.sum([t not in [0, 2] for t in tokens])>0]
                text = SPECIAL_SEPARATOR.join(psgs)
                output_lines.append((title, text))
                tot += 1

                if len(output_lines) == 1000000:
                    c.executemany("INSERT INTO documents VALUES (?,?)", output_lines)
                    output_lines = []
                    print ("Finish saving %dM documents (%dmin)" % (tot / 1000000, (time.time()-start_time)/60))

        if len(output_lines) > 0:
            c.executemany("INSERT INTO documents VALUES (?,?)", output_lines)
            print ("Finish saving %dM documents (%dmin)" % (tot / 1000000, (time.time()-start_time)/60))

        self.connection.commit()

    def load_bm25_index(self):
        cursor = self.connection.cursor()
        cursor.execute("SELECT text FROM documents")
        documents = cursor.fetchall()
        documents = [doc[0].replace(SPECIAL_SEPARATOR, " ") for doc in documents]
        self.bm25 = BM25Okapi([doc.split() for doc in documents])

    def get_text_from_title(self, title):
        """Fetch the raw text of the doc for 'doc_id'."""
        cursor = self.connection.cursor()
        cursor.execute("SELECT text FROM documents WHERE title = ?", (title,))
        results = cursor.fetchall()
        results = [r for r in results]
        cursor.close()
        assert results is not None and len(results)==1, f"`topic` in your data ({title}) is likely to be not a valid title in the DB."
        results = [{"title": title, "text": para} for para in results[0][0].split(SPECIAL_SEPARATOR)]
        assert len(results)>0, f"`topic` in your data ({title}) is likely to be not a valid title in the DB."
        return results

    def get_passages_from_query(self, query, k=5):
        """Fetch the top k passages for a given query using BM25."""
        query_tokens = query.split()
        scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[-k:][::-1]
        cursor = self.connection.cursor()
        cursor.execute("SELECT text FROM documents")
        documents = cursor.fetchall()
        results = [{"text": documents[i][0]} for i in top_indices]
        return results

class Retrieval(object):

    def __init__(self, db, cache_path, embed_cache_path,
                 retrieval_type="gtr-t5-large", batch_size=None):
        self.db = db
        self.cache_path = cache_path
        self.embed_cache_path = embed_cache_path
        self.retrieval_type = retrieval_type
        self.batch_size = batch_size
        assert retrieval_type=="bm25" or retrieval_type.startswith("gtr-")
        
        self.encoder = None
        self.load_cache()
        self.add_n = 0
        self.add_n_embed = 0

    def load_encoder(self):
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer("sentence-transformers/" + self.retrieval_type)
        encoder = encoder.cuda()
        encoder = encoder.eval()
        self.encoder = encoder
        assert self.batch_size is not None
    
    def load_cache(self):
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "r") as f:
                self.cache = json.load(f)
        else:
            self.cache = {}
        if os.path.exists(self.embed_cache_path):
            with open(self.embed_cache_path, "rb") as f:
                self.embed_cache = pkl.load(f)
        else:
            self.embed_cache = {}
    
    def save_cache(self):
        if self.add_n > 0:
            if os.path.exists(self.cache_path):
                with open(self.cache_path, "r") as f:
                    new_cache = json.load(f)
                self.cache.update(new_cache)
            
            with open(self.cache_path, "w") as f:
                json.dump(self.cache, f)
        
        if self.add_n_embed > 0:
            if os.path.exists(self.embed_cache_path):
                with open(self.embed_cache_path, "rb") as f:
                    new_cache = pkl.load(f)
                self.embed_cache.update(new_cache)
            
            with open(self.embed_cache_path, "wb") as f:
                pkl.dump(self.embed_cache, f)

    def get_bm25_passages(self, query, k):
        return self.db.get_passages_from_query(query, k)

    def get_passages(self, topic, question, k):
        retrieval_query = topic + " " + question.strip()
        cache_key = retrieval_query
        
        if cache_key not in self.cache:
            if self.retrieval_type=="bm25":
                self.cache[cache_key] = self.get_bm25_passages(retrieval_query, k)
            else:
                # No haremos cambios aquí, pero podrías implementar lógica similar para otros tipos de recuperación
                raise NotImplementedError("Solo BM25 está implementado para esta versión")
            assert len(self.cache[cache_key]) in [k, len(self.cache[cache_key])]
            self.add_n += 1
        
        return self.cache[cache_key]

# Configuración del FactScorer con la clave de API de OpenAI
fs = FactScorer(openai_key="openai.txt")

# Crear instancia de DocDB
db = DocDB(db_path="evaluation/boe_edu_factscore.db", data_path="evaluation/boe_edu.jsonl")

# Crear instancia de Retrieval
retrieval = Retrieval(db, cache_path="evaluation/cache.json", embed_cache_path="evaluation/embed_cache.pkl", retrieval_type="bm25")

topics = ["Requisitos para modificaciones justificadas en proyectos de ayuda", 
          "Disposiciones del Reglamento Delegado (UE) 2015/98",
          "Conclusiones del Consejo Europeo de Barcelona 2002 sobre energía y transporte"]

generations = ["De acuerdo con la información proporcionada en los documentos del BOE  cuando se presentan circunstancias concretas que alteran las condiciones técnicas o económicas tenidas en cuenta para la concesión de la ayuda  se pueden solicitar modificaciones en la resolución de concesión. Los requisitos para estas solicitudes de modificaciones justificadas son:  1. La modificación no debe afectar a los objetivos perseguidos con la ayuda  incluidos los de finalidad regional  a sus aspectos fundamentales  a la determinación del beneficiario  ni dañar derechos de terceros. 2. La solicitud de modificación debe realizarse al menos un mes antes de que finalice el plazo de ejecución del proyecto y ser aceptada expresamente. 3. Las modificaciones deben obedecer a causas sobrevenidas que no pudieron preverse en el momento de la solicitud. 4. La solicitud debe ir acompañada de una",
               "Lo siento  no tengo información específica sobre el Reglamento Delegado (UE) 2015/98 en los documentos disponibles del BOE. Sin embargo  basándome en los documentos que he revisado y en la información general sobre los reglamentos de la Unión Europea  puedo decir que un reglamento delegado generalmente incluye las siguientes disposiciones:  1. En su encabezamiento  el título “Reglamento Delegado”  un número de orden  la fecha de su adopción y la indicación de su objeto. 2. La fórmula “Por el [órgano responsable]”. 3. La indicación de las disposiciones que constituyen la base para la adopción del reglamento  precedidas de la palabra “Visto”. 4. Los artículos que contienen las disposiciones específicas del reglamento  organizadas en capítulos y secciones. 5. El último artículo fija la fecha de entrada en vigor  que puede ser anterior o",
               "Lo siento  no puedo encontrar información específica sobre las conclusiones del Consejo Europeo de Barcelona de marzo de 2002 relacionadas con las redes de energía y transporte en los documentos del BOE que me ha proporcionado. Sin embargo  en la Decisión de la Comisión  de 11 de julio de 2001  por la que se crea un Comité consultivo llamado \"Foro europeo de la energía y los transportes\" (disponible en el BOE)  se menciona que las partes intercambiarán información relevante sobre proyectos de inversión en el sector energético  incluyendo la construcción y puesta al día de oleoductos  gasoductos y otros medios de transporte de productos energéticos. Además  se destaca la cooperación con el objetivo de aplicar eficazmente las disposiciones del título IV y el artículo 47  relacionadas con las inversiones en el sector de la energía.  También  en el artículo 55 de"]

# Calcular la puntuación utilizando la fuente de conocimiento personalizada del BOE
out = fs.get_score(topics, generations, gamma=10, knowledge_source="edu")
print("FActScore:", out["score"])
print("FActScore sin penalización por longitud:", out["init_score"])
print("% de respuestas no abstinentes:", out["respond_ratio"])
print("Promedio de hechos atómicos por respuesta:", out["num_facts_per_response"])
