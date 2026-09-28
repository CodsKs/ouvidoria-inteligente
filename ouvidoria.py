"""Funções compartilhadas pelos notebooks e pela interface."""
import json
from pathlib import Path
from functools import lru_cache
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

MODELOS = ['sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', 'sentence-transformers/distiluse-base-multilingual-cased-v2']
BASE = Path(__file__).parent

def carregar_base(caminho=None):
    caminho = Path(caminho) if caminho else BASE / 'dados' / 'manifestacoes.json'
    dados = json.loads(caminho.read_text(encoding='utf-8'))
    df = pd.DataFrame(dados)
    campos = {'id', 'data', 'categoria_oficial', 'texto'}
    if not campos.issubset(df.columns) or df.empty:
        raise ValueError('Base deve conter id, data, categoria_oficial e texto.')
    if df.id.duplicated().any() or not df.texto.map(lambda t: isinstance(t, str) and bool(t.strip())).all():
        raise ValueError('IDs devem ser únicos e textos não vazios.')
    return df

@lru_cache(maxsize=2)
def modelo(nome=MODELOS[0]):
    return SentenceTransformer(nome)

def codificar(textos, nome=MODELOS[0]):
    return modelo(nome).encode(list(textos), normalize_embeddings=True, show_progress_bar=False)

def detectar_duplicatas(textos, limiar=0.85, nome=MODELOS[0]):
    """Retorna (índice i, índice j, score), i < j, com cosseno > limiar."""
    if not -1 <= limiar <= 1:
        raise ValueError('O limiar deve estar entre -1 e 1.')
    if len(textos) < 2:
        return []
    vetores = codificar(textos, nome)
    similaridade = vetores @ vetores.T
    return [(i, j, float(similaridade[i, j])) for i in range(len(textos)) for j in range(i + 1, len(textos)) if similaridade[i, j] > limiar]

def dividir(texto, tamanho=250, overlap=50, estrategia='Recursivo'):
    if tamanho <= 0 or not 0 <= overlap < tamanho:
        raise ValueError('Use tamanho positivo e 0 <= overlap < tamanho.')
    classe = RecursiveCharacterTextSplitter if estrategia == 'Recursivo' else CharacterTextSplitter
    args = dict(chunk_size=tamanho, chunk_overlap=overlap, length_function=len)
    if estrategia == 'Recursivo':
        args['separators'] = ['\n\n', '\n', '. ', ' ', '']
    else:
        args['separator'] = ' '
    return classe(**args).split_text(texto)

def projetar(vetores, metodo='PCA'):
    if len(vetores) < 2:
        return np.zeros((len(vetores), 2))
    if metodo == 'PCA':
        return PCA(n_components=2, random_state=42).fit_transform(vetores)
    return TSNE(n_components=2, perplexity=min(10, len(vetores)-1), random_state=42, init='pca', learning_rate='auto').fit_transform(vetores)
