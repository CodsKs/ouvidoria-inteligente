import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from ouvidoria import carregar_base, codificar, dividir, projetar, MODELOS, BASE, modelo

@st.cache_resource
def modelo_app(nome):
    return modelo(nome)

@st.cache_data
def embeddings(textos, nome):
    return modelo_app(nome).encode(list(textos), normalize_embeddings=True)

def mapa(vetores, rotulos, metodo):
    xy = projetar(vetores, metodo)
    frame = pd.DataFrame({'x': xy[:, 0], 'y': xy[:, 1], 'categoria': list(rotulos)})
    st.scatter_chart(frame, x='x', y='y', color='categoria')

def main():
    st.set_page_config(page_title='Ouvidoria Inteligente', layout='wide')
    st.title('Ouvidoria Inteligente')
    nome = st.sidebar.selectbox('Modelo de embedding', MODELOS)
    topk = st.sidebar.slider('Top-k', 1, 20, 5)
    st.sidebar.caption('Primeiro uso de cada modelo requer internet.')
    df = carregar_base()
    origem = json.loads((BASE / 'dados' / 'proveniencia.json').read_text(encoding='utf-8'))
    st.warning(origem['aviso'])
    try:
        vetores = embeddings(tuple(df.texto), nome)
    except Exception as exc:
        st.error(f'Falha ao carregar embeddings: {exc}')
        return
    busca, base, espaco, chunking = st.tabs(['🔍 Busca Semântica', '📋 Base Completa', '🌐 Espaço Vetorial', '🧩 Chunking'])
    with busca:
        consulta = st.text_input('Descreva o problema')
        if consulta.strip():
            scores = vetores @ embeddings((consulta,), nome)[0]
            resultado = df.assign(score=scores).sort_values('score', ascending=False, kind='stable').head(topk)
            for _, item in resultado.iterrows():
                cor = '🟢' if item.score > .7 else '🟡' if item.score > .5 else '🔴'
                st.markdown(f'**{cor} {item.id} · {item.categoria_oficial} · {item.score:.3f}**')
                st.write(item.texto)
            st.caption('Cosseno mede proximidade de representação; não confirma duplicidade.')
    with base:
        st.dataframe(df, hide_index=True)
        if st.button('Gerar matriz de similaridade'):
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(vetores @ vetores.T, xticklabels=df.id, yticklabels=df.id, vmin=-1, vmax=1, cmap='coolwarm', ax=ax)
            st.pyplot(fig)
            plt.close(fig)
    with espaco:
        metodo = st.selectbox('Projeção', ['PCA', 't-SNE'])
        mapa(vetores, df.categoria_oficial, metodo)
        st.write('Categorias próximas podem compartilhar vocabulário, mas denúncias multitemáticas atravessam grupos. A projeção 2D distorce distâncias e não comprova a existência de clusters; compare também as similaridades no espaço original e a análise do notebook.')
    with chunking:
        texto = st.text_area('Manifestação longa', df.loc[df.texto.str.len().idxmax(), 'texto'], height=200)
        estrategia = st.selectbox('Estratégia', ['Recursivo', 'Por palavras'])
        tamanho = st.slider('Tamanho em caracteres', 80, 800, 250, 10)
        overlap = st.slider('Overlap em caracteres', 0, tamanho-1, min(50, tamanho-1))
        chunks = dividir(texto, tamanho, overlap, estrategia)
        if chunks:
            emb = embeddings(tuple(chunks), nome)
            st.dataframe(pd.DataFrame({'chunk': chunks, 'caracteres': [len(c) for c in chunks]}))
            st.write(f'Embeddings: {emb.shape[0]} vetores de {emb.shape[1]} dimensões')
            st.dataframe(pd.DataFrame(emb))
            if len(chunks) > 1:
                mapa(emb, ['Manifestação'] * len(chunks), 'PCA')

if __name__ == '__main__':
    main()
