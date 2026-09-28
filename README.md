# Ouvidoria Inteligente

Protótipo de triagem semântica com busca, detecção de duplicatas e chunking. 

## Executar

Requer Python 3.12. Dentro desta pasta:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m streamlit run app_ouvidoria.py
```

No Linux/macOS, use `.venv/bin/python`. Com o ambiente ativado, o comando do enunciado funciona: `streamlit run app_ouvidoria.py`.

O primeiro uso de cada modelo exige internet para baixar pesos do Hugging Face. Não há API paga. Ambos os modelos são multilíngues reais, sem substituição por vetores aleatórios. Caches e modelos não devem ir para o Git.

## Notebooks e arquivos

Abra os notebooks no VS Code ou Jupyter com o ambiente Python criado acima e execute a partir desta pasta. Eles já incluem saídas da validação.

1. `análise_comparativa.ipynb`: BoW, TF-IDF e dois modelos, três pares exigidos, PCA/t-SNE e análise.
2. `deteccao_duplicatas.ipynb`: função `detectar_duplicatas(textos, limiar=0.85)`, matriz completa, pares, sensibilidade e falsos positivos/negativos.
3. `chunking_manifestacoes.ipynb`: cinco maiores textos, configurações 250/0, 250/50 e 400/80, exemplos e PCA.
4. `app_ouvidoria.py`: quatro abas, modelos, top-k, scores coloridos e embeddings dos chunks.
5. `ouvidoria.py`: funções comuns, para evitar inconsistências entre aplicativo e análises.
6. `RELATORIO.pdf`: síntese em até cinco páginas.
7. `dados/`: corpus, proveniência e gabarito; `resultados/`: tabelas e gráficos medidos.




## Validação e limites

```powershell
.\.venv\Scripts\python -m unittest discover -v
```

O limiar 0,85 é inicial, não calibrado em uma amostra independente. Scores não são probabilidades. PCA/t-SNE distorcem distâncias. Busca ocorre no nível de manifestação; a aba Chunking demonstra a divisão e visualização separadamente. 