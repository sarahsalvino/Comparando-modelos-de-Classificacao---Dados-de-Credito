# Ajuste de Parâmetros — Comparação de Modelos de Classificação

Projeto de comparação de performance entre **6 algoritmos de classificação** aplicados a dados de crédito bancário, utilizando **GridSearchCV** para otimização de parâmetros, **Validação Cruzada com KFold** para avaliação de robustez e **VotingClassifier** para combinação dos modelos.

---

## Objetivo

Avaliar e comparar a performance de diferentes algoritmos de Machine Learning para prever a **inadimplência de clientes**, identificando qual modelo entrega os melhores resultados e como a combinação de modelos se comporta frente aos modelos individuais.

---

## Estrutura do Projeto

```
 projeto
 ┣ Ajuste_dos_parâmetros.ipynb   # Notebook principal
 ┣ credit_data.csv               # Dataset original
 ┗ README.md
```

---

## Dataset

- **2.000 registros** | **4 colunas utilizadas** (após remoção do `clientid`)
- **Variável alvo:** `default` 0 = Adimplente | 1 = Inadimplente
- **Distribuição:** 1.722 adimplentes (86.1%) | 278 inadimplentes (13.9%)
- Dataset desbalanceado maioria dos clientes são bons pagadores

### Colunas

| Coluna | Descrição |
|--------|-----------|
| `income` | Renda anual do cliente |
| `age` | Idade do cliente (anos) |
| `loan` | Valor do empréstimo solicitado |
| `default` | **Variável alvo** 0 = adimplente | 1 = inadimplente |

---

## Limpeza dos Dados

Foram identificados e corrigidos problemas na coluna `age`:

| Problema | Quantidade | Solução |
|----------|-----------|---------|
| Idades negativas | 3 registros (-28, -52, -36 anos) | Convertidas para positivo com `.abs()` |
| Valores nulos | 3 registros | Preenchidos com a mediana (41.35 anos) |

Após a correção: idade mínima **18.06 anos** | máxima **63.97 anos**

---

## Pré-processamento

Os dados foram normalizados com **StandardScaler** antes da aplicação dos modelos transformação que coloca todas as variáveis na mesma escala (média 0, desvio padrão 1). Isso é essencial para modelos como KNN, SVM e Redes Neurais que são sensíveis à escala dos dados.

```python
scaler = StandardScaler()
X_credit = scaler.fit_transform(X)
```

---

## Análise Exploratória (EDA)

- Dataset **desbalanceado** 86.1% adimplentes vs 13.9% inadimplentes
- Clientes inadimplentes tendem a ter **renda menor** e **empréstimos maiores** em relação à renda
- `loan` apresenta a maior correlação com `default` valor do empréstimo é o fator mais relevante
- `income` tem correlação negativa com `default` maior renda, menor risco de inadimplência
- `age` tem correlação negativa leve clientes mais velhos tendem a ser melhores pagadores

---

## Modelos Aplicados

### Etapa 1 — GridSearchCV

O **GridSearchCV** testa todas as combinações de parâmetros definidas e retorna a melhor configuração para cada modelo, usando validação cruzada com 5 folds internamente.

| Modelo | Acurácia | Melhores Parâmetros |
|--------|----------|---------------------|
| Árvore de Decisão | 98.65% | criterion=gini, splitter=best, min_samples_leaf=1, min_samples_split=5 |
| Random Forest | 98.90% | criterion=gini, n_estimators=100, min_samples_leaf=1, min_samples_split=5 |
| KNN | 98.05% | n_neighbors=5, p=2 |
| Regressão Logística | 94.85% | C=1.5, solver=lbfgs, tol=0.0001 |
| SVM | 98.45% | C=1.5, kernel=rbf, tol=0.001 |
| **Redes Neurais** | **99.70%** | activation=relu, batch_size=56, max_iter=500, solver=adam |

### Etapa 2 — Validação Cruzada (KFold)

A validação cruzada com **30 iterações x 10 folds** avalia a robustez dos modelos testando em diferentes divisões dos dados muito mais confiável do que uma única avaliação. Cada modelo foi treinado e avaliado 300 vezes no total (30 × 10).

>  **Nota:** Na validação cruzada os modelos foram configurados de forma diferente dos melhores parâmetros encontrados pelo GridSearch os parâmetros da validação cruzada não foram os ótimos encontrados anteriormente, o que pode explicar pequenas variações nos resultados.

### Etapa 3 — VotingClassifier

O **VotingClassifier** combina todos os 6 modelos com os melhores parâmetros do GridSearchCV usando `voting='soft'` cada modelo contribui com a probabilidade de cada classe, e a decisão final é a média ponderada dessas probabilidades.

| Técnica | Acurácia Média | Desvio Padrão |
|---------|---------------|---------------|
| Voting (combinação dos 6 modelos) | **99.20%** | 0.0010 |

---

## Resultados e Conclusão

- **Redes Neurais** foi o modelo mais preciso com **99.70%** no GridSearch melhor resultado individual
- **Random Forest** surpreendeu positivamente atingindo **98.90%**, superando SVM e KNN
- **Regressão Logística** foi o modelo mais fraco (94.84%) esperado, pois dados de crédito possuem relações não-lineares que modelos simples não capturam bem
- O **VotingClassifier** atingiu **99.20%** combinando todos os modelos superior à maioria dos modelos individuais, mas não superou a Rede Neural isolada, pois a Regressão Logística mais fraca puxa levemente o resultado para baixo
- Os desvios padrão baixos em todos os modelos indicam resultados **estáveis e confiáveis**, não fruto de sorte em uma divisão específica dos dados

---

## Tecnologias Utilizadas

- Python 3
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## Como Executar

1. Clone o repositório
```bash
git clone (https://github.com/sarahsalvino/Resultados-modelos-de-Classificacao---Dados-de-Credito.git)
```

2. Instale as dependências
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

3. Abra o notebook
```bash
jupyter notebook Ajuste_dos_parâmetros.ipynb
```

> Certifique-se de que o arquivo `credit_data.csv` está na mesma pasta do notebook antes de rodar.
