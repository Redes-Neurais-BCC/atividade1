# 🏀 Dallas Mavericks 2024-25 - Análise Exploratória de Dados

## 👥 Equipe

- **[Henrique Almeida](https://github.com/henriquedalmeida)**
- **[Felipe Mendes](https://github.com/FelipeMendes1)**
- **[Gison Vilaça](https://github.com/gison-vilaca)**

## 📚 Informações Acadêmicas

- **Disciplina:** Redes Neurais
- **Curso:** Ciência da Computação
- **Professor:** Ryan Azevedo

---

## 📋 Descrição do Projeto

Este projeto desenvolve uma aplicação completa de análise exploratória de dados (EDA) dos **Dallas Mavericks** na temporada 2024-25, incluindo implementações de **Regressão Linear** e **Regressão Logística** para análises preditivas. A aplicação oferece insights detalhados sobre performance de jogadores e equipe através de uma interface web interativa.

## 🚀 Acesso à Aplicação

### 🌐 **Deploy Online**

**Acesse a aplicação em funcionamento:** [https://dallas-1-atividade.streamlit.app](https://dallas-atividade-1.streamlit.app)

> 💡 **Dica:** A aplicação está hospedada no Streamlit Cloud e pode ser acessada diretamente pelo navegador, sem necessidade de instalação local.

## 🎯 Objetivos

### Objetivo Principal

Criar uma plataforma interativa para análise estatística e preditiva do desempenho dos Dallas Mavericks, permitindo:

1. **Análise Exploratória Completa** dos dados da temporada 2024-25
2. **Implementação de Modelos de Machine Learning** (Regressão Linear e Logística)
3. **Interface Interativa** para visualização e predições
4. **Insights Estratégicos** para tomada de decisões

### Objetivos Específicos

- Análise de performance individual dos jogadores
- Análise de performance da equipe em jogos
- Implementação de modelos preditivos
- Criação de interface web responsiva
- Desenvolvimento de sistema de predições personalizadas

## 🛠️ Tecnologias Utilizadas

### Linguagens e Frameworks

- **Python 3.8+** - Linguagem principal
- **Streamlit** - Framework para interface web
- **Pandas** - Manipulação e análise de dados
- **NumPy** - Computação numérica

### Bibliotecas de Machine Learning

- **Scikit-learn** - Modelos de regressão e métricas
- **StandardScaler** - Normalização de dados

### Visualização de Dados

- **Plotly** - Gráficos interativos
- **Matplotlib** - Visualizações estáticas
- **Seaborn** - Visualizações estatísticas

### Fonte de Dados

- **NBA API** - Dados oficiais da NBA

## 📁 Estrutura do Projeto

```
atividade1/

├── data/                        # Diretório de dados
│   ├── original/                # Dados originais da NBA API
│   ├── interim/                 # Dados processados intermediários
│   ├── processed/               # Dados finais processados
│   ├── mappings/                # Mapeamentos de tradução
│   └── colabs/                  # Modelos
├── src/
│    ├── extract/                 # Extração de dados
│    ├── preprocess/              # Pré-processamento
│    └── interface/               # Interface web
├── README.md                    # Documentação do projeto
├── .gitignore                   # Arquivos ignorados pelo Git
├── requirements.txt             # Dependências Python
├── run_streamlit.bat            # Script para execução Windows
```

## 🔧 Instalação e Configuração

### Pré-requisitos

- Python 3.8 ou superior
- pip (gerenciador de pacotes Python)

### Instalação

1. **Clone o repositório:**

```bash
git clone https://github.com/Redes-Neurais-BCC/atividade1.git
cd atividade1
```

2. **Instale as dependências:**

```bash
pip install -r requirements.txt
```

3. **Execute a aplicação:**

**Windows:**

```bash
run_streamlit.bat
```

**Linux/Mac:**

```bash
streamlit run src/interface/streamlit_app.py
```

4. **Acesse a aplicação:**

- Abra o navegador em: `http://localhost:8501`

## 📊 Funcionalidades Principais

### 1. 📈 Análise de Jogadores

- **Top 10 Pontuadores** com visualização em barras
- **Análise de Eficiência de Arremessos** com scatter plot interativo
- **Distribuição por Posição** com gráfico de pizza
- **Estatísticas médias por posição**

### 2. 🏀 Análise de Jogos

- **Performance temporal** com linha do tempo de pontos
- **Comparação Casa vs Fora** com métricas detalhadas
- **Matriz de Correlação** entre estatísticas
- **Análise de tendências** da temporada

### 3. 🧠 Análise Avançada

- **Eficiência vs Taxa de Uso** dos jogadores
- **Análise de Clutch Time** (jogos apertados vs folgados)
- **Métricas avançadas** de performance

### 4. 🔍 Análise Interativa

- **Filtros dinâmicos** na sidebar
- **Gráficos customizáveis** com seleção de eixos
- **Tabela filtrada** de jogadores
- **Controles responsivos**

### 5. 🎯 Predições Específicas

- **Predições de Jogadores Individuais**
  - Pontos, rebotes, assistências por jogo
  - Probabilidade de atingir metas específicas
- **Predições do Time**
  - Performance em próximos jogos
  - Influência do mando de campo

### 6. 📈 Regressão Linear

- **Modelo matemático completo** com equações LaTeX
- **Configuração interativa** de variáveis
- **Métricas de avaliação** (R², RMSE, MSE)
- **Visualizações avançadas**:
  - Dispersão com linha de regressão
  - Predito vs Real
  - Análise de resíduos
  - Importância das variáveis
- **Interface de predição personalizada**

### 7. 📊 Regressão Logística

- **Classificação binária** com probabilidades
- **Configuração flexível** de limites de classificação
- **Matriz de confusão interativa**
- **Métricas de classificação** (Acurácia, Precisão, Recall)
- **Visualizações especializadas**:
  - Distribuição de probabilidades
  - Coeficientes e Odds Ratio
  - Importância das features

## 🧮 Modelos de Machine Learning

### Regressão Linear Múltipla

**Equação Matemática:**

```
y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε
```

**Características:**

- **Variáveis Dependentes:** Pontos, Rebotes, Assistências, etc.
- **Variáveis Independentes:** Estatísticas de jogo configuráveis
- **Métricas:** R² (Coeficiente de Determinação), RMSE, MSE
- **Validação:** Train/Test Split configurável

### Regressão Logística

**Equação Matemática:**

```
p(y=1) = 1 / (1 + e^(-z))
z = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
```

**Características:**

- **Classificações:** Vitória/Derrota, Alto/Baixo desempenho
- **Probabilidades:** Cálculo de probabilidades de classes
- **Métricas:** Acurácia, Matriz de Confusão, F1-Score
- **Features:** Normalização automática com StandardScaler

## 📈 Resultados e Insights

### Modelo de Regressão Logística (Exemplo)

- **Acurácia:** 55.56%
- **Principais Fatores de Vitória:**
  - Mando de jogo (coef: +0.731)
  - Triplos convertidos (coef: +0.174)
  - Porcentagem de triplos (coef: +0.075)

### Correlações Identificadas

- **Saldo de pontos** ↔ **Resultado** (r = 0.783)
- **Pontos totais** ↔ **Resultado** (r = 0.493)
- **Porcentagem de triplos** ↔ **Resultado** (r = 0.469)

## 🎨 Interface do Usuário

### Design e Usabilidade

- **Layout Responsivo** com Streamlit
- **Tema Personalizado** com cores dos Mavericks
- **Navegação por Tabs** para organização
- **Sidebar Interativa** com filtros dinâmicos
- **Visualizações Interativas** com Plotly

### Seções da Aplicação

1. **👥 Jogadores** - Análise individual e estatísticas
2. **🏀 Jogos** - Performance da equipe por jogo
3. **🧠 Análise Avançada** - Métricas sofisticadas
4. **🔍 Interativa** - Filtros e customização
5. **🎯 Predições Específicas** - Previsões personalizadas
6. **📈 Regressão Linear** - Modelo matemático completo
7. **📊 Regressão Logística** - Classificação e probabilidades

## 🔄 Pipeline de Dados

### 1. Extração (`extract_data.py`)

- Conexão com NBA API
- Download de dados da temporada 2024-25
- Salvamento em formato CSV

### 2. Pré-processamento (`data_preprocessing.py`)

- Limpeza de dados nulos e inconsistentes
- Tradução de colunas para português
- Normalização de formatos de data
- Criação de variáveis derivadas

### 3. Processamento Final (`data_cleanup_selection.py`)

- Seleção de features relevantes
- Agregação de estatísticas
- Criação de datasets finais

### 4. Interface (`streamlit_app.py`)

- Carregamento de dados processados
- Implementação de modelos ML
- Criação de visualizações
- Gerenciamento de estado da aplicação

## 🧪 Validação e Testes

### Validação dos Modelos

- **Cross-validation** para robustez
- **Train/Test Split** configurável (10-40%)
- **Métricas múltiplas** para avaliação
- **Análise de resíduos** para diagnóstico

### Testes de Interface

- **Responsividade** em diferentes telas
- **Performance** com datasets grandes
- **Usabilidade** com feedback do usuário
- **Robustez** com dados faltantes

## 📝 Considerações Técnicas

### Performance

- **Cache de dados** com `@st.cache_data`
- **Otimização de gráficos** com Plotly
- **Processamento eficiente** com Pandas/NumPy

### Escalabilidade

- **Arquitetura modular** por funcionalidades
- **Separação de responsabilidades** (extract/process/interface)
- **Configuração flexível** de parâmetros

### Manutenibilidade

- **Código documentado** com docstrings
- **Estrutura organizada** por diretórios
- **Funções reutilizáveis** para análises

## 🚀 Futuras Melhorias

### Funcionalidades Planejadas

1. **Mais Modelos ML** (Random Forest, Neural Networks)
2. **Análise Temporal Avançada** (séries temporais)
3. **Comparação entre Times** da NBA
4. **API própria** para integração externa
5. **Dashboard executivo** com KPIs
6. **Análise de vídeo** integrada
7. **Predições em tempo real** durante jogos

### Melhorias Técnicas

1. **Containerização** com Docker
2. **Deploy automatizado** em cloud
3. **Banco de dados** para persistência
4. **Autenticação** de usuários
5. **Logs e monitoramento** avançados

## 📞 Suporte e Contato

Para dúvidas, sugestões ou contribuições:

- **🌐 Aplicação Online:** [https://dallas-atividade-1.streamlit.app](https://dallas-atividade-1.streamlit.app)
- **📂 Repositório:** [GitHub - atividade1](https://github.com/Redes-Neurais-BCC/atividade1)
- **👥 Equipe de Desenvolvimento:**
  - [Henrique Almeida](https://github.com/henriquedalmeida)
  - [Felipe Mendes](https://github.com/FelipeMendes1)
  - [Gison Vilaça](https://github.com/gison-vilaca)

---

**Desenvolvido para a disciplina de Redes Neurais - Ciência da Computação**
