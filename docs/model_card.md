# Model Card: Previsão de Churn de Telecom

## Visão Geral do Modelo
- **Tarefa**: Classificação binária para prever se um cliente irá cancelar o serviço (churn).
- **Arquitetura Base**: Rede Neural Multilayer Perceptron (MLP) construída em PyTorch.
- **Entrada**: Dados tabulares com features demográficas (ex: gênero), informações sobre o serviço contratado (ex: tipo de internet) e informações de faturamento (ex: MonthlyCharges, TotalCharges).
- **Saída**: Uma probabilidade contínua (0 a 1) indicando a chance de churn, além da classe predita.

## Dados e Preprocessamento
- **Dataset**: Telco Customer Churn (IBM Cognos Analytics 11.1.3+ base samples). Dados referentes a uma operadora fictícia na Califórnia, com 7.043 observações.
- **Principais Variáveis Utilizadas**:
  - **Demográficas**: `Gender`, `Senior Citizen`, `Partner`, `Dependents`.
  - **Serviços**: `Tenure Months`, `Phone Service`, `Multiple Lines`, `Internet Service`, `Online Security`, `Online Backup`, `Device Protection`, `Tech Support`, `Streaming TV`, `Streaming Movies`.
  - **Contrato e Faturamento**: `Contract`, `Paperless Billing`, `Payment Method`, `Monthly Charge`, `Total Charges`.
- **Pré-processamento**:
  - Imputação de dados faltantes (especialmente em `Total Charges`).
  - Remoção de identificadores e informações com vazamento de target (leakers) (`CustomerID`, `Churn Reason`, `Churn Score`, `CLTV`, etc.).
  - Remoção de dados geográficos em formato texto/agregado sem poder de generalização geral (`Country`, `State`, `City`, `Lat Long`). Coordenadas puras (`Latitude`, `Longitude`) e `Zip Code` foram mantidas para tentar capturar clusters locais de churn.
  - Encoding de variáveis categóricas via `OneHotEncoder`.
  - Escalonamento numérico via `StandardScaler`.
- As transformações foram salvas e encapsuladas em um pipeline Scikit-Learn reprodutível para garantia de qualidade na inferência.

## Treinamento e Avaliação (MLflow Tracking)
Foram treinados modelos Baseline (Dummy Classifier e Regressão Logística) e Redes Neurais (MLP PyTorch).
O treinamento do MLP utilizou:
- **Função de Perda**: Binary Cross Entropy with Logits Loss (BCEWithLogitsLoss).
- **Métrica Principal**: F1-Score e ROC-AUC.
- **Estratégia de Validação**: Divisão de Treino/Teste estratificada (80/20) para lidar com possível desbalanceamento das classes e `Early Stopping` configurado para prevenir overfitting.
- Todos os hiperparâmetros (como `learning_rate`, `batch_size`, épocas) e a **versão do dataset** (hash MD5) foram documentados através do MLflow, garantindo rastreabilidade completa dos experimentos.

## Performance
A Rede Neural alcançou uma performance consistente contra os baselines lineares e de árvore, demonstrando robustez em capturar interações não-lineares. O trade-off de custo (Falsos Positivos vs. Falsos Negativos) foi equilibrado dependendo do threshold de decisão. Um falso negativo tem alto impacto no negócio pois significa a perda da receita do cliente.

## Limitações e Vieses
- **Desbalanceamento**: Tipicamente, dados de churn sofrem de desbalanceamento (mais clientes fiéis do que cancelados). Caso as distribuições sofram alteração, a acurácia do modelo pode degradar rapidamente.
- **Vieses Demográficos**: O modelo inclui características que podem trazer questões éticas se utilizadas para penalizações de serviço (por ex: `SeniorCitizen`, `Gender`). Deve-se restringir o uso preditivo unicamente a ações de retenção e marketing positivo, jamais para encarecimento ou discriminação tarifária.
- **Limitações de Temporalidade**: O modelo reflete o comportamento passado do cliente. Se a operadora introduzir um novo serviço ou reajuste de preço agressivo de uma só vez, o modelo não saberá lidar de imediato.

## Cenários de Falha e Out of Scope
- **Valores Anômalos (Outliers Absurdos)**: Faturamentos mensais extremamente fora da curva podem causar inferências distorcidas. A validação de entrada na API é realizada pelo schema `Pydantic`, que garante tipos e estrutura do payload. O `pandera` é utilizado na validação offline do dataset durante o treinamento e nos testes automatizados.
- **Out of Scope**: Este modelo não realiza recomendação sobre *qual* oferta deve ser feita ao cliente (apenas prevê a probabilidade do churn). O "Next Best Action" fica a cargo de regras de negócio adicionais após a inferência.
