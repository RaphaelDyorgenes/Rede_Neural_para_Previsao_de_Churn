# ML Canvas — Churn Prediction (Telco)

Decisões de *feature selection* e pré-processamento foram confirmadas após a EDA (vide notebook `01_eda.ipynb`) e estão implementadas de forma a alimentar as avaliações no `02_baselines.ipynb` e `03_mlp_pytorch.ipynb`. Decisões de modelagem fina (threshold ótimo, early stopping) já foram calibradas no pipeline (MLP-v2).

| Item | Valor de referência |
| :--- | :--- |
| **Tipo de tarefa** | Classificação binária supervisionada |
| **Modelo principal** | MLP (PyTorch), 3 camadas ocultas (128 → 64 → 32) |
| **Métrica técnica primária** | PR-AUC (sensível ao desbalanceamento) |
| **Métrica de negócio** | Custo total esperado (FP × R$50 + FN × R$200) |
| **SLO de latência** | P95 < 200 ms por predição na API |
| **SLO de disponibilidade** | ≥ 99,9% |
| **Cadência de retreino** | Trimestral, ou sob alerta de drift |
| **Seed de reprodutibilidade** | SEED = 42 |

## 1. Proposta de Valor
A operadora de telecomunicações (fictícia, baseada na Califórnia/IBM) enfrenta churn em ritmo elevado (cerca de 26,5%) e perde receita recorrente. O modelo entrega, para cada cliente ativo, uma probabilidade de cancelamento no próximo período, permitindo ao time de retenção agir de forma proativa — antes do cancelamento acontecer — e priorizar esforço onde o custo esperado de inação é maior.

**Sucesso** = redução do custo total de churn (cliente perdido + ação desnecessária), não simplesmente acertar a classificação. A mitigação do custo demonstra uma economia esperada de R$ 46.400,00 no hold-out vis-à-vis atuar sem modelo.

## 2. Stakeholders e Usuários

| Papel | Como interage com o sistema |
| :--- | :--- |
| **Diretoria comercial** | Sponsor. Define meta de redução de churn e orçamento de retenção. Consome KPIs agregados. |
| **Time de retenção** | Usuário primário. Recebe lista priorizada de clientes em risco e executa as ações (oferta, contato, desconto). |
| **Time de CRM / canais** | Operacionaliza o contato (e-mail, ligação, push) na ferramenta atual. |
| **Engenharia de dados / MLOps** | Mantém pipeline, monitoramento e retreino. |
| **Cliente final** | Impactado indiretamente; recebe ofertas de retenção quando classificado como em risco. |

## 3. Tarefa de ML
* **Tipo:** Classificação binária supervisionada.
* **Variável-alvo:** `Churn Label` (Yes=1, cancelou no último ciclo; No=0, permaneceu).
* **Saída do modelo:** $P(churn = 1) \in [0, 1]$.
* **Decisão downstream:** Aplicar limite (threshold de 50%, ou calibrado dinamicamente para priorizar recall baseado na regra `FN = 4x FP`) para classificar os propensos ao churn.

## 4. Fonte de Dados
* **Dataset:** Telco Customer Churn (IBM), versão pública tratada (`telco_churn_clean.csv`).
* **Volume:** 7.043 observações × 33 colunas pré-engenharia.
* **Granularidade:** Uma linha por cliente, snapshot trimestral.
* **Class balance esperado:** ~26,5% de churn — desbalanceado, exige tratamento (`pos_weight = 2.77` na loss da rede neural).
* **Limitações conhecidas:**
  * Snapshot estático sem histórico serial completo.
  * Colunas identificadoras e derivadas intrínsecas ao rótulo atuam como *leakers* e sofreram expurgo.

## 5. Features (Decisões Reais — Pós-EDA)
Decisões justificadas no `01_eda.ipynb` de maneira a barrar fuga de predição e garantir integridade:

### 5.1 Descartes Comprovados
* **Vazamento de Target (Leakers):** `Churn Reason`, `Churn Score`, `Churn Value`, `CLTV` — Retiradas por pertencerem à própria mecânica do churn (anulando o poder profético da inferência).
* **Identificadores Constantes:** `CustomerID` descartado, não generaliza dados úteis.
* **Invariantes Regionais:** `Country`, `State`, `City`, `Lat Long` — Constantes ou demograficamente não correlatas em amostra homogênea.

### 5.2 Estrutura do Dummification (One-Hot)
As demais variáveis de contratação ("*Monthly Charges*", "*Internet Service*", "*Contract*", "*Tech Support*") compõem as **50 features numéricas/dummificadas** de entrada injetadas na rede, onde *Features Numéricas Contínuas* sofrem fit-transform com `StandardScaler` apenas para os conjuntos de treino.

## 6. Métricas

### Técnicas (Offline)
A métrica principal baseia-se na avaliação desbalanceada via precision-recall da classe focal.

| Métrica | Por quê | Alvo/Estado Operacional |
| :--- | :--- | :--- |
| **PR-AUC** | Essencial quando o custo de FN exige avaliação sob desbalanceamento; foca na classe alvo. | 0,62 ~ 0,65 |
| **ROC-AUC** | Analítica discriminativa geral dos baselines x rede neural. | ≥ 0,84 |
| **Recall** | Prioridade técnica derivada da matriz de custos (capturar churn é necessário). | ≥ 82% |
| **F1-Score** | Média harmônica de balanço Precision-Recall | Curva de Report |

### De Negócio
* **Custo Total Esperado:** Redução na métrica da equação $= (\text{Falso Positivo} \times R\$ 50) + (\text{Falso Negativo} \times R\$ 200)$.
* O foco prioritário (*recall* agressivo) atende à premissa de que perder de fato um cliente ($FN = R\$ 200$) custa 400% a mais do que dar uma oferta equivocada a um cliente satisfeito ($FP = R\$ 50$). 

## 7. SLOs e Sistema
| Dimensão | Alvo |
| :--- | :--- |
| **Latência por predição (API, P95)** | < 200 ms |
| **Disponibilidade do endpoint** | ≥ 99,9% |
| **Artefatos de Inferência** | PyTorch / Pickled Scalers rest-API |
| **Avaliação Operacional** | Deploy on-demand (FastAPI em Docker) real-time; A retenção reage dinamicamente aos alertas da API. |

## 8. Avaliação Offline Estrutural
* **Split de Validação (Neural):** 80% Treino / 20% Validação e Teste. Estratificados preservando taxa natural de ~26,5% do `Churn`. 
* **Baselines Comparativos:** Validação cruzada (Stratified 5-Fold) aplicados em Dummy, Logistic Regression, Decision Tree e Random Forest.  

## 9. Construção dos Modelos

| Modelo | Papel | Configuração Base |
| :--- | :--- | :--- |
| **Dummy** | Baseline zero | *most_frequent* |
| **Logistic Regression / Árvores** | Intermediários interpretáveis | Scikit-Learn `RandomForest`/`LogReg` / Max Iter=1000 |
| **MLP PyTorch (MLP-v2)** | Redutor extremo do Churn | **128 → 64 → 32 → 1**; BCELogitsLoss com *pos_weight*; Batch_Size=32; Adam; *Dropout(0.2)* |

**Controle Profissional:** Retenção de curva máxima usando `Early Stopping` estrito não na *loss de Treino* (que causava forte overfit), mas atuando na **Loss de Validação**, parando com `patience=15`. Rastreamento de métricas e hashes via backend unificado do **MLflow SQLite** registrando o portfólio completo de hiperparâmetros e pesos (`.pt`).

## 10. Inferências & Governança (Monitoramento e Retreino)
* A inferência do PyTorch e do Dicionário de Inputs aguardam consumo online.
* Monitoramento semântico requer verificação do `Data Drift`, bem como `Prediction Drift` se a porcentagem de *Churners* detectada exceder ou rebaixar > 10% nas janelas inter-períodos operacionais.
* **Riscos e Viés:** A detecção enviesada de churn contra populações minoritárias (Idosos/Seniors) é neutralizada através da auditoria periódica e de um forte direcionamento a evitar precificações punitivas aos classificados de forma errônea. O foco deve permanecer na prospecção benéfica de retenção (ofertas ganha-ganha para o cliente final).
