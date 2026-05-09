# Plano de Monitoramento do Modelo

Após o deploy da API de inferência, o monitoramento contínuo é essencial para detectar degradações tanto no serviço (latência, disponibilidade) quanto no próprio modelo (drift de dados, queda de performance). Este documento define as métricas, limiares de alerta e o playbook de resposta propostos para o acompanhamento do sistema em produção.

## 1. Métricas de Serviço (Monitoramento Operacional)

Métricas destinadas a acompanhar a disponibilidade e o desempenho da API de inferência:

- **Latência (P50, P90, P99)**: Tempo de resposta da API por requisição. Conforme implementado no projeto, o middleware de latência registra o tempo de processamento de cada requisição no header `X-Process-Time` da resposta, viabilizando a coleta desta métrica em ambiente produtivo.
- **Taxa de Erro HTTP**: Percentual de respostas com códigos 4xx (erro de entrada do cliente) e 5xx (falha no servidor). O endpoint `/health` permite a verificação do estado dos modelos carregados em memória.
- **Throughput (RPS)**: Volume de requisições por segundo recebidas pelo serviço.

Em ambiente produtivo, recomenda-se a integração dessas métricas com ferramentas de observabilidade como Prometheus/Grafana ou equivalente do provedor de nuvem utilizado.

## 2. Métricas de Machine Learning (Monitoramento Preditivo)

Como o ground truth (confirmação de que o cliente de fato cancelou) tipicamente demora a ser disponibilizado, o monitoramento preditivo deve atuar com proxies e observabilidade sobre os dados de entrada e as distribuições de saída do modelo.

### 2.1 Data Drift (Desvio dos Dados de Entrada)

Refere-se a mudanças significativas nas características dos dados de entrada em relação aos dados utilizados no treinamento.

- **Métricas recomendadas**: Divergência de Kullback-Leibler (KL) e Teste de Kolmogorov-Smirnov (KS).
- **Variáveis prioritárias**: `Monthly Charges` e proporção de contratos `Month-to-month`, identificadas na EDA como as features de maior relevância para a predição de churn.
- **Exemplo de cenário**: Um aumento repentino de clientes com faturamento zero (possível erro sistêmico na ingestão) ou o surgimento de um novo tipo de contrato não previsto pelo `OneHotEncoder` (tratado com `handle_unknown="ignore"`).

### 2.2 Concept Drift (Desvio de Conceito)

Ocorre quando a relação entre as variáveis de entrada e o target (Churn) sofre alteração ao longo do tempo.

- **Exemplo de cenário**: A introdução de uma nova taxa pela operadora que leva clientes antigos e de longo prazo a cancelarem — um padrão não observado nos dados de treinamento.
- **Detecção**: Comparar as métricas de ROC-AUC e F1-Score obtidas no treinamento (registradas no MLflow) com as métricas recalculadas sobre janelas temporais de 30 dias, utilizando os rótulos reais fornecidos pelo CRM.

### 2.3 Prediction Drift (Desvio de Predição)

Refere-se a mudanças na distribuição das probabilidades emitidas pelo modelo, mesmo na ausência do ground truth.

- **Exemplo de cenário**: A proporção de clientes classificados como "Risco de Churn" (probabilidade ≥ 0.5) salta de ~26% (taxa histórica do dataset) para 60% em uma semana, sem alteração correspondente no negócio.

## 3. Alertas e Limiares Propostos

| Alarme | Limiar | Severidade | Ação Esperada |
|:-------|:-------|:-----------|:--------------|
| API indisponível (HTTP 500) | > 1% das requisições/min | Crítica (P1) | Executar rollback para a versão anterior do container. |
| Falha na validação de entrada | Schema rejections > 5% | Alta (P2) | Investigar a origem dos dados (integração com CRM/Frontend). |
| Prediction Drift | Aumento de +15% na proporção de churners previstos/semana | Média (P3) | Notificar a equipe de ciência de dados para análise exploratória. |
| Degradação de performance | ROC-AUC inferior a 0,80 na janela mensal | Alta (P2) | Acionar o processo de retreinamento com dados recentes. |

## 4. Playbook de Resposta (Retreinamento)

Procedimento proposto para o tratamento de alertas de degradação de performance ou drift:

1. **Diagnóstico**: Extrair e analisar os dados rejeitados ou que apresentaram desvio, utilizando os notebooks de EDA como base.
2. **Retreinamento em sombra (Shadow Deployment)**: Executar o pipeline de treinamento (`python -m src.training.train`) com os dados mais recentes. Os novos experimentos serão automaticamente registrados no MLflow, incluindo o hash MD5 do dataset (`dataset_hash`), permitindo rastreabilidade completa.
3. **Avaliação Campeão vs. Desafiante (Champion/Challenger)**: Comparar o modelo em produção (Campeão) com o modelo recém-treinado (Desafiante) utilizando as métricas registradas no MLflow (AUC-ROC, PR-AUC, F1, Precision, Recall).
4. **Promoção**: Caso o modelo desafiante apresente métricas superiores sem comprometer segmentos específicos (conforme verificação no Model Card), promovê-lo ao Model Registry e atualizar o serviço de inferência.
