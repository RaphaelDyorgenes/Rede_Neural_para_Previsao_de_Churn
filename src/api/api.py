# ================================
# IMPORTS
# ================================
import json
import logging
import time
from contextlib import asynccontextmanager

import pandas as pd
import torch
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

from src.config import MODELS_DIR
from src.data.preprocessing import load_pipeline
from src.models.models import ChurnMLPv2

# ================================
# CONFIGURAÇÃO DO LOGGER ESTRUTURADO
# ================================
class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "latency"):
            log_record["latency"] = record.latency
        if hasattr(record, "path"):
            log_record["path"] = record.path
        if hasattr(record, "status_code"):
            log_record["status_code"] = record.status_code
        return json.dumps(log_record)

def setup_logging():
    logger = logging.getLogger()
    # Remove handlers padrão para evitar logs duplicados
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)

setup_logging()
logger = logging.getLogger(__name__)


# ================================
# CICLO DE VIDA (LIFESPAN) E CARREGAMENTO DE MODELOS
# ================================
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Carrega os modelos ao iniciar
    logger.info("Iniciando carregamento da pipeline e modelo...")
    try:
        pipeline = load_pipeline(MODELS_DIR / "pipeline.joblib")
        ml_models["pipeline"] = pipeline

        # Para carregar o modelo precisamos saber o input_dim.
        # Podemos ler do state_dict ou inferir transformando um dummy.
        state_dict = torch.load(MODELS_DIR / "mlp_best.pt", weights_only=True, map_location="cpu")
        # Pega a dimensão da primeira camada Linear: shape (out_features, in_features)
        input_dim = state_dict['network.0.weight'].shape[1]

        model = ChurnMLPv2(input_dim=input_dim)
        model.load_state_dict(state_dict)
        model.eval()
        ml_models["model"] = model

        logger.info(f"Modelos carregados com sucesso. Input dim = {input_dim}")
    except Exception as e:
        logger.error(f"Erro ao carregar modelos: {e}. Execute 'make train' antes de iniciar a API.")

    yield

    # Limpeza ao encerrar
    ml_models.clear()


# ================================
# APLICAÇÃO FASTAPI
# ================================
app = FastAPI(title="Telco Churn API", version="1.0.0", lifespan=lifespan)

# ================================
# MIDDLEWARE DE LATÊNCIA
# ================================
@app.middleware("http")
async def latency_middleware(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = time.perf_counter() - start_time

    # Adiciona no Header
    response.headers["X-Process-Time"] = str(process_time)

    # Log Estruturado
    logger.info(
        f"{request.method} {request.url.path} - Status: {response.status_code} - {process_time:.4f}s",
        extra={
            "latency": process_time,
            "path": request.url.path,
            "status_code": response.status_code
        }
    )
    return response


# ================================
# SCHEMA DE ENTRADA (PYDANTIC)
# ================================
class CustomerData(BaseModel):
    # Numéricos
    count: int = Field(alias="Count", default=1)
    zip_code: int = Field(alias="Zip Code", default=0)
    latitude: float = Field(alias="Latitude", default=0.0)
    longitude: float = Field(alias="Longitude", default=0.0)
    tenure_months: float = Field(alias="Tenure Months", default=0)
    monthly_charges: float = Field(alias="Monthly Charges", default=0.0)
    total_charges: float = Field(alias="Total Charges", default=0.0)

    # Categóricos
    gender: str = Field(alias="Gender", default="Male")
    senior_citizen: str = Field(alias="Senior Citizen", default="No")
    partner: str = Field(alias="Partner", default="No")
    dependents: str = Field(alias="Dependents", default="No")
    phone_service: str = Field(alias="Phone Service", default="Yes")
    multiple_lines: str = Field(alias="Multiple Lines", default="No")
    internet_service: str = Field(alias="Internet Service", default="Fiber optic")
    online_security: str = Field(alias="Online Security", default="No")
    online_backup: str = Field(alias="Online Backup", default="No")
    device_protection: str = Field(alias="Device Protection", default="No")
    tech_support: str = Field(alias="Tech Support", default="No")
    streaming_tv: str = Field(alias="Streaming TV", default="No")
    streaming_movies: str = Field(alias="Streaming Movies", default="No")
    contract: str = Field(alias="Contract", default="Month-to-month")
    paperless_billing: str = Field(alias="Paperless Billing", default="Yes")
    payment_method: str = Field(alias="Payment Method", default="Electronic check")

    class Config:
        populate_by_name = True


# ================================
# ENDPOINTS
# ================================
@app.get("/health")
def health():
    # Verifica se os modelos estão em memória
    if "pipeline" not in ml_models or "model" not in ml_models:
        return {"status": "error", "message": "Modelos não carregados. Execute make train."}
    return {"status": "ok"}


@app.post("/predict")
def predict(customer: CustomerData):
    logger.info("Previsão solicitada", extra={"path": "/predict"})

    if "pipeline" not in ml_models or "model" not in ml_models:
        return {"error": "Modelos não carregados no servidor."}

    try:
        # Converter dados de entrada em DataFrame (com as chaves/aliás originais)
        # O model_dump() usará os alias que batem com as colunas do pandas original.
        df_input = pd.DataFrame([customer.model_dump(by_alias=True)])

        # Aplicar Transformação (Preprocessing)
        pipeline = ml_models["pipeline"]
        X_transformed = pipeline.transform(df_input)

        # Predição com Pytorch
        model = ml_models["model"]
        X_tensor = torch.FloatTensor(X_transformed)

        with torch.no_grad():
            logits = model(X_tensor).squeeze(dim=0) # Squeeze caso lote de 1
            prob = torch.sigmoid(logits).item()

        return {
            "churn_probability": round(prob, 4),
            "churn_prediction": prob >= 0.5,
        }
    except Exception as e:
        logger.error(f"Erro durante a predição: {str(e)}", exc_info=True)
        return {"error": str(e)}
