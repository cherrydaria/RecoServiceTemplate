from typing import List

# Импорт необходимых классов и исключений из FastAPI
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request

# Импорт механизма аутентификации HTTPBearer из FastAPI
from fastapi.security import HTTPBearer
from pydantic import BaseModel

# Импорт исключений для обработки ошибок поиска модели и пользователя
from service.api.exceptions import ModelNotFoundError, UserNotFoundError
from service.log import app_logger

# Импорт библиотеки для распаковки файла
import dill

class RecoResponse(BaseModel):
    user_id: int
    items: List[int]


# Инициализация списка имен моделей
models_list = ["tunnel_test_request", "User_KNN"]

router = APIRouter()

# Создание экземпляра объекта HTTPBearer для аутентификации
bearer = HTTPBearer()

# Распаковка файла
with open('userknn.dill', 'rb') as f:
    User_KNN = dill.load(f)

@router.get(
    path="/health",
    tags=["Health"],
)
async def health() -> str:
    return "I am alive"


@router.get(
    path="/reco/{model_name}/{user_id}",
    tags=["Recommendations"],
    response_model=RecoResponse,
)
async def get_reco(
    request: Request, model_name: str, user_id: int, token: str = Depends(bearer)  # Зависимость для извлечения токена
) -> RecoResponse:
    app_logger.info(f"Request for model: {model_name}, user_id: {user_id}")

    # Write your code here
    # Проверка совпадения токена с ожидаемым токеном
    true_token = "darias_token"
    auth_token = token.credentials
    if auth_token != true_token:
        raise HTTPException(status_code=401, detail="Wrong token")

    # Вызов исключения при отсутствии указанной модели в списке моделей
    if model_name not in models_list:
        raise ModelNotFoundError(error_message=f"Model {model_name} not found")

    if user_id > 10**9:
        raise UserNotFoundError(error_message=f"User {user_id} not found")

    k_recs = request.app.state.k_recs
    reco = list(range(k_recs))
    return RecoResponse(user_id=user_id, items=reco)


def add_views(app: FastAPI) -> None:
    app.include_router(router)
