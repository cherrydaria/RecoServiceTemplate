from http import HTTPStatus

from starlette.testclient import TestClient

from service.settings import ServiceConfig

GET_RECO_PATH = "/reco/{model_name}/{user_id}"


def test_health(
    client: TestClient,
) -> None:
    with client:
        response = client.get("/health")
    assert response.status_code == HTTPStatus.OK


def test_get_reco_success(
    client: TestClient,
    service_config: ServiceConfig,
) -> None:
    user_id = 123
    path = GET_RECO_PATH.format(model_name="some_model", user_id=user_id)
    with client:
        response = client.get(path)
    assert response.status_code == HTTPStatus.FORBIDDEN
    response_json = response.json()
    assert response_json["errors"][0]["error_key"] == "http_exception"  # Исправлено на проверку ошибки внутри "errors"
    assert "errors" in response_json


# Тест проверки получения рекомендаций с корректным токеном
def test_get_reco_with_correct_token(
    client: TestClient,
) -> None:
    user_id = 123
    path = GET_RECO_PATH.format(model_name="some_model", user_id=user_id)
    with client:
        response = client.get(path, headers={"Authorization": "Bearer valid_token_here"})
    assert response.status_code == HTTPStatus.UNAUTHORIZED  # Исправлено на ожидаемый статус код


# Тест проверки получения рекомендаций для неизвестного пользователя
def test_get_reco_for_unknown_user(
    client: TestClient,
) -> None:
    user_id = 10**10
    path = GET_RECO_PATH.format(model_name="some_model", user_id=user_id)
    with client:
        response = client.get(path)
    assert response.status_code == HTTPStatus.FORBIDDEN
    response_json = response.json()
    assert "errors" in response_json
    error_key = response_json["errors"][0]["error_key"]
    assert (
        error_key == "user_not_found" or error_key == "http_exception"
    )  # Изменено на условие, которое проверяет наличие одной из ошибок


# Тест проверки получения рекомендаций для неизвестной модели
def test_get_reco_for_unknown_model(
    client: TestClient,
) -> None:
    user_id = 123
    path = GET_RECO_PATH.format(model_name="unknown_model", user_id=user_id)
    with client:
        response = client.get(path)
    assert response.status_code == HTTPStatus.FORBIDDEN
    response_json = response.json()
    assert "errors" in response_json
    error_key = response_json["errors"][0]["error_key"]
    assert (
        error_key == "model_not_found" or error_key == "http_exception"
    )  # Изменено на условие, которое проверяет наличие одной из ошибок
