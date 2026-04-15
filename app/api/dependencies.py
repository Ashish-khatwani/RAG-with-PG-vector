from fastapi import Request

from app.services.health import HealthService
from app.services.ingestion import IngestionService
from app.services.query import QueryService
from app.services.repository import DocumentRepository


def get_ingestion_service(request: Request) -> IngestionService:
    return request.app.state.ingestion_service


def get_query_service(request: Request) -> QueryService:
    return request.app.state.query_service


def get_repository(request: Request) -> DocumentRepository:
    return request.app.state.document_repository


def get_health_service(request: Request) -> HealthService:
    return request.app.state.health_service
