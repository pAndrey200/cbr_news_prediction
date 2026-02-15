"""API-роуты для управления задачами (async)."""

import logging
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from cbr_news.database import get_async_db
from cbr_news.models import TaskStatus, TaskType
from cbr_news.task_repository import TaskRepositoryAsync
from cbr_news.tasks import run_prediction, run_training, _default_checkpoint_path

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/tasks", tags=["tasks"])


# ---- Pydantic-схемы ----

class TrainTaskRequest(BaseModel):
    overrides: list[str] = Field(
        default_factory=list,
        description="Hydra-style overrides, например ['training.max_epochs=5']",
    )


class PredictTaskRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=100)
    checkpoint_path: Optional[str] = None
    config_path: Optional[str] = None


class TaskResponse(BaseModel):
    id: str
    task_type: str
    status: str
    params: Optional[dict] = None
    result: Optional[dict] = None
    error: Optional[str] = None
    celery_task_id: Optional[str] = None
    created_at: str
    updated_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


class TaskListResponse(BaseModel):
    tasks: list[TaskResponse]
    total: int


def _task_to_response(task) -> TaskResponse:
    return TaskResponse(
        id=str(task.id),
        task_type=task.task_type,
        status=task.status,
        params=task.params,
        result=task.result,
        error=task.error,
        celery_task_id=task.celery_task_id,
        created_at=task.created_at.isoformat(),
        updated_at=task.updated_at.isoformat(),
        started_at=task.started_at.isoformat() if task.started_at else None,
        completed_at=task.completed_at.isoformat() if task.completed_at else None,
    )


# ---- Endpoints ----

@router.post("/train", response_model=TaskResponse, status_code=201)
async def create_training_task(
    request: TrainTaskRequest,
    db: AsyncSession = Depends(get_async_db),
):
    """Создать задачу на обучение модели."""
    params = {"overrides": request.overrides}
    task = await TaskRepositoryAsync.create(
        db, task_type=TaskType.train.value, params=params
    )
    run_training.delay(str(task.id), params)
    return _task_to_response(task)


@router.post("/predict", response_model=TaskResponse, status_code=201)
async def create_prediction_task(
    request: PredictTaskRequest,
    db: AsyncSession = Depends(get_async_db),
):
    """Создать задачу на батч-предсказание."""
    from cbr_news.tasks import _resolve_checkpoint
    checkpoint = (
        _resolve_checkpoint(request.checkpoint_path)
        if request.checkpoint_path
        else _default_checkpoint_path()
    )
    if not checkpoint:
        raise HTTPException(
            422,
            "Чекпоинт модели не найден. Обучите модель через POST /tasks/train"
        )

    params = {"texts": request.texts, "checkpoint_path": checkpoint}
    if request.config_path:
        params["config_path"] = request.config_path

    task = await TaskRepositoryAsync.create(
        db, task_type=TaskType.predict.value, params=params
    )
    run_prediction.delay(str(task.id), params)
    return _task_to_response(task)


@router.get("/{task_id}", response_model=TaskResponse)
async def get_task(
    task_id: UUID,
    db: AsyncSession = Depends(get_async_db),
):
    """Получить статус задачи по ID."""
    task = await TaskRepositoryAsync.get_by_id(db, task_id)
    if not task:
        raise HTTPException(404, "Задача не найдена")
    return _task_to_response(task)


@router.get("", response_model=TaskListResponse)
async def list_tasks(
    task_type: Optional[str] = Query(None, description="Фильтр по типу: train/predict"),
    status: Optional[str] = Query(None, description="Фильтр по статусу: pending/running/completed/failed"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_async_db),
):
    """Получить список задач с фильтрацией и пагинацией."""
    tasks = await TaskRepositoryAsync.list_tasks(
        db,
        task_type=task_type,
        status=status,
        limit=limit,
        offset=offset,
    )
    return TaskListResponse(
        tasks=[_task_to_response(t) for t in tasks],
        total=len(tasks),
    )
