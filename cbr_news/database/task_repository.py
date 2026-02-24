"""Репозиторий для работы с задачами — sync и async варианты."""

import logging
from datetime import datetime
from typing import List, Optional
from uuid import UUID

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from cbr_news.database.models import Task, TaskStatus

logger = logging.getLogger(__name__)


class TaskRepositorySync:
    """Используется внутри Celery-задач (sync Session, psycopg2)."""

    @staticmethod
    def get_by_id(session: Session, task_id: UUID) -> Optional[Task]:
        return session.query(Task).filter(Task.id == task_id).first()

    @staticmethod
    def update_status(
        session: Session,
        task_id: UUID,
        status: str,
        *,
        celery_task_id: str = None,
        result: dict = None,
        error: str = None,
    ) -> None:
        task = session.query(Task).filter(Task.id == task_id).first()
        if not task:
            logger.warning("Задача %s не найдена при обновлении статуса", task_id)
            return
        task.status = status
        task.updated_at = datetime.utcnow()
        if celery_task_id:
            task.celery_task_id = celery_task_id
        if status == TaskStatus.running.value:
            task.started_at = datetime.utcnow()
        if status in (TaskStatus.completed.value, TaskStatus.failed.value):
            task.completed_at = datetime.utcnow()
        if result is not None:
            task.result = result
        if error is not None:
            task.error = error
        session.commit()


class TaskRepositoryAsync:
    """Используется внутри FastAPI-роутов (AsyncSession, asyncpg)."""

    @staticmethod
    async def create(
        session: AsyncSession,
        task_type: str,
        params: dict = None,
    ) -> Task:
        import uuid as uuid_mod
        task = Task(
            id=uuid_mod.uuid4(),
            task_type=task_type,
            status=TaskStatus.pending.value,
            params=params,
        )
        session.add(task)
        await session.flush()
        await session.refresh(task)
        return task

    @staticmethod
    async def get_by_id(session: AsyncSession, task_id: UUID) -> Optional[Task]:
        result = await session.execute(
            select(Task).where(Task.id == task_id)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def list_tasks(
        session: AsyncSession,
        task_type: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> List[Task]:
        stmt = select(Task).order_by(desc(Task.created_at))
        if task_type:
            stmt = stmt.where(Task.task_type == task_type)
        if status:
            stmt = stmt.where(Task.status == status)
        stmt = stmt.offset(offset).limit(limit)
        result = await session.execute(stmt)
        return list(result.scalars().all())
