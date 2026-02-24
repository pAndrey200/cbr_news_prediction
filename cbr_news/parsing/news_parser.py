import calendar
import logging
import time
from datetime import datetime
from pathlib import Path

import httpx
import pandas as pd
from bs4 import BeautifulSoup
from hydra.utils import get_original_cwd

logger = logging.getLogger(__name__)

NEWS_TYPES = {
    "events": "https://www.cbr.ru/press/event/?id={}",
    "press": "https://www.cbr.ru/press/pr/?file={}",
}

BATCH_SIZE = 100
START_YEAR = 2013
CONTENT_SELECTORS = [".landing-text", ".news-content", "article", ".content"]


class CBRNewsParser:
    def __init__(self, config, use_database=True):
        self.config = config
        self.use_database = use_database

        try:
            project_root = Path(get_original_cwd())
        except (ValueError, AttributeError):
            project_root = Path.cwd()
        self.data_dir = project_root / "data"
        self.data_dir.mkdir(exist_ok=True)

        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

        if self.use_database:
            try:
                from cbr_news.database.db import get_db_session
                from cbr_news.database.repository import NewsRepository

                self.get_db_session = get_db_session
                self.news_repo = NewsRepository
            except Exception as e:
                logger.error(f"Не удалось подключиться к БД: {e}")
                self.use_database = False

    def _get_existing_links(self, links: list) -> set:
        if not self.use_database or not links:
            return set()
        try:
            with self.get_db_session() as session:
                return self.news_repo.get_existing_links(session, links)
        except Exception as e:
            logger.error(f"Ошибка проверки ссылок в БД: {e}")
            return set()

    def _save_batch_to_db(self, batch: list):
        if not self.use_database or not batch:
            return
        try:
            with self.get_db_session() as session:
                for item in batch:
                    date_obj = self._parse_date(item["date"])
                    if not date_obj:
                        continue
                    self.news_repo.create_or_update(session, {
                        "date": date_obj,
                        "link": item["link"],
                        "title": item["title"],
                        "content": item.get("release", ""),
                        "news_type": item["news_type"],
                    })
                logger.info(f"Сохранена пачка из {len(batch)} новостей в БД")
        except Exception as e:
            logger.error(f"Ошибка сохранения пачки в БД: {e}")

    def _parse_date(self, date_str: str):
        if not date_str:
            return None
        try:
            if "." in date_str and len(date_str.split(".")) == 3:
                parts = date_str.split(".")
                if len(parts[0]) <= 2 and len(parts[1]) <= 2 and len(parts[2]) == 4:
                    return datetime.strptime(date_str, "%d.%m.%Y").date()

            parts = date_str.split()
            if len(parts) >= 3:
                day = int(parts[0])
                month_ru = parts[1].lower()[:3]
                year = int(parts[2])
                months = {
                    "янв": 1, "фев": 2, "мар": 3, "апр": 4, "мая": 5, "май": 5,
                    "июн": 6, "июл": 7, "авг": 8, "сен": 9, "окт": 10, "ноя": 11, "дек": 12,
                }
                month = months.get(month_ru, 1)
                return datetime(year, month, day).date()
        except Exception as e:
            logger.warning(f"Не удалось распарсить дату '{date_str}': {e}")
        return None

    def _format_date_from_iso(self, date_str: str) -> str:
        if not date_str:
            return ""
        try:
            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            return f"{dt.day:02d}.{dt.month:02d}.{dt.year}"
        except Exception:
            return date_str

    def _fetch_news_content(self, url: str) -> str:
        try:
            response = httpx.get(url, headers=self.headers, timeout=30)
            tree = BeautifulSoup(response.text, "html.parser")

            content = None
            for selector in CONTENT_SELECTORS:
                content = tree.select_one(selector)
                if content:
                    break

            if not content:
                content = (
                    tree.find("main")
                    or tree.find("article")
                    or tree.find("div", class_=lambda x: x and "content" in x.lower())
                )

            if content:
                for tag in content.find_all(["script", "style"]):
                    tag.decompose()
                return content.get_text(separator=" ", strip=True)

            logger.warning(f"Не найден контент: {url}")
            return ""
        except Exception as e:
            logger.error(f"Ошибка получения контента {url}: {e}")
            return ""

    def _collect_links_for_month(self, year: int, month: int) -> list:
        base_url = "https://www.cbr.ru/news/eventandpress/"
        last_day = calendar.monthrange(year, month)[1]
        date_from = f"{year}-{month:02d}-01"
        date_to = f"{year}-{month:02d}-{last_day:02d}"
        page = 0
        links = []

        while True:
            try:
                params = {
                    "page": page,
                    "IsEng": "false",
                    "type": "100",
                    "dateFrom": date_from,
                    "dateTo": date_to,
                    "Tid": "",
                    "vol": "",
                    "phrase": "",
                }
                response = httpx.get(base_url, params=params, headers=self.headers, timeout=30)

                if response.status_code != 200:
                    break

                try:
                    data = response.json()
                except Exception:
                    break

                if not data:
                    break

                for item in data:
                    tbl_type = item.get("TBLType")
                    if tbl_type not in NEWS_TYPES:
                        continue

                    doc_htm = item.get("doc_htm")
                    if not doc_htm:
                        continue

                    url = NEWS_TYPES[tbl_type].format(doc_htm)
                    date_str = self._format_date_from_iso(item.get("DT", ""))
                    title = item.get("name_doc", "")
                    links.append({
                        "link": url,
                        "date": date_str,
                        "title": title,
                        "news_type": tbl_type,
                    })

                page += 1
                time.sleep(0.5)

            except httpx.TimeoutException:
                logger.warning(f"Таймаут: {year}-{month:02d}, page={page}")
                time.sleep(5)
                continue
            except Exception as e:
                logger.error(f"Ошибка: {year}-{month:02d}, page={page}: {e}")
                break

        logger.info(f"{year}-{month:02d}: собрано {len(links)} ссылок ({page} стр.)")
        return links

    def _parse_month(self, year: int, month: int) -> list:
        links = self._collect_links_for_month(year, month)
        if not links:
            return []

        existing = self._get_existing_links([item["link"] for item in links])
        new_links = [item for item in links if item["link"] not in existing]

        if existing:
            logger.info(f"{year}-{month:02d}: пропущено {len(existing)} существующих")

        if not new_links:
            logger.info(f"{year}-{month:02d}: нет новых ссылок")
            return []

        logger.info(f"{year}-{month:02d}: парсинг {len(new_links)} новых новостей")

        month_data = []
        batch = []

        for i, item in enumerate(new_links):
            text = self._fetch_news_content(item["link"])
            record = {
                "date": item["date"],
                "link": item["link"],
                "title": item["title"],
                "news_type": item["news_type"],
                "release": text,
            }
            month_data.append(record)
            batch.append(record)

            if len(batch) >= BATCH_SIZE:
                self._save_batch_to_db(batch)
                logger.info(f"{year}-{month:02d}: сохранено {i + 1}/{len(new_links)}")
                batch = []

            time.sleep(0.5)

        if batch:
            self._save_batch_to_db(batch)

        logger.info(f"{year}-{month:02d}: завершено, {len(month_data)} новостей")
        return month_data

    def get_all_news(self) -> pd.DataFrame:
        logger.info("Начало парсинга новостей...")

        now = datetime.now()
        all_data = []

        for year in range(START_YEAR, now.year + 1):
            last_month = now.month if year == now.year else 12
            for month in range(1, last_month + 1):
                month_data = self._parse_month(year, month)
                all_data.extend(month_data)

        df = pd.DataFrame(all_data, columns=["date", "link", "title", "news_type", "release"])

        csv_path = self.data_dir / "raw-cbr-news.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8")
        logger.info(f"Сохранено {len(df)} новых новостей в {csv_path}")

        return df
