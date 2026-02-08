import json
import locale
import logging
import time
from datetime import datetime
from pathlib import Path

import httpx
import pandas as pd
from bs4 import BeautifulSoup
from hydra.utils import get_original_cwd

logger = logging.getLogger(__name__)


class CBRDataParser:
    """Парсер данных Банка России"""

    def __init__(self, config, use_database=True):
        self.config = config
        self.use_database = use_database

        try:
            project_root = Path(get_original_cwd())
        except (ValueError, AttributeError):
            project_root = Path.cwd()
        self.data_dir = project_root / "data"
        self.data_dir.mkdir(exist_ok=True)
        locale.setlocale(locale.LC_ALL, "ru_RU.UTF-8")
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        if self.use_database:
            try:
                from cbr_news.database import get_db_session
                from cbr_news.repository import (
                    CurrencyRateRepository,
                    InflationRepository,
                    KeyRateRepository,
                    NewsRepository,
                    PreciousMetalRepository,
                    ReserveRepository,
                    RuoniaRepository,
                )
                self.get_db_session = get_db_session
                self.news_repo = NewsRepository
                self.key_rate_repo = KeyRateRepository
                self.currency_repo = CurrencyRateRepository
                self.inflation_repo = InflationRepository
                self.ruonia_repo = RuoniaRepository
                self.metal_repo = PreciousMetalRepository
                self.reserve_repo = ReserveRepository
                logger.info("Инициализировано подключение к базе данных")
            except Exception as e:
                logger.error(f"Не удалось инициализировать подключение к БД: {e}")
                self.use_database = False

    def _parse_date_to_obj(self, date_str: str):
        """Преобразование строки даты в объект date."""
        if not date_str:
            return None
        try:
            # Если дата в формате DD.MM.YYYY
            if "." in date_str and len(date_str.split(".")) == 3:
                parts = date_str.split(".")
                if len(parts[0]) <= 2 and len(parts[1]) <= 2 and len(parts[2]) == 4:
                    return datetime.strptime(date_str, "%d.%m.%Y").date()

            # Парсим формат "DD MMM YYYY" (русский)
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

    def _save_news_to_db(self, df: pd.DataFrame, news_type: str = "press_release"):
        """Сохранение новостей в базу данных."""
        if not self.use_database:
            logger.debug(f"Сохранение в БД отключено (use_database={self.use_database})")
            return

        logger.info(f"Начало сохранения {len(df)} новостей в БД (тип: {news_type})...")
        try:
            with self.get_db_session() as session:
                for _, row in df.iterrows():
                    date_obj = self._parse_date_to_obj(row.get("date", ""))
                    if not date_obj:
                        continue

                    news_data = {
                        "date": date_obj,
                        "link": row["link"],
                        "title": row["title"],
                        "content": row.get("release", ""),
                        "news_type": news_type,
                    }
                    self.news_repo.create_or_update(session, news_data)
                logger.info(f"Сохранено {len(df)} новостей в базу данных")
        except Exception as e:
            logger.error(f"Ошибка при сохранении новостей в БД: {e}")

    def _save_key_rates_to_db(self, df: pd.DataFrame):
        """Сохранение ключевой ставки в базу данных."""
        if not self.use_database:
            logger.debug(f"Сохранение в БД отключено (use_database={self.use_database})")
            return

        logger.info(f"Начало сохранения {len(df)} записей ключевой ставки в БД...")
        try:
            with self.get_db_session() as session:
                for _, row in df.iterrows():
                    date_obj = self._parse_date_to_obj(row["date"])
                    if not date_obj:
                        continue
                    self.key_rate_repo.create_or_update(session, date_obj, float(row["rate"]))
                logger.info(f"Сохранено {len(df)} записей ключевой ставки в БД")
        except Exception as e:
            logger.error(f"Ошибка при сохранении ключевой ставки в БД: {e}")

    def _save_currency_rates_to_db(self, df: pd.DataFrame, currency_code: str):
        """Сохранение курсов валют в базу данных."""
        if not self.use_database:
            logger.debug(f"Сохранение в БД отключено (use_database={self.use_database})")
            return

        logger.info(f"Начало сохранения {len(df)} записей курса {currency_code} в БД...")
        try:
            with self.get_db_session() as session:
                for _, row in df.iterrows():
                    date_obj = self._parse_date_to_obj(row["date"])
                    if not date_obj:
                        continue
                    rate = float(row[currency_code.lower()])
                    self.currency_repo.create_or_update(session, date_obj, currency_code, rate)
                logger.info(f"Сохранено {len(df)} записей курса {currency_code} в БД")
        except Exception as e:
            logger.error(f"Ошибка при сохранении курса {currency_code} в БД: {e}")

    def _save_inflation_to_db(self, df: pd.DataFrame):
        """Сохранение инфляции в базу данных."""
        if not self.use_database:
            return

        try:
            with self.get_db_session() as session:
                for _, row in df.iterrows():
                    date_obj = self._parse_date_to_obj(row["date_inflation"])
                    if not date_obj:
                        continue
                    self.inflation_repo.create_or_update(session, date_obj, float(row["inflation"]))
                logger.info(f"Сохранено {len(df)} записей инфляции в БД")
        except Exception as e:
            logger.error(f"Ошибка при сохранении инфляции в БД: {e}")

    def _save_ruonia_to_db(self, df: pd.DataFrame):
        """Сохранение ставки RUONIA в базу данных."""
        if not self.use_database:
            return

        try:
            with self.get_db_session() as session:
                for _, row in df.iterrows():
                    date_obj = self._parse_date_to_obj(row["date"])
                    if not date_obj:
                        continue
                    self.ruonia_repo.create_or_update(session, date_obj, float(row["ruonia"]))
                logger.info(f"Сохранено {len(df)} записей RUONIA в БД")
        except Exception as e:
            logger.error(f"Ошибка при сохранении RUONIA в БД: {e}")

    def _save_metals_to_db(self, df: pd.DataFrame):
        """Сохранение драгметаллов в базу данных."""
        if not self.use_database:
            return

        try:
            with self.get_db_session() as session:
                for _, row in df.iterrows():
                    date_obj = self._parse_date_to_obj(row["date"])
                    if not date_obj:
                        continue
                    for metal in ["gold", "silver", "platinum", "palladium"]:
                        if metal in row and pd.notna(row[metal]):
                            self.metal_repo.create_or_update(session, date_obj, metal, float(row[metal]))
                logger.info(f"Сохранено {len(df)} записей драгметаллов в БД")
        except Exception as e:
            logger.error(f"Ошибка при сохранении драгметаллов в БД: {e}")

    def _save_reserves_to_db(self, df: pd.DataFrame):
        """Сохранение резервов в базу данных."""
        if not self.use_database:
            return

        try:
            with self.get_db_session() as session:
                for _, row in df.iterrows():
                    date_obj = self._parse_date_to_obj(row["date_reserves"])
                    if not date_obj:
                        continue
                    self.reserve_repo.create_or_update(
                        session,
                        date_obj,
                        reserves_corset=float(row["reserves_corset"]) if pd.notna(row.get("reserves_corset")) else None,
                        reserves_avg=float(row["reserves_avg"]) if pd.notna(row.get("reserves_avg")) else None,
                        reserves_accounts=float(row["reserves_accounts"]) if pd.notna(row.get("reserves_accounts")) else None,
                    )
                logger.info(f"Сохранено {len(df)} записей резервов в БД")
        except Exception as e:
            logger.error(f"Ошибка при сохранении резервов в БД: {e}")

    def get_press_releases(self) -> pd.DataFrame:
        """Получение пресс-релизов Банка России"""
        URL = (
            "http://www.cbr.ru/Crosscut/NewsList/LoadMore/84035?intOffset=0&extOffset="
        )
        offset = 0
        data = []

        logger.info("Начало парсинга пресс-релизов Банка России...")

        while True:
            try:
                page = httpx.get(URL + str(offset), timeout=30)

                if len(page.text.strip()) == 0:
                    logger.info("Больше данных нет, завершение парсинга...")
                    break

                tree = BeautifulSoup(page.text, "html.parser")
                releases = tree.select(".previews_day")

                if not releases:
                    logger.info("Не найдено релизов на странице, завершение...")
                    break

                for item in releases:
                    date = item.select_one(".previews_day-date")
                    links = item.select("a")

                    for link in links:
                        data_item = []
                        if date and date.text:
                            data_item.append(date.text.strip())
                        else:
                            data_item.append("")

                        if "href" in link.attrs:
                            href = link["href"]
                            if href.startswith("/"):
                                data_item.append("http://www.cbr.ru" + href)
                            else:
                                data_item.append(href)
                        else:
                            data_item.append("")

                        data_item.append(link.text.strip())
                        data.append(data_item)

                logger.info(f"Получено {offset + len(releases)} ссылок на релизы")

                offset += 10
                time.sleep(1)

            except httpx.TimeoutException:
                logger.warning(f"Таймаут при offset={offset}, повторная попытка...")
                time.sleep(5)
                continue
            except Exception as e:
                logger.error(f"Ошибка при парсинге: {e}")
                break

        logger.info("Получение текстов релизов...")
        for i, (_, link, _) in enumerate(data):
            if not link:
                data[i].append("")
                continue

            try:
                response = httpx.get(link, timeout=30)
                tree = BeautifulSoup(response.text, "html.parser")
                release_div = tree.find("div", {"class": "landing-text"})

                if release_div:
                    for script in release_div.find_all(["script", "style"]):
                        script.decompose()

                    text = release_div.get_text(separator=" ", strip=True)
                    data[i].append(text)
                else:
                    data[i].append("")

                logger.info(f"Получен текст релиза {i+1}/{len(data)}")

            except Exception as e:
                logger.error(f"Ошибка при получении текста релиза {link}: {e}")
                data[i].append("")

            time.sleep(0.5)

        df = pd.DataFrame(data, columns=["date", "link", "title", "release"])

        raw_path = self.data_dir / "raw-cbr-press-releases.csv"
        df.to_csv(raw_path, index=False, encoding="utf-8")
        logger.info(f"Сохранено {len(df)} релизов в {raw_path}")

        self._save_news_to_db(df, news_type="press_release")

        return df

    def get_news_events(self) -> pd.DataFrame:
        """Получение новостей типа events с сайта Банка России"""
        base_url = "https://www.cbr.ru/news/eventandpress/"
        page = 0
        all_events = []
        
        logger.info("Начало парсинга новостей типа events Банка России...")
        
        while page < 3:
            try:
                params = {
                    "page": page,
                    "IsEng": "false",
                    "type": "100",
                    "dateFrom": "",
                    "dateTo": "",
                    "Tid": "",
                    "vol": "",
                    "phrase": ""
                }
                
                response = httpx.get(base_url, params=params, headers=self.headers, timeout=30)
                
                if response.status_code != 200:
                    logger.warning(f"Ошибка HTTP {response.status_code} на странице {page}")
                    break
                
                try:
                    events_data = response.json()
                except json.JSONDecodeError:
                    logger.warning(f"Не удалось распарсить JSON на странице {page}")
                    break
                
                if not events_data or len(events_data) == 0:
                    logger.info(f"Больше данных нет на странице {page}, завершение парсинга...")
                    break

                events_on_page = [event for event in events_data if event.get("TBLType") == "events"]

                if events_on_page:
                    all_events.extend(events_on_page)
                    logger.info(f"Получено {len(events_on_page)} событий со страницы {page} (всего: {len(all_events)})")
                else:
                    logger.info(f"Не найдено событий типа 'events' на странице {page}")

                if len(events_data) == 0:
                    logger.info(f"Больше данных нет на странице {page}, завершение парсинга...")
                    break
                
                page += 1
                time.sleep(1)
                
            except httpx.TimeoutException:
                logger.warning(f"Таймаут при page={page}, повторная попытка...")
                time.sleep(5)
                continue
            except Exception as e:
                logger.error(f"Ошибка при парсинге страницы {page}: {e}")
                break
        
        logger.info(f"Всего получено {len(all_events)} событий. Начало получения текстов новостей...")
        
        data = []
        for i, event in enumerate(all_events):
            try:
                doc_htm = event.get("doc_htm")
                if not doc_htm:
                    logger.warning(f"Пропущено событие {i+1}: отсутствует doc_htm")
                    continue

                news_url = f"https://www.cbr.ru/press/event/?id={doc_htm}"
                date_str = event.get("DT", "")
                if date_str:
                    try:
                        from datetime import datetime
                        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                        months_ru = {
                            1: "янв", 2: "фев", 3: "мар", 4: "апр", 5: "мая",
                            6: "июн", 7: "июл", 8: "авг", 9: "сен", 10: "окт",
                            11: "ноя", 12: "дек"
                        }
                        month_ru = months_ru.get(dt.month, "янв")
                        date_str = f"{dt.day} {month_ru} {dt.year}"
                    except:
                        pass

                title = event.get("name_doc", "")

                try:
                    response = httpx.get(news_url, headers=self.headers, timeout=30)
                    tree = BeautifulSoup(response.text, "html.parser")

                    content = None
                    for selector in [".landing-text", ".news-content", "article", ".content"]:
                        content = tree.select_one(selector)
                        if content:
                            break

                    if not content:
                        content = tree.find("main") or tree.find("article") or tree.find("div", class_=lambda x: x and "content" in x.lower())
                    
                    if content:
                        for script in content.find_all(["script", "style"]):
                            script.decompose()

                        text = content.get_text(separator=" ", strip=True)
                    else:
                        text = ""
                        logger.warning(f"Не удалось найти контент для новости {doc_htm}")
                    
                except Exception as e:
                    logger.error(f"Ошибка при получении текста новости {news_url}: {e}")
                    text = ""
                
                data.append([date_str, news_url, title, text])
                logger.info(f"Получен текст новости {i+1}/{len(all_events)}: {title[:50]}...")
                
            except Exception as e:
                logger.error(f"Ошибка при обработке события {i+1}: {e}")
                continue
            
            time.sleep(0.5)
        
        df = pd.DataFrame(data, columns=["date", "link", "title", "release"])

        raw_path = self.data_dir / "raw-cbr-news.csv"
        df.to_csv(raw_path, index=False, encoding="utf-8")
        logger.info(f"Сохранено {len(df)} новостей в {raw_path}")

        self._save_news_to_db(df, news_type="event")

        return df

    def get_key_rate(self) -> pd.DataFrame:
        """Получение исторических данных по ключевой ставке (упрощенный вариант)"""
        logger.info("Получение данных по ключевой ставке...")

        base_url = "https://cbr.ru/hd_base/KeyRate"
        response = httpx.get(base_url, headers=self.headers, timeout=10)

        soup = BeautifulSoup(response.text, "html.parser")
        scripts = soup.find_all("script")

        target_script = None
        for script in scripts:
            if script.string and '"data":[' in script.string:
                target_script = script.string
                break

        script_str = target_script

        dates_str = script_str.split(',"categories":["')[1].split("]")[0]
        values_str = script_str.split(',"data":[')[1].split("]")[0]

        dates_array = [s.strip('"') for s in dates_str.split(",")]
        values_array = [float(s) for s in values_str.split(",")]

        if len(dates_array) != len(values_array):
            min_len = min(len(dates_array), len(values_array))
            dates_array = dates_array[:min_len]
            values_array = values_array[:min_len]
            logger.warning(f"Длины не совпадают, обрезано до {min_len}")

        df = pd.DataFrame({"date": dates_array, "rate": values_array})

        key_rate_path = self.data_dir / "key-rates-cbr.csv"
        df.to_csv(key_rate_path, index=False)
        logger.info(f"Сохранено {len(df)} записей по ключевой ставке")

        self._save_key_rates_to_db(df)

        return df

    def get_inflation(self) -> pd.DataFrame:
        """Получение данных по инфляции (упрощенный вариант)"""
        logger.info("Получение данных по инфляции...")

        base_url = "https://www.cbr.ru/hd_base/infl/?UniDbQuery.Posted=True&UniDbQuery.From=17.09.2013"
        response = httpx.get(base_url, headers=self.headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        table = soup.find("table", attrs={"class": "data"})
        if not table:
            raise ValueError("Таблица с данными не найдена")

        rows = table.find_all("tr")
        data = []

        for row in rows:
            cols = row.find_all("td")
            cols = [c.text.strip() for c in cols]
            if len(cols) == 4:
                date_str = cols[0]
                inflation_str = cols[2].replace(",", ".")

                try:
                    inflation_value = float(inflation_str)
                    data.append([date_str, inflation_value])
                except ValueError:
                    continue

        df = pd.DataFrame(data, columns=["date_inflation", "inflation"])

        inflation_path = self.data_dir / "inflation-cbr.csv"
        df.to_csv(inflation_path, index=False)
        logger.info(f"Сохранено {len(df)} записей по инфляции")

        self._save_inflation_to_db(df)

        return df

    def _get_currency_dynamics(self, val_nm_rq: str, column_name: str, from_date: str = "01.09.2013") -> pd.DataFrame:
        """Общий метод: динамика курса валюты по коду ЦБ (VAL_NM_RQ)."""
        base_url = (
            "https://www.cbr.ru/currency_base/dynamics/?UniDbQuery.Posted=True&"
            "UniDbQuery.so=1&UniDbQuery.mode=1&UniDbQuery.date_req1=&"
            "UniDbQuery.date_req2=&UniDbQuery.VAL_NM_RQ={}&UniDbQuery.From={}"
        ).format(val_nm_rq, from_date)
        response = httpx.get(base_url, headers=self.headers, timeout=15)
        soup = BeautifulSoup(response.text, "html.parser")
        table = soup.find("table", attrs={"class": "data"})
        if not table:
            logger.warning("Таблица с курсами не найдена для %s", val_nm_rq)
            return pd.DataFrame(columns=["date", column_name])
        rows = table.find_all("tr")
        data = []
        for row in rows:
            cols = row.find_all("td")
            cols = [c.text.strip() for c in cols]
            if len(cols) == 3:
                date_str = cols[0]
                rate_str = cols[2].replace(",", ".")
                try:
                    data.append([date_str, float(rate_str)])
                except ValueError:
                    continue
        return pd.DataFrame(data, columns=["date", column_name])

    def get_usd_rate(self) -> pd.DataFrame:
        """Получение курса USD."""
        logger.info("Получение курса USD...")
        df = self._get_currency_dynamics("R01235", "usd")
        path = self.data_dir / "cur-usd-cbr.csv"
        df.to_csv(path, index=False)
        logger.info(f"Сохранено {len(df)} записей по курсу USD")
        self._save_currency_rates_to_db(df, "USD")
        return df

    def get_eur_rate(self) -> pd.DataFrame:
        """Получение курса евро (EUR)."""
        logger.info("Получение курса EUR...")
        df = self._get_currency_dynamics("R01239", "eur")
        path = self.data_dir / "cur-eur-cbr.csv"
        df.to_csv(path, index=False)
        logger.info(f"Сохранено {len(df)} записей по курсу EUR")
        self._save_currency_rates_to_db(df, "EUR")
        return df

    def get_cny_rate(self) -> pd.DataFrame:
        """Получение курса китайского юаня (CNY)."""
        logger.info("Получение курса CNY...")
        df = self._get_currency_dynamics("R01375", "cny")
        path = self.data_dir / "cur-cny-cbr.csv"
        df.to_csv(path, index=False)
        logger.info(f"Сохранено {len(df)} записей по курсу CNY")
        self._save_currency_rates_to_db(df, "CNY")
        return df

    def get_ruonia(self) -> pd.DataFrame:
        """Получение ставки RUONIA (динамика)."""
        logger.info("Получение ставки RUONIA...")
        base_url = "https://www.cbr.ru/hd_base/ruonia/dynamics/"
        # Запрашиваем большой диапазон для истории
        params = {"FromDate": "01.09.2013", "ToDate": "31.12.2030", "posted": "True"}
        try:
            response = httpx.get(base_url, params=params, headers=self.headers, timeout=15)
            soup = BeautifulSoup(response.text, "html.parser")
            table = soup.find("table", attrs={"class": "data"})
            if not table:
                logger.warning("Таблица RUONIA не найдена")
                return pd.DataFrame(columns=["date", "ruonia"])
            rows = table.find_all("tr")
            header = [th.get_text(strip=True) for th in rows[0].find_all(["th", "td"])] if rows else []
            data = []
            for row in rows[1:]:
                cols = row.find_all("td")
                cols = [c.get_text(strip=True) for c in cols]
                if len(cols) >= 2:
                    date_str = cols[0]
                    rate_str = cols[1].replace(",", ".")
                    try:
                        data.append([date_str, float(rate_str)])
                    except ValueError:
                        continue
            df = pd.DataFrame(data, columns=["date", "ruonia"])
        except Exception as e:
            logger.warning("Ошибка парсинга RUONIA: %s", e)
            df = pd.DataFrame(columns=["date", "ruonia"])
        path = self.data_dir / "ruonia-cbr.csv"
        df.to_csv(path, index=False)
        logger.info(f"Сохранено {len(df)} записей RUONIA")
        if len(df) > 0:
            self._save_ruonia_to_db(df)
        return df

    def get_precious_metals(self) -> pd.DataFrame:
        """Получение учётных цен на драгоценные металлы (золото, серебро, платина, палладий), руб/грамм."""
        logger.info("Получение учётных цен на драгоценные металлы...")
        base_url = "https://www.cbr.ru/hd_base/metall/metall_base_new/"
        params = {}  # при необходимости добавить диапазон дат
        try:
            response = httpx.get(base_url, params=params, headers=self.headers, timeout=15)
            soup = BeautifulSoup(response.text, "html.parser")
            table = soup.find("table", attrs={"class": "data"})
            if not table:
                logger.warning("Таблица драгметаллов не найдена")
                return pd.DataFrame(columns=["date", "gold", "silver", "platinum", "palladium"])
            rows = table.find_all("tr")
            data = []
            for row in rows[1:]:
                cols = row.find_all("td")
                if len(cols) < 5:
                    continue
                date_str = cols[0].get_text(strip=True)
                vals = []
                for c in cols[1:5]:
                    s = c.get_text(strip=True).replace("\xa0", "").replace(" ", "").replace(",", ".")
                    try:
                        vals.append(float(s))
                    except ValueError:
                        vals.append(float("nan"))
                data.append([date_str] + vals)
            df = pd.DataFrame(data, columns=["date", "gold", "silver", "platinum", "palladium"])
        except Exception as e:
            logger.warning("Ошибка парсинга драгметаллов: %s", e)
            df = pd.DataFrame(columns=["date", "gold", "silver", "platinum", "palladium"])
        path = self.data_dir / "metall-cbr.csv"
        df.to_csv(path, index=False)
        logger.info(f"Сохранено {len(df)} записей по драгметаллам")
        if len(df) > 0:
            self._save_metals_to_db(df)
        return df

    def get_reserves(self) -> pd.DataFrame:
        """Получение показателей обязательных резервов кредитных организаций (млрд руб.)."""
        logger.info("Получение данных по обязательным резервам...")
        base_url = "https://www.cbr.ru/hd_base/RReserves/"
        try:
            response = httpx.get(base_url, headers=self.headers, timeout=15)
            soup = BeautifulSoup(response.text, "html.parser")
            table = soup.find("table", attrs={"class": "data"})
            if not table:
                logger.warning("Таблица резервов не найдена")
                return pd.DataFrame(columns=["date_reserves", "reserves_avg", "reserves_accounts"])
            rows = table.find_all("tr")
            data = []
            for row in rows[1:]:
                cols = row.find_all("td")
                if len(cols) < 4:
                    continue
                date_str = cols[0].get_text(strip=True)
                vals = []
                for c in cols[1:4]:
                    s = c.get_text(strip=True).replace("\xa0", "").replace(" ", "").replace(",", ".")
                    if s == "—" or not s:
                        vals.append(float("nan"))
                    else:
                        try:
                            vals.append(float(s))
                        except ValueError:
                            vals.append(float("nan"))
                data.append([date_str] + vals)
            df = pd.DataFrame(
                data,
                columns=["date_reserves", "reserves_corset", "reserves_avg", "reserves_accounts"],
            )
        except Exception as e:
            logger.warning("Ошибка парсинга резервов: %s", e)
            df = pd.DataFrame(columns=["date_reserves", "reserves_avg", "reserves_accounts"])
        path = self.data_dir / "reserves-cbr.csv"
        df.to_csv(path, index=False)
        logger.info(f"Сохранено {len(df)} записей по резервам")
        if len(df) > 0:
            self._save_reserves_to_db(df)
        return df

    def preprocess_data(self, df_releases: pd.DataFrame) -> pd.DataFrame:
        """Предобработка данных (упрощенная)"""
        logger.info("Начало предобработки данных...")

        def parse_date(date_str):
            try:
                if "." in date_str and len(date_str.split(".")) == 3:
                    parts = date_str.split(".")
                    if len(parts[0]) <= 2 and len(parts[1]) <= 2 and len(parts[2]) == 4:
                        return date_str

                if "T" in date_str or (len(date_str) >= 10 and date_str[4] == "-" and date_str[7] == "-"):
                    try:
                        from datetime import datetime
                        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                        return f"{dt.day:02d}.{dt.month:02d}.{dt.year}"
                    except:
                        pass

                parts = date_str.split()
                if len(parts) >= 3:
                    day = parts[0]
                    month_ru = parts[1].lower()[:3]
                    year = parts[2]

                    months = {
                        "янв": "01", "jan": "01",
                        "фев": "02", "feb": "02",
                        "мар": "03", "mar": "03",
                        "апр": "04", "apr": "04",
                        "мая": "05", "май": "05", "may": "05",
                        "июн": "06", "jun": "06",
                        "июл": "07", "jul": "07",
                        "авг": "08", "aug": "08",
                        "сен": "09", "sep": "09",
                        "окт": "10", "oct": "10",
                        "ноя": "11", "nov": "11",
                        "дек": "12", "dec": "12",
                    }

                    month = months.get(month_ru, "01")
                    return f"{day}.{month}.{year}"
            except:
                pass
            return None

        df_releases["date_parsed"] = df_releases["date"].apply(parse_date)
        df_releases = df_releases.dropna(subset=["date_parsed"])

        df_key_rates = pd.read_csv(
            self.data_dir / "key-rates-cbr.csv", dtype={"date": str, "rate": float}
        )
        df_usd = pd.read_csv(
            self.data_dir / "cur-usd-cbr.csv", dtype={"date": str, "usd": float}
        )
        df_inflation = pd.read_csv(
            self.data_dir / "inflation-cbr.csv",
            dtype={"date_inflation": str, "inflation": float},
        )
        # Дополнительные данные (могут отсутствовать)
        dfs_extra = {}
        for name, fname, date_col, rename_map in [
            ("eur", "cur-eur-cbr.csv", "date", {"date": "date_eur", "eur": "eur"}),
            ("cny", "cur-cny-cbr.csv", "date", {"date": "date_cny", "cny": "cny"}),
            ("ruonia", "ruonia-cbr.csv", "date", {"date": "date_ruonia", "ruonia": "ruonia"}),
        ]:
            p = self.data_dir / fname
            if p.exists():
                try:
                    df = pd.read_csv(p, dtype=str)
                    for c in df.columns:
                        if c != date_col and c in df.columns:
                            df[c] = pd.to_numeric(df[c], errors="coerce")
                    df = df.rename(columns=rename_map)
                    dfs_extra[name] = df
                except Exception as e:
                    logger.warning("Не удалось загрузить %s: %s", fname, e)
        df_metall = None
        if (self.data_dir / "metall-cbr.csv").exists():
            try:
                df_metall = pd.read_csv(self.data_dir / "metall-cbr.csv", dtype=str)
                for c in ["gold", "silver", "platinum", "palladium"]:
                    if c in df_metall.columns:
                        df_metall[c] = pd.to_numeric(df_metall[c], errors="coerce")
                df_metall = df_metall.rename(columns={"date": "date_metall"})
            except Exception as e:
                logger.warning("Не удалось загрузить metall-cbr.csv: %s", e)
        df_reserves = None
        if (self.data_dir / "reserves-cbr.csv").exists():
            try:
                df_reserves = pd.read_csv(self.data_dir / "reserves-cbr.csv", dtype=str)
                for c in ["reserves_avg", "reserves_accounts"]:
                    if c in df_reserves.columns:
                        df_reserves[c] = pd.to_numeric(df_reserves[c], errors="coerce")
            except Exception as e:
                logger.warning("Не удалось загрузить reserves-cbr.csv: %s", e)

        def get_month(date_str):
            try:
                parts = date_str.split(".")
                return f"{parts[1]}.{parts[2]}"
            except:
                return ""

        df_releases["month"] = df_releases["date_parsed"].apply(get_month)

        # Переименовываем колонки перед merge, чтобы избежать конфликтов
        df_usd = df_usd.rename(columns={"date": "date_usd"})
        df_key_rates = df_key_rates.rename(columns={"date": "date_keyrate", "rate": "key_rate"})

        df_merged = pd.merge(
            df_releases,
            df_inflation,
            left_on="month",
            right_on="date_inflation",
            how="left",
        )

        df_merged = pd.merge(
            df_merged, df_usd, left_on="date_parsed", right_on="date_usd", how="left"
        )

        df_merged = pd.merge(
            df_merged,
            df_key_rates,
            left_on="date_parsed",
            right_on="date_keyrate",
            how="left",
        )
        for name, df_extra in dfs_extra.items():
            date_col = f"date_{name}"
            if date_col not in df_extra.columns:
                continue
            df_merged = pd.merge(
                df_merged,
                df_extra,
                left_on="date_parsed",
                right_on=date_col,
                how="left",
                suffixes=("", "_drop"),
            )
            df_merged = df_merged.drop(columns=[date_col], errors="ignore")
        if df_metall is not None and "date_metall" in df_metall.columns:
            df_merged = pd.merge(
                df_merged,
                df_metall,
                left_on="date_parsed",
                right_on="date_metall",
                how="left",
            )
            df_merged = df_merged.drop(columns=["date_metall"], errors="ignore")
        if df_reserves is not None and "date_reserves" in df_reserves.columns:
            df_merged = pd.merge(
                df_merged,
                df_reserves,
                left_on="date_parsed",
                right_on="date_reserves",
                how="left",
            )
            df_merged = df_merged.drop(columns=["date_reserves"], errors="ignore")

        df_merged = df_merged.drop(columns=["date_usd", "date_keyrate", "date"], errors="ignore")
        drop_drop = [c for c in df_merged.columns if c.endswith("_drop")]
        df_merged = df_merged.drop(columns=drop_drop, errors="ignore")
        df_merged = df_merged.rename(columns={"date_parsed": "date"})

        columns_to_keep = [
            "date", "link", "title", "release",
            "inflation", "usd", "eur", "cny", "key_rate", "ruonia",
            "gold", "silver", "platinum", "palladium",
            "reserves_avg", "reserves_accounts", "reserves_corset",
        ]
        df_merged = df_merged[
            [col for col in columns_to_keep if col in df_merged.columns]
        ]

        if df_merged.columns.duplicated().any():
            df_merged = df_merged.loc[:, ~df_merged.columns.duplicated()]

        for col in ["usd", "inflation", "eur", "cny", "ruonia", "gold", "silver", "platinum", "palladium", "reserves_avg", "reserves_accounts", "reserves_corset"]:
            if col in df_merged.columns:
                df_merged[col] = df_merged[col].ffill().bfill()

        # Проверяем, что колонка date существует и уникальна перед сортировкой
        if "date" not in df_merged.columns:
            raise ValueError("Колонка 'date' не найдена в DataFrame")
        if df_merged.columns.tolist().count("date") > 1:
            # Если все еще есть дубликаты, оставляем только первую
            df_merged = df_merged.loc[:, ~df_merged.columns.duplicated()]

        df_merged = df_merged.sort_values("date")
        df_merged["rate_change"] = df_merged["key_rate"].diff()

        # Определение категории
        def get_category(change):
            if pd.isna(change):
                return "same"
            elif change < -0.1:
                return "down"
            elif change > 0.1:
                return "up"
            else:
                return "same"

        df_merged["target"] = df_merged["rate_change"].apply(get_category)

        df_merged = df_merged.dropna(subset=["release", "target"])

        processed_path = self.data_dir / "cbr-press-releases.csv"
        df_merged.to_csv(processed_path, index=False)
        logger.info(f"Сохранено {len(df_merged)} обработанных записей")

        return df_merged

    def collect_all_data(self) -> pd.DataFrame:
        """Сбор всех данных"""
        logger.info("Начало сбора всех данных...")

        logger.info("1. Получение пресс-релизов...")
        df_releases = self.get_press_releases()

        logger.info("2. Получение новостей типа events...")
        df_news = self.get_news_events()

        logger.info("3. Объединение пресс-релизов и новостей...")
        df_all_releases = pd.concat([df_releases, df_news], ignore_index=True)
        logger.info(f"Всего получено {len(df_all_releases)} документов ({len(df_releases)} пресс-релизов + {len(df_news)} новостей)")

        logger.info("4. Получение ключевой ставки...")
        df_key_rates = self.get_key_rate()

        logger.info("5. Получение инфляции...")
        df_inflation = self.get_inflation()

        logger.info("6. Получение курсов валют (USD, EUR, CNY)...")
        df_usd = self.get_usd_rate()
        self.get_eur_rate()
        self.get_cny_rate()

        logger.info("7. Получение ставки RUONIA...")
        self.get_ruonia()

        logger.info("8. Получение учётных цен на драгоценные металлы...")
        self.get_precious_metals()

        logger.info("9. Получение данных по обязательным резервам...")
        self.get_reserves()

        logger.info("10. Предобработка данных...")
        df_processed = self.preprocess_data(df_all_releases)

        if "release" in df_processed.columns and "target" in df_processed.columns:
            final_df = df_processed[["release", "target"]].copy()
            final_df = final_df.rename(columns={"release": "text", "target": "label"})

            final_df["cleaned_text"] = final_df["text"].apply(
                lambda x: " ".join(str(x).split()).lower().strip()
            )

            final_df = final_df[final_df["cleaned_text"].str.len() > 50]

            dataset_path = self.data_dir / "cbr_dataset.csv"
            final_df[["cleaned_text", "label"]].to_csv(dataset_path, index=False)

            logger.info(f"Финальный датасет: {len(final_df)} записей")
            logger.info(
                f"Распределение меток: {final_df['label'].value_counts().to_dict()}"
            )

            return final_df
        else:
            logger.error("Не удалось создать финальный датасет")
            return pd.DataFrame()

    def prepare_multitask_dataset(self, df_processed: pd.DataFrame = None) -> pd.DataFrame:
        """Подготовка датасета для multi-task обучения с метками для всех экономических параметров"""
        logger.info("Подготовка multi-task датасета...")

        if df_processed is None:
            processed_path = self.data_dir / "cbr-press-releases.csv"
            if not processed_path.exists():
                logger.error("Файл cbr-press-releases.csv не найден. Запустите collect_all_data() сначала.")
                return pd.DataFrame()
            df_processed = pd.read_csv(processed_path)

        df_processed = df_processed.sort_values("date").reset_index(drop=True)

        def get_binary_change(change_value, threshold=0.01):
            """Порог для определения значимого изменения"""
            if pd.isna(change_value):
                return None
            elif change_value > threshold:
                return 1
            else:
                return 0

        economic_params = ['usd', 'eur', 'cny', 'inflation', 'ruonia']

        for param in economic_params:
            if param in df_processed.columns:
                df_processed[f'{param}_change'] = df_processed[param].pct_change() * 100
                df_processed[f'{param}_label'] = df_processed[f'{param}_change'].apply(
                    lambda x: get_binary_change(x, threshold=0.01)
                )
            else:
                logger.warning(f"Параметр {param} не найден в данных")
                df_processed[f'{param}_change'] = None
                df_processed[f'{param}_label'] = None

        if 'target' in df_processed.columns:
            df_processed['key_rate_label_binary'] = df_processed['target'].map({
                'up': 1,
                'down': 0,
                'same': 0
            })

        columns_to_keep = ['date', 'release', 'target']

        for param in economic_params:
            if f'{param}_label' in df_processed.columns:
                columns_to_keep.append(f'{param}_label')

        if 'key_rate_label_binary' in df_processed.columns:
            columns_to_keep.append('key_rate_label_binary')

        for param in economic_params + ['key_rate']:
            if param in df_processed.columns:
                columns_to_keep.append(param)

        columns_to_keep = [col for col in columns_to_keep if col in df_processed.columns]
        multitask_df = df_processed[columns_to_keep].copy()

        multitask_df = multitask_df.rename(columns={'release': 'text'})

        multitask_df['cleaned_text'] = multitask_df['text'].apply(
            lambda x: " ".join(str(x).split()).lower().strip()
        )

        multitask_df = multitask_df[multitask_df['cleaned_text'].str.len() > 50]

        multitask_df = multitask_df.iloc[1:].reset_index(drop=True)

        auxiliary_label_cols = [f'{param}_label' for param in economic_params if f'{param}_label' in multitask_df.columns]
        if auxiliary_label_cols:
            multitask_df = multitask_df.dropna(subset=auxiliary_label_cols, how='all')

        multitask_path = self.data_dir / "cbr_multitask_dataset.csv"
        multitask_df.to_csv(multitask_path, index=False)

        logger.info(f"Multi-task датасет сохранен: {len(multitask_df)} записей")
        logger.info(f"Колонки: {list(multitask_df.columns)}")

        for param in economic_params:
            label_col = f'{param}_label'
            if label_col in multitask_df.columns:
                counts = multitask_df[label_col].value_counts()
                logger.info(f"{param} метки: {counts.to_dict()}")

        if 'target' in multitask_df.columns:
            logger.info(f"Key rate метки: {multitask_df['target'].value_counts().to_dict()}")

        return multitask_df
