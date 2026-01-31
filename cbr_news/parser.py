import json
import locale
import logging
import time
from pathlib import Path

import httpx
import pandas as pd
from bs4 import BeautifulSoup
from hydra.utils import get_original_cwd

logger = logging.getLogger(__name__)


class CBRDataParser:
    """Парсер данных Банка России"""

    def __init__(self, config):
        self.config = config
        # Используем абсолютный путь к корню проекта, чтобы данные сохранялись
        # в фиксированном месте, а не в outputs/
        try:
            project_root = Path(get_original_cwd())
        except (ValueError, AttributeError):
            # Если Hydra не используется, используем текущую директорию
            project_root = Path.cwd()
        self.data_dir = project_root / "data"
        self.data_dir.mkdir(exist_ok=True)
        locale.setlocale(locale.LC_ALL, "ru_RU.UTF-8")
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

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
                    # Удаление скриптов и стилей
                    for script in release_div.find_all(["script", "style"]):
                        script.decompose()

                    # Получение чистого текста
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

        return df

    def get_news_events(self) -> pd.DataFrame:
        """Получение новостей типа events с сайта Банка России"""
        base_url = "https://www.cbr.ru/news/eventandpress/"
        page = 0
        all_events = []
        
        logger.info("Начало парсинга новостей типа events Банка России...")
        
        while True:
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
                
                # Фильтруем только события типа "events"
                events_on_page = [event for event in events_data if event.get("TBLType") == "events"]
                
                if events_on_page:
                    all_events.extend(events_on_page)
                    logger.info(f"Получено {len(events_on_page)} событий со страницы {page} (всего: {len(all_events)})")
                else:
                    logger.info(f"Не найдено событий типа 'events' на странице {page}")
                
                # Если на странице вообще нет данных, прекращаем парсинг
                # Если есть данные, но нет событий типа "events", продолжаем
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
                
                # Формируем ссылку на полный текст новости
                news_url = f"https://www.cbr.ru/press/event/?id={doc_htm}"
                
                # Получаем дату из поля DT
                date_str = event.get("DT", "")
                if date_str:
                    # Преобразуем формат даты из ISO в формат с русскими месяцами
                    try:
                        from datetime import datetime
                        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                        # Русские названия месяцев
                        months_ru = {
                            1: "янв", 2: "фев", 3: "мар", 4: "апр", 5: "мая",
                            6: "июн", 7: "июл", 8: "авг", 9: "сен", 10: "окт",
                            11: "ноя", 12: "дек"
                        }
                        month_ru = months_ru.get(dt.month, "янв")
                        date_str = f"{dt.day} {month_ru} {dt.year}"
                    except:
                        pass
                
                # Получаем заголовок
                title = event.get("name_doc", "")
                
                # Получаем полный текст новости
                try:
                    response = httpx.get(news_url, headers=self.headers, timeout=30)
                    tree = BeautifulSoup(response.text, "html.parser")
                    
                    # Ищем основной контент новости
                    # Пробуем разные селекторы, которые могут содержать текст
                    content = None
                    for selector in [".landing-text", ".news-content", "article", ".content"]:
                        content = tree.select_one(selector)
                        if content:
                            break
                    
                    if not content:
                        # Если не нашли по селекторам, ищем основной контент в body
                        content = tree.find("main") or tree.find("article") or tree.find("div", class_=lambda x: x and "content" in x.lower())
                    
                    if content:
                        # Удаление скриптов и стилей
                        for script in content.find_all(["script", "style"]):
                            script.decompose()
                        
                        # Получение чистого текста
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

        return df

    def get_usd_rate(self) -> pd.DataFrame:
        """Получение курса USD (упрощенный вариант)"""
        logger.info("Получение курса USD...")

        base_url = (
            "https://www.cbr.ru/currency_base/dynamics/?UniDbQuery.Posted=True&"
            "UniDbQuery.so=1&UniDbQuery.mode=1&UniDbQuery.date_req1=&"
            "UniDbQuery.date_req2=&UniDbQuery.VAL_NM_RQ=R01235&UniDbQuery.From=01.09.2013"
        )

        response = httpx.get(base_url, headers=self.headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        # Ищем таблицу с классом "data"
        table = soup.find("table", attrs={"class": "data"})
        if not table:
            raise ValueError("Таблица с курсами не найдена")

        rows = table.find_all("tr")
        data = []

        for row in rows:
            cols = row.find_all("td")
            cols = [c.text.strip() for c in cols]
            if len(cols) == 3:
                date_str = cols[0]
                rate_str = cols[2].replace(",", ".")

                try:
                    rate_value = float(rate_str)
                    data.append([date_str, rate_value])
                except ValueError:
                    continue

        df = pd.DataFrame(data, columns=["date", "usd"])

        usd_path = self.data_dir / "cur-usd-cbr.csv"
        df.to_csv(usd_path, index=False)
        logger.info(f"Сохранено {len(df)} записей по курсу USD")

        return df

    def preprocess_data(self, df_releases: pd.DataFrame) -> pd.DataFrame:
        """Предобработка данных (упрощенная)"""
        logger.info("Начало предобработки данных...")

        def parse_date(date_str):
            try:
                # Если дата уже в формате DD.MM.YYYY, возвращаем как есть
                if "." in date_str and len(date_str.split(".")) == 3:
                    parts = date_str.split(".")
                    if len(parts[0]) <= 2 and len(parts[1]) <= 2 and len(parts[2]) == 4:
                        return date_str
                
                # Если дата в ISO формате (YYYY-MM-DD или YYYY-MM-DDTHH:MM:SS)
                if "T" in date_str or (len(date_str) >= 10 and date_str[4] == "-" and date_str[7] == "-"):
                    try:
                        from datetime import datetime
                        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                        return f"{dt.day:02d}.{dt.month:02d}.{dt.year}"
                    except:
                        pass
                
                # Парсим формат "DD MMM YYYY" (русский или английский)
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

        # Удаляем временные колонки дат и исходную колонку date (если есть)
        # чтобы избежать дубликатов при переименовании date_parsed в date
        df_merged = df_merged.drop(columns=["date_usd", "date_keyrate", "date"], errors="ignore")
        
        df_merged = df_merged.rename(
            columns={
                "date_parsed": "date",
            }
        )

        columns_to_keep = [
            "date",
            "link",
            "title",
            "release",
            "inflation",
            "usd",
            "key_rate",
        ]
        df_merged = df_merged[
            [col for col in columns_to_keep if col in df_merged.columns]
        ]

        # Убеждаемся, что нет дубликатов колонки date
        if df_merged.columns.duplicated().any():
            df_merged = df_merged.loc[:, ~df_merged.columns.duplicated()]

        df_merged["usd"] = df_merged["usd"].ffill().bfill()
        df_merged["inflation"] = df_merged["inflation"].ffill().bfill()

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

        # Сбор данных
        logger.info("1. Получение пресс-релизов...")
        df_releases = self.get_press_releases()

        logger.info("2. Получение новостей типа events...")
        df_news = self.get_news_events()

        logger.info("3. Объединение пресс-релизов и новостей...")
        # Объединяем пресс-релизы и новости
        df_all_releases = pd.concat([df_releases, df_news], ignore_index=True)
        logger.info(f"Всего получено {len(df_all_releases)} документов ({len(df_releases)} пресс-релизов + {len(df_news)} новостей)")

        logger.info("4. Получение ключевой ставки...")
        df_key_rates = self.get_key_rate()

        logger.info("5. Получение инфляции...")
        df_inflation = self.get_inflation()

        logger.info("6. Получение курса USD...")
        df_usd = self.get_usd_rate()

        logger.info("7. Предобработка данных...")
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
