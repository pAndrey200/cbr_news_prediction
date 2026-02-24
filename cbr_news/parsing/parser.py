import locale
import logging
from datetime import datetime
from pathlib import Path

import httpx
import pandas as pd
from bs4 import BeautifulSoup
from hydra.utils import get_original_cwd

from cbr_news.parsing.news_parser import CBRNewsParser

logger = logging.getLogger(__name__)


class CBRDataParser:
    def __init__(self, config, use_database=True):
        self.config = config
        self.use_database = use_database

        try:
            project_root = Path(get_original_cwd())
        except (ValueError, AttributeError):
            project_root = Path.cwd()
        self.data_dir = project_root / "data"
        self.data_dir.mkdir(exist_ok=True)

        try:
            locale.setlocale(locale.LC_ALL, "ru_RU.UTF-8")
        except locale.Error:
            pass

        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        self.news_parser = CBRNewsParser(config, use_database=use_database)

        if self.use_database:
            try:
                from cbr_news.database.db import get_db_session
                from cbr_news.database.repository import (
                    CurrencyRateRepository,
                    InflationRepository,
                    KeyRateRepository,
                    PreciousMetalRepository,
                    ReserveRepository,
                    RuoniaRepository,
                )
                self.get_db_session = get_db_session
                self.key_rate_repo = KeyRateRepository
                self.currency_repo = CurrencyRateRepository
                self.inflation_repo = InflationRepository
                self.ruonia_repo = RuoniaRepository
                self.metal_repo = PreciousMetalRepository
                self.reserve_repo = ReserveRepository
            except Exception as e:
                logger.error(f"Не удалось инициализировать подключение к БД: {e}")
                self.use_database = False

    def _parse_date_to_obj(self, date_str: str):
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

    def _build_indicator_series(self, df, date_col, value_col):
        df = df.copy()
        df["_dt"] = pd.to_datetime(df[date_col], format="%d.%m.%Y", errors="coerce")
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
        df = df.dropna(subset=["_dt", value_col]).drop_duplicates(subset=["_dt"]).sort_values("_dt")
        return df.set_index("_dt")[value_col]

    def _create_forward_labels(self, df_merged, indicator_series, horizon=5):
        import pandas as _pd

        df_merged = df_merged.copy()
        df_merged["_date_dt"] = _pd.to_datetime(df_merged["date"], format="%d.%m.%Y", errors="coerce")

        threshold = 0.05
        try:
            threshold = self.config.data.ruonia_threshold
        except (AttributeError, KeyError):
            pass

        if "ruonia" in indicator_series:
            ruonia_ts = indicator_series["ruonia"]
            max_date = ruonia_ts.index.max()

            def ruonia_label(d):
                if _pd.isna(d):
                    return None
                future_d = d + _pd.Timedelta(days=horizon)
                if future_d > max_date:
                    return None
                current = ruonia_ts.asof(d)
                future = ruonia_ts.asof(future_d)
                if _pd.isna(current) or _pd.isna(future):
                    return None
                change = future - current
                if change > threshold:
                    return "up"
                elif change < -threshold:
                    return "down"
                return "same"

            df_merged["target"] = df_merged["_date_dt"].apply(ruonia_label)
        else:
            df_merged["target"] = "same"

        for name, ts in indicator_series.items():
            if name == "ruonia":
                continue
            max_date = ts.index.max()

            def aux_label(d, _ts=ts, _max=max_date):
                if _pd.isna(d):
                    return None
                future_d = d + _pd.Timedelta(days=horizon)
                if future_d > _max:
                    return None
                current = _ts.asof(d)
                future = _ts.asof(future_d)
                if _pd.isna(current) or _pd.isna(future):
                    return None
                return 1 if future > current else 0

            df_merged[f"{name}_label"] = df_merged["_date_dt"].apply(aux_label)

        df_merged = df_merged.drop(columns=["_date_dt"])
        return df_merged

    def _save_key_rates_to_db(self, df: pd.DataFrame):
        if not self.use_database:
            return
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
        if not self.use_database:
            return
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

    def get_key_rate(self) -> pd.DataFrame:
        logger.info("Получение данных по ключевой ставке...")
        base_url = "https://cbr.ru/hd_base/KeyRate"
        response = httpx.get(base_url, headers=self.headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        target_script = None
        for script in soup.find_all("script"):
            if script.string and '"data":[' in script.string:
                target_script = script.string
                break

        dates_str = target_script.split(',"categories":["')[1].split("]")[0]
        values_str = target_script.split(',"data":[')[1].split("]")[0]

        dates_array = [s.strip('"') for s in dates_str.split(",")]
        values_array = [float(s) for s in values_str.split(",")]

        if len(dates_array) != len(values_array):
            min_len = min(len(dates_array), len(values_array))
            dates_array = dates_array[:min_len]
            values_array = values_array[:min_len]

        df = pd.DataFrame({"date": dates_array, "rate": values_array})
        df.to_csv(self.data_dir / "key-rates-cbr.csv", index=False)
        logger.info(f"Сохранено {len(df)} записей по ключевой ставке")
        self._save_key_rates_to_db(df)
        return df

    def get_inflation(self) -> pd.DataFrame:
        logger.info("Получение данных по инфляции...")
        base_url = "https://www.cbr.ru/hd_base/infl/?UniDbQuery.Posted=True&UniDbQuery.From=17.09.2013"
        response = httpx.get(base_url, headers=self.headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        table = soup.find("table", attrs={"class": "data"})
        if not table:
            raise ValueError("Таблица с данными не найдена")

        data = []
        for row in table.find_all("tr"):
            cols = [c.text.strip() for c in row.find_all("td")]
            if len(cols) == 4:
                try:
                    data.append([cols[0], float(cols[2].replace(",", "."))])
                except ValueError:
                    continue

        df = pd.DataFrame(data, columns=["date_inflation", "inflation"])
        df.to_csv(self.data_dir / "inflation-cbr.csv", index=False)
        logger.info(f"Сохранено {len(df)} записей по инфляции")
        self._save_inflation_to_db(df)
        return df

    def _get_currency_dynamics(self, val_nm_rq: str, column_name: str, from_date: str = "01.09.2013") -> pd.DataFrame:
        base_url = (
            "https://www.cbr.ru/currency_base/dynamics/?UniDbQuery.Posted=True&"
            "UniDbQuery.so=1&UniDbQuery.mode=1&UniDbQuery.date_req1=&"
            "UniDbQuery.date_req2=&UniDbQuery.VAL_NM_RQ={}&UniDbQuery.From={}"
        ).format(val_nm_rq, from_date)
        response = httpx.get(base_url, headers=self.headers, timeout=15)
        soup = BeautifulSoup(response.text, "html.parser")
        table = soup.find("table", attrs={"class": "data"})
        if not table:
            return pd.DataFrame(columns=["date", column_name])
        data = []
        for row in table.find_all("tr"):
            cols = [c.text.strip() for c in row.find_all("td")]
            if len(cols) == 3:
                try:
                    data.append([cols[0], float(cols[2].replace(",", "."))])
                except ValueError:
                    continue
        return pd.DataFrame(data, columns=["date", column_name])

    def get_usd_rate(self) -> pd.DataFrame:
        logger.info("Получение курса USD...")
        df = self._get_currency_dynamics("R01235", "usd")
        df.to_csv(self.data_dir / "cur-usd-cbr.csv", index=False)
        logger.info(f"Сохранено {len(df)} записей по курсу USD")
        self._save_currency_rates_to_db(df, "USD")
        return df

    def get_eur_rate(self) -> pd.DataFrame:
        logger.info("Получение курса EUR...")
        df = self._get_currency_dynamics("R01239", "eur")
        df.to_csv(self.data_dir / "cur-eur-cbr.csv", index=False)
        logger.info(f"Сохранено {len(df)} записей по курсу EUR")
        self._save_currency_rates_to_db(df, "EUR")
        return df

    def get_cny_rate(self) -> pd.DataFrame:
        logger.info("Получение курса CNY...")
        df = self._get_currency_dynamics("R01375", "cny")
        df.to_csv(self.data_dir / "cur-cny-cbr.csv", index=False)
        logger.info(f"Сохранено {len(df)} записей по курсу CNY")
        self._save_currency_rates_to_db(df, "CNY")
        return df

    def get_ruonia(self) -> pd.DataFrame:
        logger.info("Получение ставки RUONIA...")
        base_url = "https://www.cbr.ru/hd_base/ruonia/dynamics/"
        params = {"FromDate": "01.09.2013", "ToDate": "31.12.2030", "posted": "True"}
        try:
            response = httpx.get(base_url, params=params, headers=self.headers, timeout=15)
            soup = BeautifulSoup(response.text, "html.parser")
            table = soup.find("table", attrs={"class": "data"})
            if not table:
                return pd.DataFrame(columns=["date", "ruonia"])
            data = []
            for row in table.find_all("tr")[1:]:
                cols = [c.get_text(strip=True) for c in row.find_all("td")]
                if len(cols) >= 2:
                    try:
                        data.append([cols[0], float(cols[1].replace(",", "."))])
                    except ValueError:
                        continue
            df = pd.DataFrame(data, columns=["date", "ruonia"])
        except Exception as e:
            logger.warning("Ошибка парсинга RUONIA: %s", e)
            df = pd.DataFrame(columns=["date", "ruonia"])
        df.to_csv(self.data_dir / "ruonia-cbr.csv", index=False)
        logger.info(f"Сохранено {len(df)} записей RUONIA")
        if len(df) > 0:
            self._save_ruonia_to_db(df)
        return df

    def get_precious_metals(self) -> pd.DataFrame:
        logger.info("Получение учётных цен на драгоценные металлы...")
        base_url = "https://www.cbr.ru/hd_base/metall/metall_base_new/"
        try:
            response = httpx.get(base_url, headers=self.headers, timeout=15)
            soup = BeautifulSoup(response.text, "html.parser")
            table = soup.find("table", attrs={"class": "data"})
            if not table:
                return pd.DataFrame(columns=["date", "gold", "silver", "platinum", "palladium"])
            data = []
            for row in table.find_all("tr")[1:]:
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
        df.to_csv(self.data_dir / "metall-cbr.csv", index=False)
        logger.info(f"Сохранено {len(df)} записей по драгметаллам")
        if len(df) > 0:
            self._save_metals_to_db(df)
        return df

    def get_reserves(self) -> pd.DataFrame:
        logger.info("Получение данных по обязательным резервам...")
        base_url = "https://www.cbr.ru/hd_base/RReserves/"
        try:
            response = httpx.get(base_url, headers=self.headers, timeout=15)
            soup = BeautifulSoup(response.text, "html.parser")
            table = soup.find("table", attrs={"class": "data"})
            if not table:
                return pd.DataFrame(columns=["date_reserves", "reserves_avg", "reserves_accounts"])
            data = []
            for row in table.find_all("tr")[1:]:
                cols = row.find_all("td")
                if len(cols) < 4:
                    continue
                date_str = cols[0].get_text(strip=True)
                vals = []
                for c in cols[1:4]:
                    s = c.get_text(strip=True).replace("\xa0", "").replace(" ", "").replace(",", ".")
                    if s == "\u2014" or not s:
                        vals.append(float("nan"))
                    else:
                        try:
                            vals.append(float(s))
                        except ValueError:
                            vals.append(float("nan"))
                data.append([date_str] + vals)
            df = pd.DataFrame(data, columns=["date_reserves", "reserves_corset", "reserves_avg", "reserves_accounts"])
        except Exception as e:
            logger.warning("Ошибка парсинга резервов: %s", e)
            df = pd.DataFrame(columns=["date_reserves", "reserves_avg", "reserves_accounts"])
        df.to_csv(self.data_dir / "reserves-cbr.csv", index=False)
        logger.info(f"Сохранено {len(df)} записей по резервам")
        if len(df) > 0:
            self._save_reserves_to_db(df)
        return df

    def preprocess_data(self, df_releases: pd.DataFrame) -> pd.DataFrame:
        logger.info("Начало предобработки данных...")

        def parse_date(date_str):
            try:
                if "." in date_str and len(date_str.split(".")) == 3:
                    parts = date_str.split(".")
                    if len(parts[0]) <= 2 and len(parts[1]) <= 2 and len(parts[2]) == 4:
                        return date_str

                if "T" in date_str or (len(date_str) >= 10 and date_str[4] == "-" and date_str[7] == "-"):
                    try:
                        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                        return f"{dt.day:02d}.{dt.month:02d}.{dt.year}"
                    except Exception:
                        pass

                parts = date_str.split()
                if len(parts) >= 3:
                    day = parts[0]
                    month_ru = parts[1].lower()[:3]
                    year = parts[2]
                    months = {
                        "янв": "01", "jan": "01", "фев": "02", "feb": "02",
                        "мар": "03", "mar": "03", "апр": "04", "apr": "04",
                        "мая": "05", "май": "05", "may": "05",
                        "июн": "06", "jun": "06", "июл": "07", "jul": "07",
                        "авг": "08", "aug": "08", "сен": "09", "sep": "09",
                        "окт": "10", "oct": "10", "ноя": "11", "nov": "11",
                        "дек": "12", "dec": "12",
                    }
                    month = months.get(month_ru, "01")
                    return f"{day}.{month}.{year}"
            except Exception:
                pass
            return None

        df_releases["date_parsed"] = df_releases["date"].apply(parse_date)
        df_releases = df_releases.dropna(subset=["date_parsed"])

        df_key_rates = pd.read_csv(self.data_dir / "key-rates-cbr.csv", dtype={"date": str, "rate": float})
        df_usd = pd.read_csv(self.data_dir / "cur-usd-cbr.csv", dtype={"date": str, "usd": float})
        df_inflation = pd.read_csv(self.data_dir / "inflation-cbr.csv", dtype={"date_inflation": str, "inflation": float})

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
                        if c != date_col:
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
            except Exception:
                return ""

        df_releases["month"] = df_releases["date_parsed"].apply(get_month)
        df_usd = df_usd.rename(columns={"date": "date_usd"})
        df_key_rates = df_key_rates.rename(columns={"date": "date_keyrate", "rate": "key_rate"})

        df_merged = pd.merge(df_releases, df_inflation, left_on="month", right_on="date_inflation", how="left")
        df_merged = pd.merge(df_merged, df_usd, left_on="date_parsed", right_on="date_usd", how="left")
        df_merged = pd.merge(df_merged, df_key_rates, left_on="date_parsed", right_on="date_keyrate", how="left")

        for name, df_extra in dfs_extra.items():
            date_col = f"date_{name}"
            if date_col not in df_extra.columns:
                continue
            df_merged = pd.merge(df_merged, df_extra, left_on="date_parsed", right_on=date_col, how="left", suffixes=("", "_drop"))
            df_merged = df_merged.drop(columns=[date_col], errors="ignore")

        if df_metall is not None and "date_metall" in df_metall.columns:
            df_merged = pd.merge(df_merged, df_metall, left_on="date_parsed", right_on="date_metall", how="left")
            df_merged = df_merged.drop(columns=["date_metall"], errors="ignore")

        if df_reserves is not None and "date_reserves" in df_reserves.columns:
            df_merged = pd.merge(df_merged, df_reserves, left_on="date_parsed", right_on="date_reserves", how="left")
            df_merged = df_merged.drop(columns=["date_reserves"], errors="ignore")

        df_merged = df_merged.drop(columns=["date_usd", "date_keyrate", "date"], errors="ignore")
        drop_cols = [c for c in df_merged.columns if c.endswith("_drop")]
        df_merged = df_merged.drop(columns=drop_cols, errors="ignore")
        df_merged = df_merged.rename(columns={"date_parsed": "date"})

        columns_to_keep = [
            "date", "link", "title", "release",
            "inflation", "usd", "eur", "cny", "key_rate", "ruonia",
            "gold", "silver", "platinum", "palladium",
            "reserves_avg", "reserves_accounts", "reserves_corset",
        ]
        df_merged = df_merged[[col for col in columns_to_keep if col in df_merged.columns]]

        if df_merged.columns.duplicated().any():
            df_merged = df_merged.loc[:, ~df_merged.columns.duplicated()]

        for col in ["usd", "inflation", "eur", "cny", "ruonia", "gold", "silver", "platinum", "palladium", "reserves_avg", "reserves_accounts", "reserves_corset"]:
            if col in df_merged.columns:
                df_merged[col] = df_merged[col].ffill().bfill()

        if "date" not in df_merged.columns:
            raise ValueError("Колонка 'date' не найдена в DataFrame")
        if df_merged.columns.tolist().count("date") > 1:
            df_merged = df_merged.loc[:, ~df_merged.columns.duplicated()]

        df_merged = df_merged.sort_values("date")

        horizon = 5
        try:
            horizon = self.config.data.prediction_horizon
        except (AttributeError, KeyError):
            pass

        indicator_series = {}
        for csv_name, date_col, val_col, series_name in [
            ("ruonia-cbr.csv", "date", "ruonia", "ruonia"),
            ("cur-usd-cbr.csv", "date", "usd", "usd"),
            ("key-rates-cbr.csv", "date", "rate", "key_rate"),
            ("cur-eur-cbr.csv", "date", "eur", "eur"),
            ("cur-cny-cbr.csv", "date", "cny", "cny"),
        ]:
            path = self.data_dir / csv_name
            if path.exists():
                try:
                    df_ts = pd.read_csv(path, dtype=str)
                    indicator_series[series_name] = self._build_indicator_series(df_ts, date_col, val_col)
                except Exception as e:
                    logger.warning(f"Ошибка построения серии {series_name}: {e}")

        df_merged = self._create_forward_labels(df_merged, indicator_series, horizon)
        df_merged = df_merged.dropna(subset=["release", "target"])

        processed_path = self.data_dir / "cbr-press-releases.csv"
        df_merged.to_csv(processed_path, index=False)
        logger.info(f"Сохранено {len(df_merged)} обработанных записей")
        return df_merged

    def _load_news_from_db(self) -> pd.DataFrame:
        if not self.use_database:
            raise RuntimeError("БД недоступна, невозможно загрузить данные")

        from cbr_news.database.repository import NewsRepository

        with self.get_db_session() as session:
            news_list = NewsRepository.get_all(session)
            data = []
            for n in news_list:
                data.append({
                    "date": n.date.strftime("%d.%m.%Y") if n.date else "",
                    "link": n.link,
                    "title": n.title,
                    "release": n.content or "",
                })
            logger.info(f"Загружено {len(data)} новостей из БД")
            return pd.DataFrame(data, columns=["date", "link", "title", "release"])

    def _load_numeric_from_db(self):
        from cbr_news.database.repository import (
            CurrencyRateRepository,
            InflationRepository,
            KeyRateRepository,
            PreciousMetalRepository,
            ReserveRepository,
            RuoniaRepository,
        )

        with self.get_db_session() as session:
            key_rates = KeyRateRepository.get_all(session)
            df_key_rates = pd.DataFrame(
                [{"date": r.date.strftime("%d.%m.%Y"), "rate": r.rate} for r in key_rates]
            ) if key_rates else pd.DataFrame(columns=["date", "rate"])

            usd_rates = CurrencyRateRepository.get_all_by_currency(session, "USD")
            df_usd = pd.DataFrame(
                [{"date": r.date.strftime("%d.%m.%Y"), "usd": r.rate} for r in usd_rates]
            ) if usd_rates else pd.DataFrame(columns=["date", "usd"])

            eur_rates = CurrencyRateRepository.get_all_by_currency(session, "EUR")
            df_eur = pd.DataFrame(
                [{"date": r.date.strftime("%d.%m.%Y"), "eur": r.rate} for r in eur_rates]
            ) if eur_rates else pd.DataFrame(columns=["date", "eur"])

            cny_rates = CurrencyRateRepository.get_all_by_currency(session, "CNY")
            df_cny = pd.DataFrame(
                [{"date": r.date.strftime("%d.%m.%Y"), "cny": r.rate} for r in cny_rates]
            ) if cny_rates else pd.DataFrame(columns=["date", "cny"])

            inflation_list = InflationRepository.get_all(session)
            df_inflation = pd.DataFrame(
                [{"date_inflation": r.date.strftime("%d.%m.%Y"), "inflation": r.value} for r in inflation_list]
            ) if inflation_list else pd.DataFrame(columns=["date_inflation", "inflation"])

            ruonia_list = RuoniaRepository.get_all(session)
            df_ruonia = pd.DataFrame(
                [{"date": r.date.strftime("%d.%m.%Y"), "ruonia": r.rate} for r in ruonia_list]
            ) if ruonia_list else pd.DataFrame(columns=["date", "ruonia"])

            metals_list = PreciousMetalRepository.get_all(session)
            metals_data = {}
            for m in metals_list:
                key = m.date.strftime("%d.%m.%Y")
                if key not in metals_data:
                    metals_data[key] = {"date": key}
                metals_data[key][m.metal_type] = m.price
            df_metall = pd.DataFrame(list(metals_data.values())) if metals_data else pd.DataFrame(columns=["date", "gold", "silver", "platinum", "palladium"])

            reserves_list = ReserveRepository.get_all(session)
            df_reserves = pd.DataFrame(
                [{
                    "date_reserves": r.date.strftime("%d.%m.%Y"),
                    "reserves_corset": r.reserves_corset,
                    "reserves_avg": r.reserves_avg,
                    "reserves_accounts": r.reserves_accounts,
                } for r in reserves_list]
            ) if reserves_list else pd.DataFrame(columns=["date_reserves", "reserves_corset", "reserves_avg", "reserves_accounts"])

        logger.info("Все числовые данные загружены из БД")
        return df_key_rates, df_usd, df_eur, df_cny, df_inflation, df_ruonia, df_metall, df_reserves

    def _run_parsing(self) -> pd.DataFrame:
        logger.info("Режим парсинга: сбор данных с сайта ЦБ...")

        logger.info("1. Получение всех новостей (events + press)...")
        df_all_news = self.news_parser.get_all_news()
        logger.info(f"Всего получено {len(df_all_news)} новостей")

        logger.info("2. Получение ключевой ставки...")
        self.get_key_rate()

        logger.info("3. Получение инфляции...")
        self.get_inflation()

        logger.info("4. Получение курсов валют (USD, EUR, CNY)...")
        self.get_usd_rate()
        self.get_eur_rate()
        self.get_cny_rate()

        logger.info("5. Получение ставки RUONIA...")
        self.get_ruonia()

        logger.info("6. Получение учётных цен на драгоценные металлы...")
        self.get_precious_metals()

        logger.info("7. Получение данных по обязательным резервам...")
        self.get_reserves()

        return df_all_news

    def collect_all_data(self) -> pd.DataFrame:
        run_parser = True
        if self.config is not None:
            try:
                run_parser = self.config.data.run_parser
            except (AttributeError, KeyError):
                pass

        if run_parser:
            df_all_news = self._run_parsing()
        else:
            logger.info("Режим БД: загрузка данных из базы данных...")
            df_all_news = self._load_news_from_db()

        logger.info("Предобработка данных...")

        if run_parser:
            df_processed = self.preprocess_data(df_all_news)
        else:
            df_processed = self._preprocess_from_db(df_all_news)

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
            logger.info(f"Распределение меток: {final_df['label'].value_counts().to_dict()}")
            return final_df
        else:
            logger.error("Не удалось создать финальный датасет")
            return pd.DataFrame()

    def _preprocess_from_db(self, df_releases: pd.DataFrame) -> pd.DataFrame:
        logger.info("Предобработка данных из БД...")

        df_releases["date_parsed"] = df_releases["date"]
        df_releases = df_releases.dropna(subset=["date_parsed"])

        df_key_rates, df_usd, df_eur, df_cny, df_inflation, df_ruonia, df_metall, df_reserves = self._load_numeric_from_db()

        def get_month(date_str):
            try:
                parts = date_str.split(".")
                return f"{parts[1]}.{parts[2]}"
            except Exception:
                return ""

        df_releases["month"] = df_releases["date_parsed"].apply(get_month)

        indicator_series = {}
        for name, df_ind, date_col, val_col in [
            ("ruonia", df_ruonia, "date", "ruonia"),
            ("usd", df_usd, "date", "usd"),
            ("key_rate", df_key_rates, "date", "rate"),
            ("eur", df_eur, "date", "eur"),
            ("cny", df_cny, "date", "cny"),
        ]:
            if not df_ind.empty:
                try:
                    indicator_series[name] = self._build_indicator_series(df_ind, date_col, val_col)
                except Exception as e:
                    logger.warning(f"Ошибка построения серии {name}: {e}")

        df_usd = df_usd.rename(columns={"date": "date_usd"})
        df_key_rates = df_key_rates.rename(columns={"date": "date_keyrate", "rate": "key_rate"})

        df_merged = pd.merge(df_releases, df_inflation, left_on="month", right_on="date_inflation", how="left")
        df_merged = pd.merge(df_merged, df_usd, left_on="date_parsed", right_on="date_usd", how="left")
        df_merged = pd.merge(df_merged, df_key_rates, left_on="date_parsed", right_on="date_keyrate", how="left")

        for name, df_extra, rename_map in [
            ("eur", df_eur, {"date": "date_eur"}),
            ("cny", df_cny, {"date": "date_cny"}),
            ("ruonia", df_ruonia, {"date": "date_ruonia"}),
        ]:
            date_col = f"date_{name}"
            df_extra = df_extra.rename(columns=rename_map)
            if date_col in df_extra.columns:
                df_merged = pd.merge(df_merged, df_extra, left_on="date_parsed", right_on=date_col, how="left", suffixes=("", "_drop"))
                df_merged = df_merged.drop(columns=[date_col], errors="ignore")

        if not df_metall.empty:
            df_metall = df_metall.rename(columns={"date": "date_metall"})
            df_merged = pd.merge(df_merged, df_metall, left_on="date_parsed", right_on="date_metall", how="left")
            df_merged = df_merged.drop(columns=["date_metall"], errors="ignore")

        if not df_reserves.empty:
            df_merged = pd.merge(df_merged, df_reserves, left_on="date_parsed", right_on="date_reserves", how="left")
            df_merged = df_merged.drop(columns=["date_reserves"], errors="ignore")

        df_merged = df_merged.drop(columns=["date_usd", "date_keyrate", "date"], errors="ignore")
        drop_cols = [c for c in df_merged.columns if c.endswith("_drop")]
        df_merged = df_merged.drop(columns=drop_cols, errors="ignore")
        df_merged = df_merged.rename(columns={"date_parsed": "date"})

        columns_to_keep = [
            "date", "link", "title", "release",
            "inflation", "usd", "eur", "cny", "key_rate", "ruonia",
            "gold", "silver", "platinum", "palladium",
            "reserves_avg", "reserves_accounts", "reserves_corset",
        ]
        df_merged = df_merged[[col for col in columns_to_keep if col in df_merged.columns]]

        if df_merged.columns.duplicated().any():
            df_merged = df_merged.loc[:, ~df_merged.columns.duplicated()]

        for col in ["usd", "inflation", "eur", "cny", "ruonia", "gold", "silver", "platinum", "palladium", "reserves_avg", "reserves_accounts", "reserves_corset"]:
            if col in df_merged.columns:
                df_merged[col] = df_merged[col].ffill().bfill()

        df_merged = df_merged.sort_values("date")

        horizon = 5
        try:
            horizon = self.config.data.prediction_horizon
        except (AttributeError, KeyError):
            pass

        df_merged = self._create_forward_labels(df_merged, indicator_series, horizon)
        df_merged = df_merged.dropna(subset=["release", "target"])

        processed_path = self.data_dir / "cbr-press-releases.csv"
        df_merged.to_csv(processed_path, index=False)
        logger.info(f"Сохранено {len(df_merged)} обработанных записей")
        return df_merged

    def prepare_multitask_dataset(self, df_processed: pd.DataFrame = None) -> pd.DataFrame:
        logger.info("Подготовка multi-task датасета...")

        if df_processed is None:
            processed_path = self.data_dir / "cbr-press-releases.csv"
            if not processed_path.exists():
                logger.error("Файл cbr-press-releases.csv не найден.")
                return pd.DataFrame()
            df_processed = pd.read_csv(processed_path)

        df_processed = df_processed.sort_values("date").reset_index(drop=True)

        auxiliary_params = ['usd', 'eur', 'cny', 'key_rate']
        columns_to_keep = ['date', 'release', 'target']
        for param in auxiliary_params:
            label_col = f'{param}_label'
            if label_col in df_processed.columns:
                columns_to_keep.append(label_col)
            if param in df_processed.columns:
                columns_to_keep.append(param)
        for col in ['ruonia', 'inflation']:
            if col in df_processed.columns and col not in columns_to_keep:
                columns_to_keep.append(col)

        columns_to_keep = [col for col in columns_to_keep if col in df_processed.columns]
        multitask_df = df_processed[columns_to_keep].copy()
        multitask_df = multitask_df.rename(columns={'release': 'text'})
        multitask_df['cleaned_text'] = multitask_df['text'].apply(
            lambda x: " ".join(str(x).split()).lower().strip()
        )
        multitask_df = multitask_df[multitask_df['cleaned_text'].str.len() > 50]
        multitask_df = multitask_df.dropna(subset=['target'])

        multitask_path = self.data_dir / "cbr_multitask_dataset.csv"
        multitask_df.to_csv(multitask_path, index=False)

        logger.info(f"Multi-task датасет сохранен: {len(multitask_df)} записей")
        logger.info(f"Распределение target: {multitask_df['target'].value_counts().to_dict()}")
        return multitask_df
