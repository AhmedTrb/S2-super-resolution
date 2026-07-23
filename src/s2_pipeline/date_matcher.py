from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import List, Union


class TemporalDateMatcher:
    """Map observation dates to the nearest Sentinel-2 acquisition date."""

    def __init__(self, calendar_path: Union[str, Path]) -> None:
        self.calendar_path = Path(calendar_path)
        self._s2_dates: List[date] = self._load_calendar(self.calendar_path)
        self._s2_date_strings: List[str] = [d.isoformat() for d in self._s2_dates]
        self._s2_date_to_index = {date_str: idx for idx, date_str in enumerate(self._s2_date_strings)}

    @staticmethod
    def _load_calendar(calendar_path: Path) -> List[date]:
        dates: List[date] = []
        with calendar_path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                token = raw_line.strip()
                if not token:
                    continue
                dates.append(TemporalDateMatcher._parse_date_token(token))
        if not dates:
            raise ValueError(f"No dates found in calendar file: {calendar_path}")
        dates.sort()
        return dates

    @staticmethod
    def _parse_date_token(token: str) -> date:
        token = token.strip()
        if "-" in token:
            return datetime.strptime(token, "%Y-%m-%d").date()
        if len(token) == 8 and token.isdigit():
            return datetime.strptime(token, "%Y%m%d").date()
        raise ValueError(f"Unsupported date token format: {token!r}")

    @staticmethod
    def parse_observation_column(column_name: str, year: int = 2020) -> date:
        day_str, month_str = column_name.split("_")
        return date(year, int(month_str), int(day_str))

    @property
    def s2_dates(self) -> List[date]:
        return list(self._s2_dates)

    @property
    def s2_date_strings(self) -> List[str]:
        return list(self._s2_date_strings)

    def find_nearest_s2_date(self, observation_date: Union[date, datetime, str]) -> str:
        obs = self._coerce_date(observation_date)
        nearest = min(self._s2_dates, key=lambda s2_date: abs((s2_date - obs).days))
        return nearest.isoformat()

    def days_to_nearest_s2(self, observation_date: Union[date, datetime, str]) -> int:
        obs = self._coerce_date(observation_date)
        snapped = datetime.strptime(self.find_nearest_s2_date(obs), "%Y-%m-%d").date()
        return abs((snapped - obs).days)

    def get_band_offset(self, s2_date_str: str) -> int:
        return self._s2_date_to_index[s2_date_str]

    @staticmethod
    def _coerce_date(value: Union[date, datetime, str]) -> date:
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, str):
            return TemporalDateMatcher._parse_date_token(value)
        return value
