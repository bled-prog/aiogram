import sys
from datetime import datetime, timedelta, timezone
from typing import Union

from pydantic import PlainSerializer, SerializationInfo
from typing_extensions import Annotated

if sys.platform == "win32":  # pragma: no cover

    def _datetime_to_timestamp(value: datetime) -> int:
        tz = timezone.utc if value.tzinfo else None

        # https://github.com/aiogram/aiogram/issues/349
        # https://github.com/aiogram/aiogram/pull/880
        return int((value - datetime(1970, 1, 1, tzinfo=tz)).total_seconds())

else:  # pragma: no cover

    def _datetime_to_timestamp(value: datetime) -> int:
        return int(value.timestamp())


def _datetime_serializer(
    dt: "DateTime", info: SerializationInfo
) -> Union[datetime, timedelta, int]:
    if info.mode == "python":
        return dt
    if isinstance(dt, timedelta):
        dt = datetime.now(timezone.utc) + dt
    return _datetime_to_timestamp(dt)


# Make datetime compatible with Telegram Bot API (unixtime)
DateTime = Annotated[
    Union[datetime, timedelta],
    PlainSerializer(
        func=_datetime_serializer,
        when_used="unless-none",
    ),
]
