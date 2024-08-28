from __future__ import annotations

import abc
import json
from http import HTTPStatus
from types import TracebackType
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    Final,
    Optional,
    Type,
    cast,
)

from pydantic import ValidationError

from aiogram.exceptions import (
    ClientDecodeError,
    RestartingTelegram,
    TelegramAPIError,
    TelegramBadRequest,
    TelegramConflictError,
    TelegramEntityTooLarge,
    TelegramForbiddenError,
    TelegramMigrateToChat,
    TelegramNotFound,
    TelegramRetryAfter,
    TelegramServerError,
    TelegramUnauthorizedError,
)

from ...methods import Response, TelegramMethod
from ...methods.base import TelegramType
from ..telegram import PRODUCTION, TelegramAPIServer
from .middlewares.manager import RequestMiddlewareManager

if TYPE_CHECKING:
    from ..bot import Bot

_JsonLoads = Callable[..., Any]
_JsonDumps = Callable[..., str]

DEFAULT_TIMEOUT: Final[float] = 60.0


class BaseSession(abc.ABC):
    """
    This is base class for all HTTP sessions in aiogram.

    The parameters `json_loads`, `json_dumps`, and `timeout` are deprecated
    and will be removed in version 3.14.0. These parameters are no longer used,
    and the corresponding functionalities have been replaced by pydantic's
    built-in serialization/deserialization methods.
    """

    def __init__(
        self,
        api: TelegramAPIServer = PRODUCTION,
        json_loads: Optional[_JsonLoads] = None,
        json_dumps: Optional[_JsonDumps] = None,
        timeout: Optional[float] = None,
    ) -> None:
        """
        :param api: Telegram Bot API URL patterns
        :param json_loads: JSON loader (deprecated, not used)
        :param json_dumps: JSON dumper (deprecated, not used)
        :param timeout: Session scope request timeout (deprecated, not used)
        """
        if json_loads is not None or json_dumps is not None or timeout is not None:
            warnings.warn(
                "Parameters `json_loads`, `json_dumps`, and `timeout` are deprecated "
                "and will be removed in aiogram 3.14.0. These parameters are no longer use.",
                DeprecationWarning,
                stacklevel=2,
            )

        self.api = api
        self.middleware = RequestMiddlewareManager()

    def check_response(
        self,
        bot: Bot,
        method: TelegramMethod[TelegramType],
        status_code: int,
        content: str,
    ) -> Response[TelegramType]:
        """
        Check response status
        """
        try:
            response_type = Response[method.__returning__]  # type: ignore
            response = response_type.model_validate_json(content, context={"bot": bot})
        except ValidationError as e:
            raise ClientDecodeError("Failed to deserialize object", e, content)

        if HTTPStatus.OK <= status_code <= HTTPStatus.IM_USED and response.ok:
            return response

        description = cast(str, response.description)

        if parameters := response.parameters:
            if parameters.retry_after:
                raise TelegramRetryAfter(
                    method=method,
                    message=description,
                    retry_after=parameters.retry_after,
                )
            if parameters.migrate_to_chat_id:
                raise TelegramMigrateToChat(
                    method=method,
                    message=description,
                    migrate_to_chat_id=parameters.migrate_to_chat_id,
                )
        if status_code == HTTPStatus.BAD_REQUEST:
            raise TelegramBadRequest(method=method, message=description)
        if status_code == HTTPStatus.NOT_FOUND:
            raise TelegramNotFound(method=method, message=description)
        if status_code == HTTPStatus.CONFLICT:
            raise TelegramConflictError(method=method, message=description)
        if status_code == HTTPStatus.UNAUTHORIZED:
            raise TelegramUnauthorizedError(method=method, message=description)
        if status_code == HTTPStatus.FORBIDDEN:
            raise TelegramForbiddenError(method=method, message=description)
        if status_code == HTTPStatus.REQUEST_ENTITY_TOO_LARGE:
            raise TelegramEntityTooLarge(method=method, message=description)
        if status_code >= HTTPStatus.INTERNAL_SERVER_ERROR:
            if "restart" in description:
                raise RestartingTelegram(method=method, message=description)
            raise TelegramServerError(method=method, message=description)

        raise TelegramAPIError(
            method=method,
            message=description,
        )

    @abc.abstractmethod
    async def close(self) -> None:  # pragma: no cover
        """
        Close client session
        """
        pass

    @abc.abstractmethod
    async def make_request(
        self,
        bot: Bot,
        method: TelegramMethod[TelegramType],
        timeout: Optional[int] = None,
    ) -> TelegramType:  # pragma: no cover
        """
        Make request to Telegram Bot API

        :param bot: Bot instance
        :param method: Method instance
        :param timeout: Request timeout
        :return:
        :raise TelegramApiError:
        """
        pass

    @abc.abstractmethod
    async def stream_content(
        self,
        url: str,
        headers: Optional[Dict[str, Any]] = None,
        timeout: int = 30,
        chunk_size: int = 65536,
        raise_for_status: bool = True,
    ) -> AsyncGenerator[bytes, None]:  # pragma: no cover
        """
        Stream reader
        """
        yield b""

    async def __call__(
        self,
        bot: Bot,
        method: TelegramMethod[TelegramType],
        timeout: Optional[int] = None,
    ) -> TelegramType:
        middleware = self.middleware.wrap_middlewares(
            self.make_request, timeout=timeout
        )
        return cast(TelegramType, await middleware(bot, method))

    async def __aenter__(self) -> BaseSession:
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        await self.close()
