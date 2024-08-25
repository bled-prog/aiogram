from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional, Union

from pydantic import Field

from .base import TelegramMethod
from ..client.default_annotations import (
    DefaultAllowSendingWithoutReply,
    DefaultDisableNotification,
    DefaultLinkPreviewOptions,
    DefaultParseMode,
    DefaultProtectContent,
)
from ..types import (
    ForceReply,
    InlineKeyboardMarkup,
    Message,
    MessageEntity,
    ReplyKeyboardMarkup,
    ReplyKeyboardRemove,
    ReplyParameters,
)


class SendMessage(TelegramMethod[Message]):
    """
    Use this method to send text messages. On success, the sent :class:`aiogram.types.message.Message` is returned.

    Source: https://core.telegram.org/bots/api#sendmessage
    """

    __returning__ = Message
    __api_method__ = "sendMessage"

    chat_id: Union[int, str]
    """Unique identifier for the target chat or username of the target channel (in the format :code:`@channelusername`)"""
    text: str
    """Text of the message to be sent, 1-4096 characters after entities parsing"""
    business_connection_id: Optional[str] = None
    """Unique identifier of the business connection on behalf of which the message will be sent"""
    message_thread_id: Optional[int] = None
    """Unique identifier for the target message thread (topic) of the forum; for forum supergroups only"""
    parse_mode: DefaultParseMode = None
    """Mode for parsing entities in the message text. See `formatting options <https://core.telegram.org/bots/api#formatting-options>`_ for more details."""
    entities: Optional[List[MessageEntity]] = None
    """A JSON-serialized list of special entities that appear in message text, which can be specified instead of *parse_mode*"""
    link_preview_options: DefaultLinkPreviewOptions = None
    """Link preview generation options for the message"""
    disable_notification: DefaultDisableNotification = None
    """Sends the message `silently <https://telegram.org/blog/channels-2-0#silent-messages>`_. Users will receive a notification with no sound."""
    protect_content: DefaultProtectContent = None
    """Protects the contents of the sent message from forwarding and saving"""
    message_effect_id: Optional[str] = None
    """Unique identifier of the message effect to be added to the message; for private chats only"""
    reply_parameters: Optional[ReplyParameters] = None
    """Description of the message to reply to"""
    reply_markup: Optional[
        Union[InlineKeyboardMarkup, ReplyKeyboardMarkup, ReplyKeyboardRemove, ForceReply]
    ] = None
    """Additional interface options. A JSON-serialized object for an `inline keyboard <https://core.telegram.org/bots/features#inline-keyboards>`_, `custom reply keyboard <https://core.telegram.org/bots/features#keyboards>`_, instructions to remove a reply keyboard or to force a reply from the user"""
    allow_sending_without_reply: DefaultAllowSendingWithoutReply = Field(
        None, json_schema_extra={"deprecated": True}
    )
    """Pass :code:`True` if the message should be sent even if the specified replied-to message is not found

.. deprecated:: API:7.0
   https://core.telegram.org/bots/api-changelog#december-29-2023"""
    disable_web_page_preview: Optional[bool] = Field(None, json_schema_extra={"deprecated": True})
    """Disables link previews for links in this message

.. deprecated:: API:7.0
   https://core.telegram.org/bots/api-changelog#december-29-2023"""
    reply_to_message_id: Optional[int] = Field(None, json_schema_extra={"deprecated": True})
    """If the message is a reply, ID of the original message

.. deprecated:: API:7.0
   https://core.telegram.org/bots/api-changelog#december-29-2023"""

    if TYPE_CHECKING:
        # DO NOT EDIT MANUALLY!!!
        # This section was auto-generated via `butcher`

        def __init__(
            __pydantic__self__,
            *,
            chat_id: Union[int, str],
            text: str,
            business_connection_id: Optional[str] = None,
            message_thread_id: Optional[int] = None,
            parse_mode: DefaultParseMode = None,
            entities: Optional[List[MessageEntity]] = None,
            link_preview_options: DefaultLinkPreviewOptions = None,
            disable_notification: DefaultDisableNotification = None,
            protect_content: DefaultProtectContent = None,
            message_effect_id: Optional[str] = None,
            reply_parameters: Optional[ReplyParameters] = None,
            reply_markup: Optional[
                Union[InlineKeyboardMarkup, ReplyKeyboardMarkup, ReplyKeyboardRemove, ForceReply]
            ] = None,
            allow_sending_without_reply: DefaultAllowSendingWithoutReply = None,
            disable_web_page_preview: Optional[bool] = None,
            reply_to_message_id: Optional[int] = None,
            **__pydantic_kwargs: Any,
        ) -> None:
            # DO NOT EDIT MANUALLY!!!
            # This method was auto-generated via `butcher`
            # Is needed only for type checking and IDE support without any additional plugins

            super().__init__(
                chat_id=chat_id,
                text=text,
                business_connection_id=business_connection_id,
                message_thread_id=message_thread_id,
                parse_mode=parse_mode,
                entities=entities,
                link_preview_options=link_preview_options,
                disable_notification=disable_notification,
                protect_content=protect_content,
                message_effect_id=message_effect_id,
                reply_parameters=reply_parameters,
                reply_markup=reply_markup,
                allow_sending_without_reply=allow_sending_without_reply,
                disable_web_page_preview=disable_web_page_preview,
                reply_to_message_id=reply_to_message_id,
                **__pydantic_kwargs,
            )
