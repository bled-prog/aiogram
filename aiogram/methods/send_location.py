from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Union

from pydantic import Field

from ..client.default_annotations import (
    DefaultAllowSendingWithoutReply,
    DefaultDisableNotification,
    DefaultProtectContent,
)
from ..types import (
    ForceReply,
    InlineKeyboardMarkup,
    Message,
    ReplyKeyboardMarkup,
    ReplyKeyboardRemove,
    ReplyParameters,
)
from .base import TelegramMethod


class SendLocation(TelegramMethod[Message]):
    """
    Use this method to send point on the map. On success, the sent :class:`aiogram.types.message.Message` is returned.

    Source: https://core.telegram.org/bots/api#sendlocation
    """

    __returning__ = Message
    __api_method__ = "sendLocation"

    chat_id: Union[int, str]
    """Unique identifier for the target chat or username of the target channel (in the format :code:`@channelusername`)"""
    latitude: float
    """Latitude of the location"""
    longitude: float
    """Longitude of the location"""
    business_connection_id: Optional[str] = None
    """Unique identifier of the business connection on behalf of which the message will be sent"""
    message_thread_id: Optional[int] = None
    """Unique identifier for the target message thread (topic) of the forum; for forum supergroups only"""
    horizontal_accuracy: Optional[float] = None
    """The radius of uncertainty for the location, measured in meters; 0-1500"""
    live_period: Optional[int] = None
    """Period in seconds during which the location will be updated (see `Live Locations <https://telegram.org/blog/live-locations>`_, should be between 60 and 86400, or 0x7FFFFFFF for live locations that can be edited indefinitely."""
    heading: Optional[int] = None
    """For live locations, a direction in which the user is moving, in degrees. Must be between 1 and 360 if specified."""
    proximity_alert_radius: Optional[int] = None
    """For live locations, a maximum distance for proximity alerts about approaching another chat member, in meters. Must be between 1 and 100000 if specified."""
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
            latitude: float,
            longitude: float,
            business_connection_id: Optional[str] = None,
            message_thread_id: Optional[int] = None,
            horizontal_accuracy: Optional[float] = None,
            live_period: Optional[int] = None,
            heading: Optional[int] = None,
            proximity_alert_radius: Optional[int] = None,
            disable_notification: DefaultDisableNotification = None,
            protect_content: DefaultProtectContent = None,
            message_effect_id: Optional[str] = None,
            reply_parameters: Optional[ReplyParameters] = None,
            reply_markup: Optional[
                Union[InlineKeyboardMarkup, ReplyKeyboardMarkup, ReplyKeyboardRemove, ForceReply]
            ] = None,
            allow_sending_without_reply: DefaultAllowSendingWithoutReply = None,
            reply_to_message_id: Optional[int] = None,
            **__pydantic_kwargs: Any,
        ) -> None:
            # DO NOT EDIT MANUALLY!!!
            # This method was auto-generated via `butcher`
            # Is needed only for type checking and IDE support without any additional plugins

            super().__init__(
                chat_id=chat_id,
                latitude=latitude,
                longitude=longitude,
                business_connection_id=business_connection_id,
                message_thread_id=message_thread_id,
                horizontal_accuracy=horizontal_accuracy,
                live_period=live_period,
                heading=heading,
                proximity_alert_radius=proximity_alert_radius,
                disable_notification=disable_notification,
                protect_content=protect_content,
                message_effect_id=message_effect_id,
                reply_parameters=reply_parameters,
                reply_markup=reply_markup,
                allow_sending_without_reply=allow_sending_without_reply,
                reply_to_message_id=reply_to_message_id,
                **__pydantic_kwargs,
            )
