"""Hermes Assist conversation agent for Home Assistant pipelines.

Registers Hermes as a selectable conversation agent in Home Assistant
Assist, so HA Voice devices can route transcribed text to Hermes and
receive spoken responses through the configured TTS pipeline.
"""

from __future__ import annotations

import dataclasses
import logging
import re
import uuid
from typing import Any

try:
    from homeassistant.components.conversation import (
        ConversationEntity,
        ConversationInput,
        ConversationResult,
    )
    from homeassistant.helpers import intent
except ImportError:  # pragma: no cover - HA stubs / older cores
    ConversationEntity = object  # type: ignore[misc,assignment]
    ConversationInput = object  # type: ignore[misc,assignment]
    ConversationResult = object  # type: ignore[misc,assignment]
    intent = None  # type: ignore[assignment]

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import (
    CONF_LOCAL_INTENTS,
    DEFAULT_LOCAL_INTENTS,
    DOMAIN,
    LOCAL_INTENTS_ANSWERS,
    LOCAL_INTENTS_COMMANDS,
    LOCAL_INTENTS_OFF,
    LOCAL_INTENTS_OPTIONS,
    MAX_QUERY_TEXT_LENGTH,
)

_LOGGER = logging.getLogger(__name__)

AGENT_ID = f"{DOMAIN}_assist"
AGENT_NAME = "Hermes"

# Agente nativo do HA: resolve comandos/consultas de casa em milissegundos.
# É o entity_id do agente (não o slug "home_assistant").
LOCAL_AGENT_ID = "conversation.home_assistant"

# ---------------------------------------------------------------------------
# Limpeza da transcrição
# ---------------------------------------------------------------------------
# O whisper.cpp cola anotações de NÃO-FALA no fim da frase quando o áudio
# termina em silêncio ou ruído ("[música]", "(risos)", "[BLANK_AUDIO]"). Essas
# anotações não são inofensivas: fazem falhar o matching de intents do próprio
# HA, o que empurra um comando de casa para o agente completo do Hermes
# (10-30 s) em vez de ser resolvido localmente em milissegundos.
#
# Só marcadores conhecidos desta lista são removidos — nunca se cortam frases
# ou palavras "a mais". Texto legítimo como "send Sam a notification (urgent)"
# ou "como se diz obrigado?" fica intacto, e o texto ENVIADO AO HERMES é
# sempre o original: a limpeza serve apenas para tentar o caminho local.
_NON_SPEECH_MARKERS = frozenset(
    {
        "applause",
        "aplausos",
        "background noise",
        "blank audio",
        "blank_audio",
        "breathing",
        "cough",
        "inaudible",
        "inaudível",
        "laughter",
        "music",
        "musica",
        "música",
        "musica de fundo",
        "música de fundo",
        "no speech",
        "noise",
        "risadas",
        "risos",
        "ruido",
        "ruído",
        "silence",
        "silencio",
        "silêncio",
        "sneeze",
        "som",
        "sound",
        "static",
    }
)
_MARKER_TAIL = re.compile(r"\s*[\[(]([^\])]*)[\])][\s.,;:!?]*$")

# Only these read-only intent handlers may execute in answers-only mode.
_SAFE_LOCAL_QUERY_INTENTS = frozenset(
    {
        "HassClimateGetTemperature",
        "HassGetCurrentDate",
        "HassGetCurrentTime",
        "HassGetState",
        "HassTimerStatus",
    }
)


def _exclude_non_query_intents(result: Any) -> bool:
    """Return True when HA must not execute this result in answers mode."""
    intent_name = getattr(getattr(result, "intent", None), "name", None)
    return intent_name not in _SAFE_LOCAL_QUERY_INTENTS


def _is_non_speech_marker(inner: str) -> bool:
    """True quando o conteúdo entre parênteses é um marcador de não-fala."""
    valor = inner.strip().strip("_").lower()
    if not valor:
        return True
    return valor in _NON_SPEECH_MARKERS or valor.replace("_", " ") in _NON_SPEECH_MARKERS


def _strip_non_speech_markers(text: str) -> str:
    """Remove marcadores de não-fala no fim da transcrição.

    Devolve sempre texto não vazio: se a limpeza esvaziar a frase, devolve o
    original — nunca fica pior do que veio.
    """
    out = text.strip()
    for _ in range(3):
        match = _MARKER_TAIL.search(out)
        if match is None or not _is_non_speech_marker(match.group(1)):
            break
        out = out[: match.start()].strip()
    return out or text


def _local_candidate(text: str) -> str:
    """Return the sole safe candidate for native Home Assistant handling.

    Only evidence-backed non-speech markers are removed. Arbitrary sentence or
    word truncation is never attempted because even an informational request
    can change meaning when a clause is dropped.
    """
    return _strip_non_speech_markers(text)


def _resposta_texto(result: ConversationResult) -> str:
    """Extrair o texto falado de uma resposta de intent (vazio se não houver)."""
    try:
        speech = result.response.speech.get("plain") or {}
        texto = speech.get("speech") or ""
    except Exception:  # noqa: BLE001 - nunca rebentar por causa do formato
        return ""
    if isinstance(texto, (list, tuple)):
        texto = " ".join(str(p) for p in texto)
    return str(texto).strip()


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the Hermes conversation agent from a config entry."""
    bridge = hass.data.get(DOMAIN, {}).get(entry.entry_id)
    if bridge is None:
        _LOGGER.warning("Hermes conversation: no bridge found for entry %s", entry.entry_id)
        return

    async_add_entities([HermesConversationAgent(bridge, entry.entry_id, entry)])


class HermesConversationAgent(ConversationEntity):
    """Conversation agent that routes Assist pipeline text to Hermes."""

    _attr_has_entity_name = True
    _attr_name = AGENT_NAME
    _attr_icon = "mdi:robot"

    def __init__(
        self,
        bridge: Any,
        entry_id: str,
        entry: ConfigEntry | None = None,
    ) -> None:
        """Initialise the Hermes conversation agent.

        `entry` é guardado para ler as opções a cada pedido: o HA atualiza as
        opções no mesmo objeto, por isso mudar o modo local não exige reiniciar
        o Home Assistant.
        """
        self._bridge = bridge
        self._entry = entry
        self._attr_unique_id = f"{DOMAIN}_{entry_id}_conversation"

    @property
    def supported_languages(self) -> list[str] | str:
        """Return the list of supported languages.

        Returns '*' (string) on HA 2026.6+ to indicate wildcard language
        support, falling back to ['*'] for older HA versions. Hermes
        handles language routing internally based on the model/provider
        configuration.
        """
        return "*"

    @property
    def local_intents(self) -> str:
        """Modo de tratamento local: "off", "answers" ou "commands".

        Por omissão "off": este caminho executa intents do HA dentro do próprio
        HA, sem passar pelo Hermes e portanto sem as listas de bloqueio/
        audit do plugin do Hermes. Só corre quando o utilizador o liga
        explicitamente nas opções da integração (ver README).
        """
        options = getattr(self._entry, "options", None) or {}
        modo = options.get(CONF_LOCAL_INTENTS, DEFAULT_LOCAL_INTENTS)
        return modo if modo in LOCAL_INTENTS_OPTIONS else DEFAULT_LOCAL_INTENTS

    @staticmethod
    def _make_error_result(
        language: str,
        speech: str,
        conversation_id: str | None = None,
    ) -> ConversationResult:
        """Build a ConversationResult with an error speech response."""
        response = intent.IntentResponse(language=language)
        response.async_set_speech(speech)
        return ConversationResult(
            response=response,
            conversation_id=conversation_id,
        )

    async def _try_local_agent(
        self, user_input: ConversationInput, text: str
    ) -> ConversationResult | None:
        """Tentar resolver o pedido pelo caminho estrito de intents do HA.

        Devolve None quando o pedido não é resolvido localmente, para o Hermes
        tratar dele. Este caminho ignora automações de sentence trigger. Em
        modo "answers", um filtro impede a execução de qualquer intent que não
        esteja na lista de consultas conhecidas. Em modo "commands", o
        utilizador optou explicitamente por permitir a execução local dos
        restantes intents reconhecidos.
        """
        modo = self.local_intents
        if modo == LOCAL_INTENTS_OFF or not text:
            return None

        candidato = _local_candidate(text)
        resultado = await self._tenta_local(user_input, candidato, modo)
        if resultado is None:
            return None

        if modo == LOCAL_INTENTS_ANSWERS:
            if not _resposta_texto(resultado):
                return None
            _LOGGER.info("Consulta respondida localmente pelo HA: %r", candidato)
            return resultado

        _LOGGER.warning(
            "Intent do HA processado localmente, sem passar pelo Hermes: %r",
            candidato,
        )
        # A returned response means HA matched and processed the strict intent.
        # Return errors too: falling through after a handler error could execute
        # the same command a second time through Hermes.
        return resultado

    async def _tenta_local(
        self, user_input: ConversationInput, text: str, modo: str
    ) -> ConversationResult | None:
        """Tenta UM candidato pelo caminho estrito de intents do HA."""
        try:
            from homeassistant.components import conversation as ha_conversation
            from homeassistant.components.conversation.chat_log import (
                ChatLog,
                current_chat_log,
            )

            local_input = dataclasses.replace(
                user_input, text=text, agent_id=LOCAL_AGENT_ID
            )
            chat_log = current_chat_log.get()
            if chat_log is None:
                chat_log = ChatLog(
                    self.hass,
                    user_input.conversation_id or f"{DOMAIN}-local-{uuid.uuid4()}",
                )
            handle_intents = ha_conversation.async_handle_intents
        except Exception as exc:  # noqa: BLE001 - caminho local é opcional
            _LOGGER.debug("Agente nativo do HA indisponível: %s", exc)
            return None

        try:
            response = await handle_intents(
                self.hass,
                local_input,
                chat_log,
                intent_filter=(
                    _exclude_non_query_intents
                    if modo == LOCAL_INTENTS_ANSWERS
                    else None
                ),
            )
        except Exception as exc:  # noqa: BLE001 - limite de dupla execução
            if modo == LOCAL_INTENTS_COMMANDS:
                _LOGGER.error(
                    "O intent local pode ter sido executado antes da falha; "
                    "não será reenviado ao Hermes: %s",
                    exc,
                )
                return self._make_error_result(
                    getattr(user_input, "language", None) or "en",
                    "Home Assistant may have processed that command, but its "
                    "response failed. I won't send it again.",
                    user_input.conversation_id,
                )
            _LOGGER.debug("Consulta local do HA falhou: %s", exc)
            return None

        if response is None:
            return None
        return ConversationResult(
            response=response,
            conversation_id=user_input.conversation_id,
        )

    async def async_process(self, user_input: ConversationInput) -> ConversationResult:
        """Process a conversation input from the Assist pipeline.

        Forwards the user text to Hermes over the WebSocket bridge
        and returns the agent's response.
        """
        # Keep the original transcript byte-for-byte for Hermes. Normalisation
        # is only allowed on the isolated local-recognition candidate.
        text = user_input.text or ""
        language = getattr(user_input, "language", None) or "en"
        conversation_id = user_input.conversation_id

        if not text.strip():
            return self._make_error_result(
                language,
                "I didn't catch that. Could you repeat?",
                conversation_id,
            )

        # Never turn an oversized request into a different executable request
        # by truncating it. Reject it before either local or Hermes processing.
        if len(text) > MAX_QUERY_TEXT_LENGTH:
            _LOGGER.warning(
                "Rejecting conversation query of %d chars (maximum %d)",
                len(text),
                MAX_QUERY_TEXT_LENGTH,
            )
            return self._make_error_result(
                language,
                "That request is too long. Please shorten it.",
                conversation_id,
            )

        # Pedido de casa? Com o modo local ligado, o HA resolve-o em
        # milissegundos em vez de gastar uma volta completa do Hermes
        # (10-30 s).
        local_result = await self._try_local_agent(user_input, text)
        if local_result is not None:
            return local_result

        try:
            result = await self._bridge.async_send_conversation_query(
                text=text,
                conversation_id=conversation_id,
                language=language,
            )
        except (ConnectionError, TimeoutError) as exc:
            _LOGGER.warning("Hermes conversation query failed: %s", exc)
            return self._make_error_result(
                language,
                "Sorry, Hermes is not responding right now.",
                conversation_id,
            )
        except Exception as exc:
            _LOGGER.error("Unexpected error in Hermes conversation: %s", exc)
            return self._make_error_result(
                language,
                "Something went wrong. Please try again.",
                conversation_id,
            )

        response_text = result.get("text", "")
        speech_data = result.get("speech", {})
        if speech_data:
            response_speech = speech_data.get("plain", {}).get("speech", response_text)
        else:
            response_speech = response_text

        if not response_speech:
            response_speech = "I processed your request but got no response."

        response = intent.IntentResponse(language=language)
        response.async_set_speech(response_speech)

        return ConversationResult(
            response=response,
            conversation_id=result.get("conversation_id", conversation_id),
        )
