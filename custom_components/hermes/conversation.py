"""Hermes Assist conversation agent for Home Assistant pipelines.

Registers Hermes as a selectable conversation agent in Home Assistant
Assist, so HA Voice devices can route transcribed text to Hermes and
receive spoken responses through the configured TTS pipeline.
"""

from __future__ import annotations

import dataclasses
import logging
import re
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

# Home Assistant's native agent executes a matched intent inside
# ``async_process``. In answers-only mode, recognise first and only execute
# built-in query intents whose handlers are read-only. This prevents a command
# from running locally before its ACTION_DONE response can be inspected.
_SAFE_LOCAL_QUERY_INTENTS = frozenset(
    {
        "HassClimateGetTemperature",
        "HassGetCurrentDate",
        "HassGetCurrentTime",
        "HassGetState",
        "HassTimerStatus",
    }
)


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


def _tipo_resposta_local(result: ConversationResult) -> str:
    """Classificar uma resposta do agente nativo: "answer", "action" ou "none".

    - `QUERY_ANSWER` com fala é uma resposta informativa (consulta de casa) —
      sem efeitos secundários.
    - `ACTION_DONE` é uma execução (comando de casa).
    - Erros e `NOT_UNDERSTOOD` devolvem "none" e o pedido segue para o Hermes.
    """
    if intent is None:  # pragma: no cover - HA stubs ausentes
        return "none"
    response = getattr(result, "response", None)
    response_type = getattr(response, "response_type", None)
    if response_type == intent.IntentResponseType.QUERY_ANSWER:
        return "answer" if _resposta_texto(result) else "none"
    if response_type == intent.IntentResponseType.ACTION_DONE:
        return "action"
    return "none"


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
        """Tentar resolver o pedido com o agente NATIVO do HA.

        Devolve None quando o pedido não é resolvido localmente, para o Hermes
        tratar dele. Em modo "answers", o agente nativo reconhece primeiro o
        intent sem o executar; apenas intents de consulta conhecidos são então
        processados. Em modo "commands", o utilizador optou explicitamente por
        permitir a execução local de comandos.
        """
        modo = self.local_intents
        if modo == LOCAL_INTENTS_OFF or not text:
            return None

        candidato = _local_candidate(text)
        resultado = await self._tenta_local(user_input, candidato, modo)
        if resultado is None:
            return None

        tipo = _tipo_resposta_local(resultado)
        if tipo == "answer":
            _LOGGER.info("Consulta respondida localmente pelo HA: %r", candidato)
            return resultado
        if tipo == "action" and modo == LOCAL_INTENTS_COMMANDS:
            _LOGGER.warning(
                "Comando de casa executado localmente pelo HA, sem passar "
                "pelo Hermes: %r",
                candidato,
            )
            return resultado
        return None

    async def _tenta_local(
        self, user_input: ConversationInput, text: str, modo: str
    ) -> ConversationResult | None:
        """Tenta UM candidato no agente nativo do HA; None se não resolver."""
        try:
            from homeassistant.components.conversation import async_get_agent

            agent = async_get_agent(self.hass, LOCAL_AGENT_ID)
            if agent is None:
                return None
            local_input = dataclasses.replace(
                user_input, text=text, agent_id=LOCAL_AGENT_ID
            )

            if modo == LOCAL_INTENTS_ANSWERS:
                # ``async_process`` runs the intent before returning its result,
                # so classifying ACTION_DONE afterwards is too late. The native
                # DefaultAgent exposes a side-effect-free recogniser in current
                # Home Assistant. If unavailable, fail closed to Hermes.
                reconhecer = getattr(agent, "async_recognize_intent", None)
                if reconhecer is None:
                    return None
                reconhecimento = await reconhecer(local_input)
                intent_name = getattr(
                    getattr(reconhecimento, "intent", None), "name", None
                )
                if (
                    intent_name not in _SAFE_LOCAL_QUERY_INTENTS
                    or getattr(reconhecimento, "unmatched_entities", ())
                ):
                    return None

            result = await agent.async_process(local_input)
        except Exception as exc:  # noqa: BLE001 - nunca bloquear o Hermes
            _LOGGER.debug("Agente nativo do HA indisponível: %s", exc)
            return None

        if not isinstance(result, ConversationResult):
            return None
        return result

    async def async_process(self, user_input: ConversationInput) -> ConversationResult:
        """Process a conversation input from the Assist pipeline.

        Forwards the user text to Hermes over the WebSocket bridge
        and returns the agent's response.
        """
        # Preserve the integration's established whitespace normalisation and
        # maximum WebSocket payload, while ensuring local cleanup never changes
        # what is sent to Hermes.
        text = (user_input.text or "").strip()
        language = getattr(user_input, "language", None) or "en"

        if not text:
            return self._make_error_result(
                language,
                "I didn't catch that. Could you repeat?",
                user_input.conversation_id,
            )

        # Enforce a reasonable maximum input length to prevent oversized
        # WebSocket frames and memory pressure from developer-tool bypasses.
        allow_local = len(text) <= MAX_QUERY_TEXT_LENGTH
        if not allow_local:
            _LOGGER.warning(
                "Truncating conversation query from %d to %d chars",
                len(text),
                MAX_QUERY_TEXT_LENGTH,
            )
            text = text[:MAX_QUERY_TEXT_LENGTH]

        conversation_id = user_input.conversation_id

        # Pedido de casa? Com o modo local ligado, o HA resolve-o em
        # milissegundos em vez de gastar uma volta completa do Hermes
        # (10-30 s).
        if allow_local:
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
