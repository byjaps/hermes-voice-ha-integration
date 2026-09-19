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

from .const import DOMAIN, MAX_QUERY_TEXT_LENGTH

_LOGGER = logging.getLogger(__name__)

AGENT_ID = f"{DOMAIN}_assist"
AGENT_NAME = "Hermes"

# Agente nativo do HA: resolve comandos de casa localmente, em milissegundos.
# O identificador e o entity_id do agente (nao o slug "home_assistant").
LOCAL_AGENT_ID = "conversation.home_assistant"

# --- Limpeza da transcricao -------------------------------------------------
# O Whisper (whisper.cpp, modelo `medium`, pt-PT) cola lixo no fim da frase
# quando o audio termina em silencio ou ruido: "[musica]", "E ai?", etc.
# Esse lixo nao e inofensivo - faz falhar o matching de intents do proprio HA,
# o que empurra o comando para o agente completo do Hermes (10-30 s) em vez de
# ser resolvido localmente em milissegundos. Medido nesta instalacao:
#   "ligar a luz pequena"          -> HA responde "Ligado"   (0,0 s)
#   "ligar a luz pequena [musica]" -> erro do HA, cai no Hermes (~10 s)
_TRASH_TAIL = re.compile(r"(?:\s*[\[(][^\])]*[\])])+\s*[.,;:!?]*\s*$")
_TRASH_WORDS = re.compile(
    r"(?:\s*[.,;:!?]*\s*\b(?:e\s+a[ií]|n[aã]o\s+[eé]\?|pronto|obrigad[oa]"
    r"|[eé]\s+isso|est[aá]\s+bem|mais\s+alguma\s+coisa))\s*[.,;:!?]*\s*$",
    re.IGNORECASE,
)


def _clean_transcript(text: str) -> str:
    """Remove o lixo que o Whisper cola no fim da frase.

    Devolve sempre texto nao vazio: se a limpeza esvaziar a frase, devolve o
    original - nunca fica pior do que veio.
    """
    out = text.strip()
    for _ in range(3):
        novo = _TRASH_TAIL.sub("", out).strip()
        novo = _TRASH_WORDS.sub("", novo).strip()
        if novo == out:
            break
        out = novo
    return out or text


def _local_candidates(text: str):
    """Candidatos a testar contra o agente nativo, do mais fiel ao mais curto.

    O Whisper nao cola so palavras soltas: cola FRASES inteiras no fim
    ("Desligar a luz pequena. Foi bonito.") - impossivel prever todas. Em vez
    de manter uma lista de lixo, o comando original costuma estar intacto na
    frente da frase, por isso vai-se cortando a ultima oracao e tentando outra
    vez. So se aceita um corte quando o HA confirma a execucao (ACTION_DONE),
    por isso um corte a mais nunca executa nada pela metade.
    """
    yield text
    partes = re.split(r"(?<=[.!?;])\s+", text)
    for n in range(len(partes) - 1, 0, -1):
        candidato = " ".join(partes[:n]).strip()
        if candidato and candidato != text:
            yield candidato
    # Sem pontuacao antes do lixo ("...sala de estar Tchau!"): o corte por
    # frases nao apanha nada, por isso corta-se tambem a ultima palavra (so 2
    # tentativas - cada candidato so executa se o HA confirmar ACTION_DONE).
    palavras = text.split()
    for n in (1, 2):
        if len(palavras) - n >= 2:
            candidato = " ".join(palavras[:-n]).rstrip(" ,;:")
            if candidato and candidato != text:
                yield candidato


def _resposta_local_valida(result: ConversationResult) -> bool:
    """Aceitar respostas de CONSULTA do agente nativo (ex.: temperatura).

    O agente local do HA responde a perguntas de casa em milissegundos
    (`QUERY_ANSWER`), com frases ja traduzidas. Antes so se aceitavam acoes
    (`ACTION_DONE`), portanto perguntas simples ("esta quente a cozinha?",
    "que luzes estao ligadas?", "a janela esta aberta?") gastavam uma volta
    completa do Hermes (~5,5 s). Se o HA nao tiver frase/intent para a
    pergunta, devolve erro ou `NOT_UNDERSTOOD` e o pedido segue para o Hermes.
    """
    if intent is None:  # pragma: no cover - HA stubs ausentes
        return False
    if result.response.response_type != intent.IntentResponseType.QUERY_ANSWER:
        return False
    return bool(_resposta_texto(result))


def _resposta_texto(result: ConversationResult) -> str:
    """Extrair o texto falado de uma resposta de intent (vazio se nao houver)."""
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

    async_add_entities([HermesConversationAgent(bridge, entry.entry_id)])


class HermesConversationAgent(ConversationEntity):
    """Conversation agent that routes Assist pipeline text to Hermes."""

    _attr_has_entity_name = True
    _attr_name = AGENT_NAME
    _attr_icon = "mdi:robot"

    def __init__(self, bridge: Any, entry_id: str) -> None:
        """Initialise the Hermes conversation agent."""
        self._bridge = bridge
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
        """Tenta resolver o comando com o agente NATIVO do HA.

        Devolve None quando o comando nao e um comando de casa, para o Hermes
        tratar dele. So aceita o resultado quando o HA CONFIRMA a execucao
        (ACTION_DONE) - se o comando nao casar, devolve None e nada se perde.
        """
        if not text:
            return None
        for candidato in _local_candidates(text):
            resultado = await self._tenta_local(user_input, candidato)
            if resultado is not None:
                return resultado
        return None

    async def _tenta_local(
        self, user_input: ConversationInput, text: str
    ) -> ConversationResult | None:
        """Tenta UM candidato no agente nativo do HA; None se nao resolver."""
        try:
            from homeassistant.components.conversation import async_get_agent

            agent = async_get_agent(self.hass, LOCAL_AGENT_ID)
            if agent is None:
                return None
            local_input = dataclasses.replace(
                user_input, text=text, agent_id=LOCAL_AGENT_ID
            )
            result = await agent.async_process(local_input)
        except Exception as exc:  # noqa: BLE001 - nunca bloquear o Hermes
            _LOGGER.debug("Agente nativo do HA indisponivel: %s", exc)
            return None

        if not isinstance(result, ConversationResult):
            return None
        if result.response.response_type == intent.IntentResponseType.ACTION_DONE:
            _LOGGER.info("Comando de casa resolvido localmente pelo HA: %r", text)
            return result
        if _resposta_local_valida(result):
            _LOGGER.info("Consulta respondida localmente pelo HA: %r", text)
            return result
        _LOGGER.debug("HA nao resolveu %r localmente; segue para o Hermes", text)
        return None

    async def async_process(self, user_input: ConversationInput) -> ConversationResult:
        """Process a conversation input from the Assist pipeline.

        Forwards the user text to Hermes over the WebSocket bridge
        and returns the agent's response.
        """
        text = (user_input.text or "").strip()
        language = getattr(user_input, "language", None) or "en"

        # Limpar o lixo que o Whisper cola no fim ANTES de decidir: e isso que
        # devolve os comandos de casa ao caminho instantaneo do HA.
        cleaned = _clean_transcript(text)
        if cleaned != text:
            _LOGGER.info("Transcricao limpa: %r -> %r", text, cleaned)
            text = cleaned

        if not text:
            return self._make_error_result(
                language,
                "I didn't catch that. Could you repeat?",
                user_input.conversation_id,
            )

        # Enforce a reasonable maximum input length to prevent oversized
        # WebSocket frames and memory pressure from developer-tool bypasses.
        if len(text) > MAX_QUERY_TEXT_LENGTH:
            _LOGGER.warning(
                "Truncating conversation query from %d to %d chars",
                len(text),
                MAX_QUERY_TEXT_LENGTH,
            )
            text = text[:MAX_QUERY_TEXT_LENGTH]

        conversation_id = user_input.conversation_id

        # Comando de casa? O HA resolve-o localmente em milissegundos. So se
        # nao casar e que vale a pena gastar uma volta completa do Hermes
        # (10-30 s) - e, com o HA a desistir aos 30 s, uma volta longa ainda
        # segura o cadeado da sessao e faz falhar o pedido seguinte.
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
