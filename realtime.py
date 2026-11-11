"""OpenAI Realtime-compatible WebSocket endpoint (`/v1/realtime`).

Lets any client that speaks the OpenAI Realtime protocol hold a direct voice conversation with Ming
over one WebSocket — no separate STT/TTS leg. Works with:
  * Open WebUI's "OpenAI Realtime" engine (voice calls / realtime chat),
  * the OpenAI Python SDK with a rewritten websocket base URL,
  * any other realtime client (the protocol is the OpenAI one; the model name is cosmetic).

Wire format (OpenAI Realtime standard):
  * in:  16 kHz mono PCM16, base64, `input_audio_buffer.append`
  * out: 24 kHz mono PCM16, base64, `response.audio.delta`
  * turns: server-side VAD (Silero) + the same smart-turn model as the main app; the assistant can be
    barged in on (its own voice is filtered out by the echo guard).

Conversation state is shared with the main app: one thinker, one conversation. A new /v1/realtime
connection takes the session over the same way a second browser tab does, and the conversation is
persisted to the same file.
"""
from __future__ import annotations

import asyncio
import base64
import json
import queue
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from api import OPENAI_VOICES, StreamResampler
from echo_guard import EchoGuard
from engine import Fragmenter, Talker, log
from tools import FILLERS  # spoken while a tool runs (same fillers as the main app)

router = APIRouter(prefix="/v1")

FRAME = 512          # 32 ms @ 16 kHz (the Silero VAD window)
IN_SR = 16000
OUT_SR = 24000
DEFAULT_MODEL = "ming-3.0-audio-tiny"

TURN_POOL = ThreadPoolExecutor(1)  # smart-turn scoring runs off the event loop

E: dict = {}  # engine, gpu lock, voices, toolbox, vad, turn model — filled by setup()


def _oid(kind: str) -> str:
    return f"{kind}_{uuid.uuid4().hex[:24]}"


def _voice_for(name: str | None, voices: list[str]) -> str:
    if name is not None and name in voices:
        return name
    if name is not None and name in E["engine"].get("custom_voices", {}):
        return name  # a cloned voice: the talker resolves it by name
    if name is not None and name in OPENAI_VOICES and OPENAI_VOICES[name] in voices:
        return OPENAI_VOICES[name]  # OpenAI clients send "alloy"/"nova"/...
    return voices[0]


def setup(engine: dict, gpu_lock: threading.Lock, voices: list[str], toolbox=None, version: str = ""):
    # `engine` is the server's ENGINE dict itself (same object): its "vad_proto" and "turn" slots
    # are filled by the boot thread, so they are read per-connection, not here.
    E.update(engine=engine, gpu_lock=gpu_lock, voices=voices, toolbox=toolbox, version=version)
    log(f"realtime: /v1/realtime ready (version {version or 'dev'}, {len(voices)} voices)"
        + (", smart turn" if engine.get("turn") is not None else ""))


class StreamResampler16:
    """44.1 kHz -> 16 kHz (the echo guard's reference rate)."""

    def __init__(self):
        from torchaudio.transforms import Resample
        self.r = Resample(Talker.SR, 16000)

    def push(self, pcm: np.ndarray) -> np.ndarray:
        return self.r(torch.from_numpy(pcm)).numpy()


class RealtimeSession:
    def __init__(self, ws: WebSocket, loop: asyncio.AbstractEventLoop, model: str = ""):
        self.ws, self.loop = ws, loop
        self.out: asyncio.Queue = asyncio.Queue()
        self.close_evt = threading.Event()

        self.model = model or DEFAULT_MODEL
        self.voice = E["voices"][0]
        self.modalities = ["audio", "text"]
        self.instructions = ""
        self.max_out = 400
        self.temperature = 1.0
        self.server_vad = True  # turn_detection null (client VAD) -> turns only on explicit commit
        self.vad_threshold = 0.5
        self.end_ms = 500
        self.fast_end_ms = 200
        self.slow_end_ms = 1500
        self.short_extra_ms = 250
        self.barge_threshold = 0.75
        self.barge_ms = 250
        self.smart_turn = E["engine"].get("turn") is not None
        self.conversation_id = _oid("conv")
        self.session_id = _oid("sess")
        self.session_obj = {
            "id": self.session_id, "object": "realtime.session", "model": self.model,
            "modalities": self.modalities, "voice": self.voice, "instructions": self.instructions,
            "input_audio_format": "pcm16", "output_audio_format": "pcm16",
            "input_sample_rate": IN_SR, "output_sample_rate": OUT_SR,
            "input_audio_transcription": None, "output_audio_transcription": {"model": "ming"},
            "temperature": self.temperature, "max_output_tokens": self.max_out,
            "turn_detection": {"type": "server_vad", "threshold": self.vad_threshold,
                               "silence_duration_ms": self.end_ms, "prefix_padding_ms": 320},
            "tools": [], "tool_choice": "auto", "reasoning": {"effort": "none"},
            "version": E.get("version", ""),  # which build answered: shown in client logs / health checks
        }

        # VAD / turn state (mirrors server.Session, minus wake word and undo/resume)
        vad_proto = E["engine"].get("vad_proto")
        if vad_proto is None:  # belt and braces: the route only runs once ready, but never KeyError
            from silero_vad import load_silero_vad
            vad_proto = load_silero_vad
        self.vad = vad_proto()  # one Silero instance per session, like the main app
        self.in_speech = False
        self.frames: list[np.ndarray] = []
        self.preroll: list[np.ndarray] = []
        self.speech_frames = 0
        self.pause_frames = 0
        self.speech_rms, self.n_loud = 0.0, 0
        self.noise_rms = 1e-6
        self.turn_fut = None
        self.turn_p = None
        self.speaking_frames = 0
        self.audio_ms = 0  # total input audio received (for speech_started timestamps)
        self.tail = np.zeros(0, dtype=np.float32)
        self.conv_created = False  # conversation.created is deferred: see run()
        self.in_rate = IN_SR       # declared input wire rate; a session.update may say 24 kHz
        self._res_in = None        # in_rate -> 16 kHz resampler, created when needed

        # explicit-commit mode keeps raw input audio until commit/clear
        self.utt_pcm = b""

        # echo guard (the assistant's own voice, via a speakerphone)
        self.echo = EchoGuard()
        self.echo_lock = threading.Lock()
        self.play_until = 0.0

        self.response: tuple | None = None  # (resp_id, item_id, cancel, worker)
        self.pending_text = ""  # user text from conversation.item.create, awaiting response.create
        self.new_proto = False  # client uses the nested audio.* session shape (Open WebUI >= 0.7)
        self.create_response = True  # server_vad: auto-start the response on commit (OWUI 0.7: false -> it sends response.create with the item id)
        self.interrupt_response = True  # server_vad: barge-in cancels the playing response
        self.committed_text = ""  # user transcript for the pending (unanswered) committed turn
        self.commit_item = None  # item id issued on commit, for response.create metadata
        self.out_rate = OUT_SR  # actual reply sample rate (OWUI accepts any rate)

    # ---------- transport ----------

    def _put(self, payload: dict):
        self.loop.call_soon_threadsafe(self.out.put_nowait, payload)

    def send(self, payload: dict):
        self._put(payload)

    def _error(self, code: str, message: str):
        self.send({"type": "error", "error": {"type": "invalid_request_error", "code": code, "message": message}})

    async def run(self):
        sender = asyncio.create_task(self._sender())
        try:
            self.send({"type": "session.created", "session": self.session_obj})
            # conversation.created is deferred on purpose: strict one-message-at-a-time clients
            # (Open WebUI's realtime bridge) must see session.updated as the reply to their
            # session.update, and an interleaved conversation.created breaks the handshake.
            while not self.close_evt.is_set():
                msg = json.loads(await self.ws.receive_text())
                if msg.get("type") != "session.update":
                    self._send_conv_created()  # clients that never send it get it before anything else
                self._dispatch(msg)
        except (WebSocketDisconnect, json.JSONDecodeError) as e:
            log(f"realtime: connection closed ({type(e).__name__})")
        except Exception as e:
            log("realtime: session error:", repr(e))
        finally:
            self.close()
            sender.cancel()
            try:
                await sender
            except BaseException:  # it was cancelled
                pass

    async def _sender(self):
        while True:
            payload = await self.out.get()
            await self.ws.send_text(json.dumps(payload, ensure_ascii=False))

    def close(self):
        self.close_evt.set()
        if self.response is not None:
            self.response[2].set()

    # ---------- events in ----------

    def _send_conv_created(self):
        if not self.conv_created:
            self.conv_created = True
            self.send({"type": "conversation.created", "conversation": {"id": self.conversation_id,
                                                                        "object": "realtime.conversation"}})

    def _dispatch(self, msg: dict):
        t = msg.get("type")
        try:
            if t == "session.update":
                self._on_session_update(msg.get("session") or {})
            elif t == "input_audio_buffer.append":
                self._on_audio_append(msg.get("audio") or "")
            elif t == "input_audio_buffer.commit":
                self._on_commit()
            elif t == "input_audio_buffer.clear":
                self._on_clear()
            elif t == "conversation.item.create":
                self._on_item_create(msg.get("item") or {})
            elif t == "conversation.item.delete":
                self._on_item_delete(msg.get("item_id"))
            elif t == "conversation.item.truncate":
                self._on_item_truncate(msg.get("item_id"), msg.get("content_index", 0),
                                        msg.get("audio_end_ms"))
            elif t == "session.truncate":
                log(f"realtime: session.truncate ignored (context is rebuilt per turn)")
            elif t == "response.create":
                self._on_response_create(msg)
            elif t == "response.cancel":
                self._cancel_response("client")
            elif t.startswith(("session.", "conversation.item.", "response.", "input_audio_buffer.")):
                pass  # accepted and ignored (protocol completeness: item.created/removed, output events)
            else:
                self._error("unknown_event", f"unsupported event type: {t!r}")
        except Exception as e:
            log("realtime: dispatch error:", repr(e))
            self._error("invalid_request_error", f"bad {t}: {e}")

    def _on_session_update(self, s: dict):
        # Two wire shapes: classic top-level (voice, modalities, turn_detection, ...) and the nested
        # audio.input/audio.output shape that Open WebUI >= 0.7 sends (24 kHz in/out, server_vad with
        # create_response/interrupt_response flags, transcription.model). Top-level values win when set.
        a_in = (s.get("audio") or {}).get("input") or {}
        a_out = (s.get("audio") or {}).get("output") or {}
        if a_in or a_out:
            self.new_proto = True
        fmt = (a_in.get("format") or {}).get("rate") if isinstance(a_in.get("format"), dict) else a_in.get("sample_rate")
        self.in_rate = int(fmt) if fmt else IN_SR
        fmt_out = (a_out.get("format") or {}).get("rate") if isinstance(a_out.get("format"), dict) else None
        if fmt_out:  # OWUI declares 24 kHz; we answer at the engine's native rate and say so in audio.delta
            self.out_rate = OUT_SR
        self.model = s.get("model") or self.model
        self.voice = s.get("voice") or a_out.get("voice") or self.voice
        mods = s.get("modalities") or s.get("output_modalities")
        if mods and "audio" in mods:
            self.modalities = list(mods)
        if s.get("instructions"):  # the client's system prompt: used as the context on each new turn
            self.chat_context = s["instructions"]
        self.max_out = int(s.get("max_output_tokens") or self.max_out)
        if s.get("temperature") is not None:
            self.temperature = float(s["temperature"])
        td = s.get("turn_detection", a_in.get("turn_detection"))  # top-level presence wins (even null)
        if isinstance(td, dict):
            self.server_vad = td.get("type") in (None, "server_vad", "server_vad_beta")
            if "silence_duration_ms" in td:
                self.end_ms = int(td["silence_duration_ms"])
            if "threshold" in td:
                self.vad_threshold = float(td["threshold"])
            self.create_response = bool(td.get("create_response", True))  # false (OWUI 0.7): it sends response.create
            self.interrupt_response = bool(td.get("interrupt_response", True))  # false: no barge-in while it speaks
        trans = a_in.get("transcription") or {}  # OWUI sets a transcription model; Ming transcribes with its own Whisper
        if trans:
            log(f"realtime: transcription {trans!r} requested; Ming uses its own Whisper")
        if s.get("tools"):
            log(f"realtime: ignoring {len(s['tools'])} client tool(s) — Ming uses its own toolbox")
        self.session_obj = {"id": self.session_id, "object": "realtime.session", "model": self.model,
                           "voice": self.voice, "modalities": self.modalities,
                           "instructions": s.get("instructions"),
                           "input_audio_format": f"pcm16:{self.in_rate}",
                           "output_audio_format": f"pcm16:{self.out_rate}",
                           "max_output_tokens": self.max_out, "temperature": self.temperature,
                           "turn_detection": ({"type": "server_vad", "threshold": self.vad_threshold,
                                               "silence_duration_ms": self.end_ms,
                                               "prefix_padding_ms": 320,
                                               "create_response": self.create_response,
                                               "interrupt_response": self.interrupt_response}
                                              if self.server_vad else None)}
        if self.new_proto:  # echo the shape the client sent, so its parser is happy
            self.session_obj["output_modalities"] = self.modalities
            self.session_obj["audio"] = {"input": {"format": {"type": "audio/pcm", "rate": self.in_rate},
                                              "transcription": trans or {"model": "ming"},
                                              "turn_detection": self.session_obj["turn_detection"]},
                                          "output": {"format": {"type": "audio/pcm", "rate": self.out_rate},
                                                    "voice": self.voice}}
        self.send({"type": "session.updated", "session": self.session_obj})
        self._send_conv_created()  # reply to the update first; Open WebUI breaks otherwise

    def _on_audio_append(self, b64: str):
        raw = base64.b64decode(b64)
        raw = raw[:len(raw) // 2 * 2]
        if not raw:
            return
        self.audio_ms += len(raw) // 2 / self.in_rate * 1000
        if self.in_rate != IN_SR:  # e.g. Open WebUI declares 24 kHz; the VAD/ASR run at 16 kHz
            if self._res_in is None:
                self._res_in = StreamResampler(self.in_rate, IN_SR)
            x = self._res_in.push(np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0)
            raw = (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
        if self.server_vad:
            self.tail = np.concatenate([self.tail,
                                        np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0])
            while len(self.tail) >= FRAME:
                self._on_frame(self.tail[:FRAME].copy())
                self.tail = self.tail[FRAME:]
        else:
            self.utt_pcm += raw  # client VAD: raw buffer, committed explicitly

    def _on_commit(self):
        if self.response is not None:  # a response is already running: this is its user audio
            self.committed_audio = self._utterance()
            return
        if len(self.tail) < 0.1 * IN_SR:
            return  # sub-100 ms noise floor
        audio = self._utterance()
        self._reset_listen()
        self.send({"type": "input_audio_buffer.committed"})
        self.send({"type": "input_audio_buffer.speech_stopped", "audio_duration_ms": int(len(audio) / IN_SR * 1000)})
        if self.create_response:  # classic OpenAI: the server starts the response itself
            self._begin_response(audio=audio)
        else:  # OWUI 0.7: the client sends response.create {metadata:{input_item_id}} when the transcript lands
            self.commit_item = self._commit_item_id = _oid("item")
            self.send({"type": "conversation.item.created", "item": {
                "id": self.commit_item, "object": "realtime.item", "type": "message", "status": "completed",
                "role": "user", "content": [{"type": "input_audio", "transcript": ""}]}})  # the ASR fill-in follows

    def _on_item_create(self, item: dict):
        role = item.get("role")
        texts: list[str] = []
        for c in item.get("content") or []:
            t = c.get("text") or c.get("transcript")
            if t:
                texts.append(t)
        text = " ".join(texts).strip()
        item_id = item.get("id") or _oid("item")
        status = item.get("status") or "completed"
        if status != "completed":
            # OWUI marks a typed item in_progress while it streams: tell the UI it is being handled
            self.send({"type": "conversation.item.created", "item": {"id": item_id,
                "object": "realtime.item", "type": "message", "status": status, "role": role, "content": []}})
            if status == "in_progress":
                self.send({"type": "response.audio_transcript.delta", "response_id": "", "item_id": item_id,
                           "content_index": 0, "delta": ""})  # typing indicator, zero-length on purpose
            return
        if role == "system" and text:
            # OWUI 0.7 re-sends the system context as a system item before each turn
            self.chat_context = text
        elif role == "assistant" and text:
            # client-provided assistant context: primed into the conversation history (like the main app's
            # "I don't have access to ..." priming)
            th = E["engine"]["thinker"]
            with E["gpu_lock"]:
                th.history.append({"role": "assistant", "content": text})
                import server  # (lazy: server imports this module)
                server.save_conversation(th)
        elif role == "user" and text:
            self.pending_text = text
            if self.server_vad and self.create_response and self.response is None:  # a typed turn, like a spoken one
                self._begin_response(user_text=self.pending_text)
                self.pending_text = ""

    def _on_response_create(self, msg: dict):
        user_text = None
        if isinstance(msg.get("item"), dict):
            for c in msg["item"].get("content") or []:
                if (t := c.get("text")):
                    user_text = t
                    break
        resp = msg.get("response") or {}
        meta = resp.get("metadata") or {}
        if resp.get("instructions"):
            self.chat_context = resp["instructions"]
        item_id = meta.get("input_item_id")
        if item_id == self.commit_item and not user_text:
            # OWUI 0.7 (create_response=false): this is the reply request for the turn we just committed —
            # answer with its transcript (or the voice turn itself); OWUI would otherwise sit in silence.
            user_text = self.committed_text or None
        if self.response is not None:
            self._cancel_response("new response requested")
        self._begin_response(user_text=user_text or (self.pending_text or None), audio=None)
        self.pending_text = ""
        self.commit_item = None

    def _on_item_delete(self, item_id):
        # OWUI 0.7 deletes stale items; Ming's context is rebuilt per turn, so there is nothing to remove
        log(f"realtime: conversation.item.delete {item_id} (no-op)")

    def _on_item_truncate(self, item_id, content_index, audio_end_ms):
        # voice was cut short: drop the trailing part of that turn's recorded audio (best effort)
        log(f"realtime: conversation.item.truncate {item_id} idx={content_index} end={audio_end_ms} ms (no-op: per-turn context)")

    # ---------- VAD, echo, barge-in (event loop, 32 ms cadence) ----------

    def _is_echo(self, f: np.ndarray) -> bool:
        now = time.time()
        with self.echo_lock:
            if not self.echo.active(now):
                return False
            self.echo.push_mic(f, now)
            return self.echo.is_echo(now)

    def _on_frame(self, f: np.ndarray):
        p = float(self.vad(torch.from_numpy(f), IN_SR).item())
        if p > 0 and self._is_echo(f):
            p = 0.0

        if self.response is not None and self.response[3].is_alive() and not self.response[2].is_set():
            if self.in_speech:  # collect the barge turn like any other (the response ends, it starts)
                self.frames.append(f)
                self.speech_frames += 1
                self.pause_frames = 0
            if self.interrupt_response:  # OWUI may disable barge-in (interrupt_response=false)
                self.speaking_frames = self.speaking_frames + 1 if p >= self.barge_threshold else 0
                if self.speaking_frames * FRAME / IN_SR * 1000 >= self.barge_ms:
                    self.speaking_frames = 0
                    self._cancel_response("barge-in")
            else:  # it is talking: gather the new turn in the pre-roll, start it on the next threshold hit
                self.preroll.append(f)
                self.preroll = self.preroll[-10:]
                if p >= self.vad_threshold and not self.in_speech:
                    self.in_speech = True
                    self.frames = list(self.preroll)
                    self.speech_frames, self.pause_frames = 1, 0
                    self.speech_rms, self.n_loud = 0.0, 0
                    self.turn_fut, self.turn_p = None, None
                    self.send({"type": "input_audio_buffer.speech_started",
                               "audio_start_ms": int(self.audio_ms - 320)})
            return

        if not self.in_speech:
            rms = float(np.sqrt(np.mean(f * f)))
            self.noise_rms += 0.05 * (rms - self.noise_rms)
            self.preroll.append(f)
            self.preroll = self.preroll[-10:]  # ~320 ms of prefix padding
            if p >= self.vad_threshold:
                self.in_speech = True
                self.frames = list(self.preroll)
                self.speech_frames, self.pause_frames = 1, 0
                self.speech_rms, self.n_loud = 0.0, 0
                self.turn_fut, self.turn_p = None, None
                self.send({"type": "input_audio_buffer.speech_started",
                           "audio_start_ms": int(self.audio_ms - 320)})
            return

        self.frames.append(f)
        rms = float(np.sqrt(np.mean(f * f)))
        quiet = self.noise_rms > 0 and rms / self.noise_rms < 1.2
        released = p < max(0.3, self.vad_threshold - 0.2)
        soft_speech = p < 0.5 and rms < self.speech_rms * 0.5

        if not (quiet and released) and not soft_speech:
            self.speech_frames += 1
            self.speech_rms += 0.02 * (rms - self.speech_rms)
            self.n_loud += 1
            if self.pause_frames:
                self.pause_frames = 0
            return

        self.pause_frames += 1
        if self.pause_frames > 62:  # > 2 s of pause: the turn is over even if unfinished
            self.pause_frames = 62
        silence_ms = int(self.pause_frames * FRAME / IN_SR * 1000)
        speech_s = self.speech_frames * FRAME / IN_SR
        long_enough = speech_s >= 0.25
        if silence_ms >= self._end_of_turn_ms(silence_ms, speech_s, long_enough):
            audio = self._utterance()
            self._reset_listen()
            self.send({"type": "input_audio_buffer.speech_stopped",
                       "audio_duration_ms": int(speech_s * 1000)})
            if long_enough:
                if self.create_response:  # classic OpenAI: respond automatically
                    self._begin_response(audio=audio)
                # else: OWUI 0.7 — it sends response.create with metadata.input_item_id once the transcript
                # is final (its protocol always does, and it needs the item id for that message)
                self._begin_response(audio=audio)

    def _end_of_turn_ms(self, silence_ms: int, speech_s: float, long_enough: bool) -> int:
        extra = self.short_extra_ms if speech_s < 1.0 else 0
        det = E["engine"].get("turn")
        if not (self.smart_turn and det is not None and long_enough):
            return self.end_ms + extra
        if self.turn_fut is None and silence_ms >= 64:
            self.turn_fut = TURN_POOL.submit(det.prob, np.concatenate(self.frames))
        if self.turn_p is None and self.turn_fut is not None and self.turn_fut.done():
            try:
                self.turn_p = self.turn_fut.result()
            except Exception as e:
                log("realtime: smart turn error:", repr(e))
                self.turn_p = 0.5
        p = self.turn_p
        if p is None:
            return self.end_ms + extra
        if p >= 0.8:
            return min(self.end_ms, self.fast_end_ms) + extra
        if p < 0.3:
            return max(self.end_ms, self.slow_end_ms)
        return self.end_ms + extra

    def _utterance(self) -> np.ndarray:
        audio = np.concatenate(self.frames)
        keep = len(audio) - int(max(0, self.pause_frames * FRAME - 150 * 16))  # keep ~150 ms of silence
        return audio[:max(keep, FRAME)]

    def _reset_listen(self):
        self.in_speech = False
        self.frames = []
        self.speech_frames, self.pause_frames = 0, 0
        self.speech_rms, self.n_loud = 0.0, 0
        self.turn_fut, self.turn_p = None, None

    # ---------- the response ----------

    def _cancel_response(self, reason: str):
        r = self.response
        if r is None:
            return
        resp_id, item_id, cancel, worker = r
        cancel.set()
        with self.echo_lock:
            self.echo.truncate(time.time() + 0.25)  # playback stops; what is in flight still echoes
        self.play_until = 0.0
        self.speaking_frames = 0
        self.committed_text = ""  # a barge-in abandons the interrupted turn
        self.send({"type": "response.cancelled", "response": {
            "id": resp_id, "object": "realtime.response", "status": "cancelled", "output": []}})
        if reason == "barge-in":
            log("realtime: barged in")

    def _begin_response(self, audio: np.ndarray | None = None, user_text: str | None = None):
        self.response = None  # close the previous slot before the new thread can race it
        cancel = threading.Event()
        resp_id, item_id = _oid("resp"), _oid("item")
        worker = threading.Thread(target=self._run_response, args=(resp_id, item_id, audio, user_text, cancel),
                                  daemon=True)
        worker.start()
        self.response = (resp_id, item_id, cancel, worker)

    def _run_response(self, resp_id: str, item_id: str, audio, user_text, cancel: threading.Event):
        th, tk = E["engine"]["thinker"], E["engine"]["talker"]
        toolbox = E.get("toolbox")
        # snapshot the per-session context (system prompt / item priming); OWUI 0.7 updates it every turn
        dialect = E.get("dialect")
        chat_context = (self.chat_context or "").strip()
        if dialect and chat_context and dialect not in chat_context:  # dialect last: it wins over the prompt
            chat_context = f"{chat_context}\n\n{dialect}"
        self.send({"type": "response.created", "response": {
            "id": resp_id, "object": "realtime.response", "status": "in_progress",
            "model": self.model, "modalities": self.modalities, "voice": self.voice, "output": []}})
        self.send({"type": "response.output_item.added", "response_id": resp_id, "output_index": 0,
                   "item": {"id": item_id, "object": "realtime.item", "type": "message",
                            "status": "in_progress", "role": "assistant", "content": []}})
        self.send({"type": "response.content_part.added", "response_id": resp_id, "item_id": item_id,
                   "output_index": 0, "content_index": 0, "part": {"type": "audio", "transcript": ""}})

        voice = E["engine"]["custom_voices"].get(self.voice, self.voice)
        frag_q: queue.Queue = queue.Queue()
        fr = Fragmenter()
        text_parts: list[str] = []
        n_samples = 0

        def talker_loop():
            nonlocal n_samples
            res = StreamResampler()    # 44.1 -> wire rate (24 kHz; the session can request another — audio.delta says which)
            ref = StreamResampler16()  # 44.1 -> 16 kHz for the echo guard
            while True:
                item = frag_q.get()
                if item is None or cancel.is_set():
                    break
                pieces = [item]
                while True:  # merge whatever is already queued (talker behind -> bigger pieces)
                    try:
                        nxt = frag_q.get_nowait()
                    except queue.Empty:
                        break
                    if nxt is None:
                        frag_q.put(None)
                        break
                    pieces.append(nxt)
                try:
                    for pcm in tk.synth_stream(" ".join(pieces), voice, cancel=cancel):
                        if cancel.is_set():
                            break
                        y24 = res.push(pcm)  # 24 kHz, per the session's output format
                        if y24.size:
                            n_samples += len(y24)
                            with self.echo_lock:
                                start = max(self.play_until, time.time())  # chain contiguously (the client keeps its own jitter buffer)
                                self.play_until = start + len(pcm) / Talker.SR
                                self.echo.add_playback(ref.push(pcm), start)
                            self.send({"type": "response.audio.delta", "response_id": resp_id,
                                       "item_id": item_id, "content_index": 0,
                                       "sample_rate": self.out_rate,
                                       "delta": base64.b64encode(
                                           (np.clip(y24, -1, 1) * 32767).astype("<i2").tobytes()).decode()})
                except Exception as e:
                    log("realtime: talker error:", repr(e))
                    import server  # (lazy: server imports this module)
                    server.check_fatal(e)
                    self._error("internal_error", f"talker: {e}")
            try:  # flush the resampler's ~3 ms of history so the reply doesn't end early (a silent probe sample)
                y24 = res.push(np.zeros(1, np.float32), final=True)
            except Exception:
                y24 = np.zeros(0, np.float32)
            if y24.size and not cancel.is_set():
                n_samples += len(y24)
                self.send({"type": "response.audio.delta", "response_id": resp_id, "item_id": item_id,
                           "content_index": 0, "sample_rate": self.out_rate,
                           "delta": base64.b64encode(
                               (np.clip(y24, -1, 1) * 32767).astype("<i2").tobytes()).decode()})

        tt = threading.Thread(target=talker_loop, daemon=True)
        tt.start()

        text = ""
        try:
            with E["gpu_lock"]:
                if toolbox is not None:
                    th.set_tools(toolbox)
                for delta in th.stream(audio, cancel=cancel, user_text=user_text,
                                       max_new_tokens=self.max_out, events=True, history_text=chat_context):
                    if isinstance(delta, dict):  # a tool call / its result (Ming runs its own tools)
                        if delta.get("type") == "tool_call" and not text.strip():
                            filler = FILLERS.get(delta.get("name"))
                            if filler:  # it went straight to a tool: say something meanwhile
                                text += filler + " "
                                text_parts.append(filler + " ")
                                self.send({"type": "response.audio_transcript.delta",
                                           "response_id": resp_id, "item_id": item_id,
                                           "content_index": 0, "delta": filler + " "})
                                self.send({"type": "response.status", "response_id": resp_id,
                                           "status": "status.vad_grace_period"})  # typing indicator (OWUI 0.7)
                                for piece in fr.push(filler):
                                    frag_q.put(piece)
                        continue
                    text += delta
                    text_parts.append(delta)
                    self.send({"type": "response.audio_transcript.delta", "response_id": resp_id,
                               "item_id": item_id, "content_index": 0, "delta": delta})
                    self.send({"type": "response.status", "response_id": resp_id,
                               "status": "status.vad_grace_period"})  # typing indicator (OWUI 0.7)
                    for piece in fr.push(delta):
                        frag_q.put(piece)
        except Exception as e:
            log("realtime: thinker error:", repr(e))
            import server  # (lazy: server imports this module)
            server.check_fatal(e)
            self._error("internal_error", f"thinker: {e}")
        finally:
            for piece in fr.flush():
                frag_q.put(piece)
            frag_q.put(None)
        tt.join()

        full = "".join(text_parts)
        if cancel.is_set():
            return  # response.cancelled was already sent
        n_out = len(th.tok(full).input_ids) if full else 0
        self.send({"type": "response.audio_transcript.done", "response_id": resp_id, "item_id": item_id,
                   "content_index": 0, "transcript": full})
        self.send({"type": "response.audio.done", "response_id": resp_id, "item_id": item_id,
                   "content_index": 0, "sample_rate": self.out_rate})
        self.send({"type": "response.status", "response_id": resp_id,
                   "status": "status.response_done"})  # typing indicator off (OWUI 0.7)
        self.send({"type": "response.content_part.done", "response_id": resp_id, "item_id": item_id,
                   "output_index": 0, "content_index": 0, "part": {"type": "audio", "transcript": full}})
        self.send({"type": "response.output_item.done", "response_id": resp_id, "output_index": 0,
                   "item": {"id": item_id, "object": "realtime.item", "type": "message",
                            "status": "completed", "role": "assistant",
                            "content": [{"type": "audio", "transcript": full, "id": _oid("audio")}]}})
        self.send({"type": "response.completed", "response": {
            "id": resp_id, "object": "realtime.response", "status": "completed", "model": self.model,
            "voice": self.voice,
            "output": [{"id": item_id, "type": "message", "role": "assistant",
                        "content": [{"type": "audio", "transcript": full}]}],
            "usage": {"input_tokens": th.ctx_len(), "output_tokens": max(1, n_out),
                      "total_tokens": th.ctx_len() + max(1, n_out)}  # OWUI 0.7 requires usage in response.completed
        }})  # closes the response dict + the send dict, then the send call

        # background ASR of the user's utterance (same as the main app): the transcript for the record
        if audio is not None and len(audio) > 4000:
            def asr():
                try:
                    with E["gpu_lock"]:
                        t = th.transcribe(audio, max_new_tokens=160)
                except Exception as e:
                    log("realtime: transcribe error:", repr(e))
                    return
                if t:
                    log(f"realtime heard: {t!r}")
                    if th.history:
                        th.history[-1]["user"] = t
            threading.Thread(target=asr, daemon=True).start()

        import server  # (lazy: server imports this module)
        with E["gpu_lock"]:
            server.save_conversation(th)
            if th.needs_compact():
                th.compact()
        if self.response is not None and self.response[0] == resp_id:
            self.response = None


async def _take_over(old):
    """Tell the session we replace that it is being replaced, then close it - on a short leash.
    The old client may be suspended (a mobile tab) and never drain our write, so a stuck send must
    not block the connection that replaces it (fire-and-forget, 3 s cap)."""
    try:
        await asyncio.wait_for(old.ws.send_text(json.dumps({"type": "taken_over", "by": "realtime"})), timeout=3)
    except Exception:
        pass
    try:
        old.close()
    except Exception:
        pass


@router.websocket("/realtime")
async def realtime(ws: WebSocket, model: str = ""):
    from server import ACTIVE
    await ws.accept()
    if not E.get("engine", {}).get("ready"):
        # model still loading (right after a restart / image update): the ENGINE dict is only fully
        # populated at the end of boot, and building a session against it would crash (KeyError).
        # Reject cleanly - OpenAI-realtime clients retry on close code 1013 (Try Again Later).
        log("realtime: connection refused - model still loading")
        try:
            await ws.send_text(json.dumps({"type": "error",
                                           "error": {"type": "api_error", "code": "model_not_ready",
                                                     "version": E.get("version", ""),
                                                     "message": "the model is still loading - try again shortly"}}))
        except Exception:
            pass
        await ws.close(code=1013)
        return
    sess = RealtimeSession(ws, asyncio.get_running_loop(), model=model)
    old = ACTIVE.get("sess")
    if old is not None and old is not sess:
        asyncio.ensure_future(_take_over(old))  # never awaited here: see _take_over
    ACTIVE["sess"] = sess
    try:
        await sess.run()
    finally:
        if ACTIVE.get("sess") is sess:
            ACTIVE.pop("sess", None)
