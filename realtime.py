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

        # explicit-commit mode keeps raw input audio until commit/clear
        self.utt_pcm = b""

        # echo guard (the assistant's own voice, via a speakerphone)
        self.echo = EchoGuard()
        self.echo_lock = threading.Lock()
        self.play_until = 0.0

        self.response: tuple | None = None  # (resp_id, item_id, cancel, worker)
        self.pending_text = ""  # user text from conversation.item.create, awaiting response.create

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
            self.send({"type": "conversation.created", "conversation": {"id": self.conversation_id,
                                                                        "object": "realtime.conversation"}})
            while not self.close_evt.is_set():
                raw = await self.ws.receive_text()
                self._dispatch(json.loads(raw))
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

    def _dispatch(self, msg: dict):
        t = msg.get("type")
        try:
            if t == "session.update":
                self._on_session_update(msg.get("session") or {})
            elif t == "input_audio_buffer.append":
                self._on_audio_append(msg.get("audio", ""))
            elif t == "input_audio_buffer.commit":
                self._on_commit()
            elif t == "input_audio_buffer.clear":
                self.utt_pcm = b""
            elif t == "conversation.item.create":
                self._on_item_create(msg.get("item") or {})
            elif t == "response.create":
                self._on_response_create(msg)
            elif t == "response.cancel":
                self._cancel_response("client")
            else:
                self._error("unknown_event", f"unsupported event type: {t!r}")
        except Exception as e:
            log("realtime: dispatch error:", repr(e))
            self._error("invalid_request_error", f"bad {t}: {e}")

    def _on_session_update(self, s: dict):
        if (v := s.get("voice")):
            self.voice = _voice_for(v, E["voices"])
        if (v := s.get("modalities")):
            self.modalities = v or self.modalities
        if "instructions" in s:
            self.instructions = s.get("instructions") or ""
        if (v := s.get("max_output_tokens")):
            self.max_out = int(min(max(64, int(v)), 4096))
        if (v := s.get("temperature")):
            self.temperature = float(v)
        if s.get("model"):
            self.model = s["model"]
        td = s.get("turn_detection")
        self.server_vad = td is not None
        if isinstance(td, dict):
            if (v := td.get("silence_duration_ms")):
                self.end_ms = int(v)
            if (v := td.get("threshold")):
                self.vad_threshold = float(v)
        self.session_obj.update({"model": self.model, "modalities": self.modalities, "voice": self.voice,
                                 "instructions": self.instructions, "temperature": self.temperature,
                                 "max_output_tokens": self.max_out,
                                 "turn_detection": ({"type": "server_vad", "threshold": self.vad_threshold,
                                                     "silence_duration_ms": self.end_ms,
                                                     "prefix_padding_ms": 320}
                                                    if self.server_vad else None)})
        if s.get("tools"):
            log(f"realtime: ignoring {len(s['tools'])} client tool(s) — Ming uses its own toolbox")
        self.send({"type": "session.updated", "session": self.session_obj})

    def _on_audio_append(self, b64: str):
        raw = base64.b64decode(b64)
        raw = raw[:len(raw) // 2 * 2]
        if not raw:
            return
        self.audio_ms += len(raw) // 2 / IN_SR * 1000
        if self.server_vad:
            self.tail = np.concatenate([self.tail,
                                        np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0])
            while len(self.tail) >= FRAME:
                self._on_frame(self.tail[:FRAME].copy())
                self.tail = self.tail[FRAME:]
        else:
            self.utt_pcm += raw  # client VAD: raw buffer, committed explicitly

    def _on_commit(self):
        if self.utt_pcm:
            audio = np.frombuffer(self.utt_pcm, dtype="<i2").astype(np.float32) / 32768.0
            self.utt_pcm = b""
            if len(audio) >= 1600:  # >= 100 ms
                self.send({"type": "input_audio_buffer.committed", "item_id": _oid("item")})
                self._begin_response(audio=audio)

    def _on_item_create(self, item: dict):
        self.send({"type": "conversation.item.created",
                   "item": {**item, "id": item.get("id") or _oid("item"), "status": "completed"}})
        if item.get("type") != "message":
            return
        role, texts = item.get("role"), []
        for c in item.get("content") or []:
            t = c.get("text") or c.get("transcript")
            if t:
                texts.append(t)
        text = " ".join(texts).strip()
        if role == "user" and text:
            self.pending_text = text
            if self.server_vad and self.response is None:  # a typed turn, like a spoken one
                self._begin_response(user_text=self.pending_text)
                self.pending_text = ""
        elif role == "assistant" and text:
            # client-provided assistant context: primed into the conversation history (like the main app's
            # "I don't have access to ..." priming)
            th = E["engine"]["thinker"]
            with E["gpu_lock"]:
                th.history.append({"role": "assistant", "content": text})
                import server  # (lazy: server imports this module)
                server.save_conversation(th)

    def _on_response_create(self, msg: dict):
        user_text = None
        if isinstance(msg.get("item"), dict):
            for c in msg["item"].get("content") or []:
                if (t := c.get("text")):
                    user_text = t
                    break
        if self.response is not None:
            self._cancel_response("new response requested")
        self._begin_response(user_text=user_text or (self.pending_text or None))
        self.pending_text = ""

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
            # Barge-in: sustained, confident speech while it is talking
            self.speaking_frames = self.speaking_frames + 1 if p >= self.barge_threshold else 0
            if self.speaking_frames * FRAME / IN_SR * 1000 >= self.barge_ms:
                self.speaking_frames = 0
                self._cancel_response("barge-in")
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
        self.send({"type": "response.cancelled", "response": {
            "id": resp_id, "object": "realtime.response", "status": "cancelled", "output": []}})
        if reason == "barge-in":
            log("realtime: barged in")

    def _begin_response(self, audio: np.ndarray | None = None, user_text: str | None = None):
        if self.response is not None:
            self._cancel_response("new response requested")
        resp_id, item_id = _oid("resp"), _oid("item")
        cancel = threading.Event()
        worker = threading.Thread(target=self._run_response,
                                  args=(resp_id, item_id, audio, user_text, cancel), daemon=True)
        self.response = (resp_id, item_id, cancel, worker)
        worker.start()

    def _run_response(self, resp_id: str, item_id: str, audio, user_text, cancel: threading.Event):
        th, tk = E["engine"]["thinker"], E["engine"]["talker"]
        toolbox = E.get("toolbox")
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
            res = StreamResampler()    # 44.1 -> 24 kHz for the wire
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
                           "content_index": 0,
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
                                       max_new_tokens=self.max_out, events=True):
                    if isinstance(delta, dict):  # a tool call / its result (Ming runs its own tools)
                        if delta.get("type") == "tool_call" and not text.strip():
                            filler = FILLERS.get(delta.get("name"))
                            if filler:  # it went straight to a tool: say something meanwhile
                                text += filler + " "
                                text_parts.append(filler + " ")
                                self.send({"type": "response.audio_transcript.delta",
                                           "response_id": resp_id, "item_id": item_id,
                                           "content_index": 0, "delta": filler + " "})
                                for piece in fr.push(filler):
                                    frag_q.put(piece)
                        continue
                    text += delta
                    text_parts.append(delta)
                    self.send({"type": "response.audio_transcript.delta", "response_id": resp_id,
                               "item_id": item_id, "content_index": 0, "delta": delta})
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
                   "content_index": 0})
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
                      "total_tokens": th.ctx_len() + max(1, n_out)}}})

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
