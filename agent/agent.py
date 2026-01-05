import os
import asyncio
import logging
from dotenv import load_dotenv

from livekit import agents, rtc
from livekit.agents import AgentServer, AgentSession, Agent, RoomOutputOptions, cli
from livekit.agents.voice import room_io
from livekit.plugins import tavus

load_dotenv(".env.local")

logger = logging.getLogger("agent")
server = AgentServer()


def require_env(*names: str) -> None:
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        raise RuntimeError(
            "Missing required environment variables in agent/.env.local:\n"
            + "\n".join([f"  - {m}" for m in missing])
        )


def on_text(session: AgentSession, ev: room_io.TextInputEvent) -> None:
    msg = (ev.text or "").strip()
    if not msg:
        return
    try:
        session.interrupt()
    except Exception:
        pass
    session.say(msg, allow_interruptions=True)


@server.rtc_session()
async def entrypoint(ctx: agents.JobContext):
    require_env(
        "LIVEKIT_URL",
        "LIVEKIT_API_KEY",
        "LIVEKIT_API_SECRET",
        "TAVUS_API_KEY",
        "TAVUS_REPLICA_ID",
        "TAVUS_PERSONA_ID",
    )

    await ctx.connect()
    logger.info("connected to room: %s", ctx.room.name)

    stt_model = os.environ.get("STT_MODEL", "deepgram/nova-3:multi")
    tts_model = os.environ.get(
        "TTS_MODEL",
        "cartesia/sonic-3:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
    )

    # If finals are slow/nonexistent, these can help endpointing behavior.
    # LiveKit documents these as AgentSession parameters related to endpointing. :contentReference[oaicite:3]{index=3}
    session = AgentSession(
        stt=stt_model,
        tts=tts_model,
        min_endpointing_delay=0.5,
        max_endpointing_delay=6.0,
        use_tts_aligned_transcript=True,
    )

    # Start Tavus avatar worker in the room (audio output is forwarded to the avatar).
    avatar = tavus.AvatarSession(
        replica_id=os.environ["TAVUS_REPLICA_ID"],
        persona_id=os.environ["TAVUS_PERSONA_ID"],
        avatar_participant_name=os.environ.get("TAVUS_AVATAR_NAME", "tavus-avatar"),
    )
    await asyncio.wait_for(avatar.start(session, room=ctx.room), timeout=30)
    logger.info("tavus avatar started")

    # --- AUDIO -> STT -> SPEAK BACK (robust) ---
    loop = asyncio.get_running_loop()
    pending_task: asyncio.Task | None = None
    pending_text: str = ""

    def speak_now(text: str) -> None:
        text = text.strip()
        if not text:
            return
        try:
            session.interrupt()
        except Exception:
            pass
        session.say(text, allow_interruptions=True)

    def schedule_debounced_speak(text: str) -> None:
        nonlocal pending_task, pending_text
        pending_text = text

        if pending_task is not None and not pending_task.done():
            pending_task.cancel()

        async def _later():
            await asyncio.sleep(0.7)  # short silence debounce
            speak_now(pending_text)

        pending_task = asyncio.create_task(_later())

    @session.on("user_input_transcribed")
    def on_transcript(event):
        # LiveKit emits transcript + is_final on this event. :contentReference[oaicite:4]{index=4}
        text = (getattr(event, "transcript", "") or "").strip()
        is_final = bool(getattr(event, "is_final", False))

        # Log everything so you can see whether finals are arriving.
        logger.info("TRANSCRIPT is_final=%s text=%r", is_final, text)

        if not text:
            return

        # Ensure we call speak logic on the main asyncio loop thread.
        if is_final:
            loop.call_soon_threadsafe(speak_now, text)
        else:
            loop.call_soon_threadsafe(schedule_debounced_speak, text)

    # Start session:
    await session.start(
        room=ctx.room,
        agent=Agent(instructions="Echo what the user types or says."),
        room_options=room_io.RoomOptions(
            audio_input=True,  # enables mic ingestion :contentReference[oaicite:5]{index=5}
            text_input=room_io.TextInputOptions(text_input_cb=on_text),
            # Default already includes STANDARD + SIP; being explicit is fine.
            participant_kinds=[rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD],
        ),
        # Tavus integration: disable normal audio output track; avatar handles it. :contentReference[oaicite:6]{index=6}
        room_output_options=RoomOutputOptions(audio_enabled=False),
    )

    logger.info("session started (text + audio ready)")
    session.say("Ready. Speak or type.", allow_interruptions=True)


if __name__ == "__main__":
    cli.run_app(server)
