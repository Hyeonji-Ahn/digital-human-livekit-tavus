import os
import asyncio
import logging

from dotenv import load_dotenv

from livekit import agents
from livekit.agents import AgentServer, AgentSession, Agent, RoomOutputOptions, cli
from livekit.agents.voice import room_io
from livekit.plugins import tavus

# Load environment variables from agent/.env.local
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


def on_text(session, ev):
    msg = (ev.text or "").strip()
    if not msg:
        return
    session.interrupt()
    session.say(msg, allow_interruptions=True)


@server.rtc_session()
async def entrypoint(ctx: agents.JobContext):
    logger.info("1) entrypoint started")

    require_env(
        "LIVEKIT_URL",
        "LIVEKIT_API_KEY",
        "LIVEKIT_API_SECRET",
        "TAVUS_API_KEY",
        "TAVUS_REPLICA_ID",
        "TAVUS_PERSONA_ID",
    )

    await ctx.connect()
    logger.info("2) connected to room")

    # TTS model: keep it configurable.
    # If you don’t have Cartesia via LiveKit Inference, replace this with your own TTS provider.
    tts_model = os.environ.get(
        "TTS_MODEL",
        "cartesia/sonic-3:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
    )

    session = AgentSession(
        tts=tts_model,
        # Helps perceived alignment when transcripts are used (optional)
        use_tts_aligned_transcript=True,
    )
    logger.info("3) AgentSession created")

    # Start Tavus avatar worker that publishes synced audio+video into the room
    avatar = tavus.AvatarSession(
        replica_id=os.environ["TAVUS_REPLICA_ID"],
        persona_id=os.environ["TAVUS_PERSONA_ID"],
        api_key=os.environ["TAVUS_API_KEY"],
        # Optional: avoid identity collisions if you run multiple tests
        avatar_participant_name=os.environ.get("TAVUS_AVATAR_NAME", "tavus-avatar"),
    )
    logger.info("4) AvatarSession created")

    # Avoid hanging forever if Tavus fails
    await asyncio.wait_for(avatar.start(session, room=ctx.room), timeout=30)
    logger.info("5) avatar started and joined")

    # Start session and wire text input callback
    await session.start(
        room=ctx.room,
        agent=Agent(instructions="Repeat exactly what the user types."),
        room_options=room_io.RoomOptions(
            audio_input=False,  # text-only prototype
            text_input=room_io.TextInputOptions(text_input_cb=on_text),
        ),
        # Tavus publishes audio separately; disabling standard audio track is recommended
        room_output_options=RoomOutputOptions(audio_enabled=False),
    )
    logger.info("6) session.start() finished (initialization complete)")

    # Optional greeting:
    await session.say("Ready. Type a message and I will speak it.", allow_interruptions=True)


if __name__ == "__main__":
    cli.run_app(server)
