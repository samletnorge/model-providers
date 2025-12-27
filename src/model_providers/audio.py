'''Audio model provider module.

This module introduces lightweight FastRTC wrappers for speech-to-text (STT)
and text-to-speech (TTS).  The FastRTC library is only hinted at to keep the
initial implementation minimal; the actual API calls are stubbed and raise
``NotImplementedError`` so that the code compiles without requiring the
third‑party dependency.  This satisfies the current request of adding a
placeholder for future FastRTC support.

The module exposes two entry points:

* ``FastRTCSTTProvider`` – class with ``transcribe(audio_bytes: bytes)``
* ``FastRTCTTSProvider`` – class with ``synthesize(text: str) -> bytes``

Both classes store the FastRTC config as a dictionary so the actual
implementation can later be swapped in.  They are exported via
``__all__`` so that consumers can import them directly from
``model_providers.audio``.
'''