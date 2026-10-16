import os
import time

from dotenv import load_dotenv
from google import genai

load_dotenv()


def generate_with_gemini(
    prompt: str,
    system_prompt: str,
    model: str | None = None,
) -> str:
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured in your environment."
        )

    model = model or os.getenv(
        "GEMINI_MODEL", "gemini-3.8-flash"
    )

    client = genai.Client(api_key=api_key)

    # Retry temporary API failures up to 3 times.
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config={"system_instruction": system_prompt},
            )

            answer = (response.text or "").strip()

            if not answer:
                raise ValueError("Gemini returned an empty answer")

            return answer

        except Exception as exc:
            error_message = str(exc).lower()
            error_code = getattr(exc, "code", None)

            retryable_codes = {429, 500, 502, 503, 504}
            retryable_messages = (
                "503",
                "429",
                "500",
                "502",
                "504",
                "unavailable",
                "high demand",
                "resource exhausted",
                "deadline exceeded",
            )

            is_retryable = (
                error_code in retryable_codes
                or any(
                    message in error_message
                    for message in retryable_messages
                )
            )

            # Do not retry permanent errors such as invalid API keys.
            if not is_retryable:
                raise

            # Final attempt failed: preserve the original error.
            if attempt == 2:
                raise

            # Wait 1 second, then 2 seconds, before retrying.
            time.sleep(2 ** attempt)

    raise RuntimeError("Gemini request failed unexpectedly.")
