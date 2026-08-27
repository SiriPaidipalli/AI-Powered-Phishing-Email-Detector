import hashlib
import re
from typing import Callable, Dict, Tuple


EmailParts = Tuple[str, str]
Transformation = Callable[[str, str], EmailParts]


def _replace_phrases(text: str, replacements: Dict[str, str]) -> str:
    for phrase, replacement in replacements.items():
        text = re.sub(re.escape(phrase), replacement, text, flags=re.IGNORECASE)
    return text


def soften_urgency(subject: str, body: str) -> EmailParts:
    replacements = {
        "urgent": "important",
        "immediately": "when possible",
        "act now": "please review",
        "within 24 hours": "soon",
        "expires today": "requires attention",
        "final warning": "final notice",
        "time sensitive": "please review",
    }
    return _replace_phrases(subject, replacements), _replace_phrases(body, replacements)


def soften_credentials(subject: str, body: str) -> EmailParts:
    replacements = {
        "verify your account": "review your account access",
        "confirm your identity": "review your profile details",
        "reset your password": "update your access settings",
        "enter your password": "complete the requested sign-in step",
        "login credentials": "account access details",
        "sign in to verify": "visit the account page to review",
        "validate your account": "review your account status",
        "account verification": "account review",
    }
    return _replace_phrases(subject, replacements), _replace_phrases(body, replacements)


def add_benign_padding(subject: str, body: str, seed: int = 42) -> EmailParts:
    padding = (
        "This message also includes the routine team update for reference.",
        "Please refer to the regular meeting schedule and project notes as needed.",
        "The standard administrative notice is included for completeness.",
    )
    digest = hashlib.sha256(f"{seed}\0{subject}\0{body}".encode("utf-8")).digest()
    selected = padding[int.from_bytes(digest[:2], "big") % len(padding)]
    return subject, f"{body.rstrip()}\n\n{selected}"


def _defang_url(match: re.Match) -> str:
    url = match.group(0)
    url = re.sub(r"^https://", "hxxps://", url, flags=re.IGNORECASE)
    url = re.sub(r"^http://", "hxxp://", url, flags=re.IGNORECASE)
    url = re.sub(r"^www\.", "www[.]", url, flags=re.IGNORECASE)
    return url.replace(".", "[.]")


def obfuscate_urls(subject: str, body: str) -> EmailParts:
    pattern = re.compile(r"(?:https?://|www\.)[^\s<>\"']+", re.IGNORECASE)

    def transform(text: str) -> str:
        text = pattern.sub(_defang_url, text)
        return re.sub(r"\[URL\]", "[U R L]", text, flags=re.IGNORECASE)

    return transform(subject), transform(body)


def obfuscate_whitespace_punctuation(subject: str, body: str) -> EmailParts:
    replacements = {
        "urgent": "u r g e n t",
        "verify": "ver.ify",
        "password": "pass-word",
        "account": "acc ount",
        "login": "log-in",
        "click": "cl.ick",
        "payment": "pay-ment",
        "invoice": "in.voice",
    }
    return _replace_phrases(subject, replacements), _replace_phrases(body, replacements)


def substitute_homoglyphs(subject: str, body: str) -> EmailParts:
    replacements = {
        "urgent": "urgеnt",
        "verify": "vеrify",
        "account": "аccount",
        "password": "pаssword",
        "login": "logіn",
        "payment": "pаyment",
        "invoice": "invоice",
    }
    return _replace_phrases(subject, replacements), _replace_phrases(body, replacements)


def insert_html_noise(subject: str, body: str) -> EmailParts:
    replacements = {
        "urgent": "ur<span></span>gent",
        "verify": "ver<span></span>ify",
        "account": "acc<span></span>ount",
        "password": "pass<span></span>word",
        "login": "log<span></span>in",
        "payment": "pay<span></span>ment",
        "invoice": "inv<span></span>oice",
        "click": "cl<span></span>ick",
    }
    return _replace_phrases(subject, replacements), _replace_phrases(body, replacements)


TRANSFORMATIONS: Dict[str, Transformation] = {
    "urgency_softening": soften_urgency,
    "credential_softening": soften_credentials,
    "benign_text_padding": add_benign_padding,
    "url_obfuscation": obfuscate_urls,
    "whitespace_punctuation_obfuscation": obfuscate_whitespace_punctuation,
    "homoglyph_substitution": substitute_homoglyphs,
    "html_noise_insertion": insert_html_noise,
}
