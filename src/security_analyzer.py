import re
from html.parser import HTMLParser
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse


URL_PATTERN = re.compile(r"(?:https?://|www\.)[^\s<>\"']+", re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w.-]+\.\w+\b", re.IGNORECASE)
TRAILING_URL_PUNCTUATION = ".,;:!?)]}"

PHRASE_RULES = {
    "urgency": {
        "description": "Urgent or time-pressure language",
        "phrases": (
            "urgent",
            "immediately",
            "act now",
            "within 24 hours",
            "expires today",
            "final warning",
            "time sensitive",
        ),
    },
    "credentials": {
        "description": "Credential or account-verification request",
        "phrases": (
            "verify your account",
            "confirm your identity",
            "reset your password",
            "enter your password",
            "login credentials",
            "sign in to verify",
            "validate your account",
            "account verification",
        ),
    },
    "payment": {
        "description": "Payment, invoice, or banking language",
        "phrases": (
            "invoice",
            "payment overdue",
            "wire transfer",
            "bank details",
            "banking information",
            "billing information",
            "payroll",
            "gift card",
            "cryptocurrency",
        ),
    },
    "call_to_action": {
        "description": "Suspicious call-to-action language",
        "phrases": (
            "click here",
            "click the link",
            "open the attachment",
            "download the attachment",
            "reply with",
            "submit your",
            "sign in",
            "log in",
            "scan the qr",
            "call this number",
        ),
    },
    "security_alert": {
        "description": "Impersonation or security-alert language",
        "phrases": (
            "security alert",
            "suspicious login",
            "unusual activity",
            "account locked",
            "account suspended",
            "mailbox quota",
            "it support",
            "help desk",
            "fraud department",
        ),
    },
    "attachment_execution": {
        "description": "Request to execute active attachment content",
        "phrases": ("enable macros", "enable content", "run the attachment"),
    },
    "secrecy": {
        "description": "Request to conceal or bypass normal communication",
        "phrases": ("keep this confidential", "do not tell anyone", "do not call me"),
    },
}


class _LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.active_href: Optional[str] = None
        self.active_text: List[str] = []
        self.links: List[Tuple[str, str]] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag.lower() == "a":
            attributes = dict(attrs)
            self.active_href = attributes.get("href")
            self.active_text = []

    def handle_data(self, data: str) -> None:
        if self.active_href is not None:
            self.active_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "a" and self.active_href is not None:
            self.links.append((self.active_href, " ".join(self.active_text).strip()))
            self.active_href = None
            self.active_text = []


def _unique(values: List[str]) -> List[str]:
    return list(dict.fromkeys(values))


def _extract_urls(text: str, links: List[Tuple[str, str]]) -> List[str]:
    urls = [match.rstrip(TRAILING_URL_PUNCTUATION) for match in URL_PATTERN.findall(text)]
    urls.extend(href for href, _ in links if URL_PATTERN.fullmatch(href))
    return _unique(urls)


def _hostname(value: str) -> Optional[str]:
    candidate = value if "://" in value else f"http://{value}"
    try:
        return urlparse(candidate).hostname
    except ValueError:
        return None


def _domains_match(first: str, second: str) -> bool:
    first = first.lower().removeprefix("www.")
    second = second.lower().removeprefix("www.")
    return first == second or first.endswith(f".{second}") or second.endswith(f".{first}")


def _phrase_indicators(text: str) -> List[Dict[str, object]]:
    indicators = []
    for category, rule in PHRASE_RULES.items():
        evidence = [
            phrase
            for phrase in rule["phrases"]
            if re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text, re.IGNORECASE)
        ]
        if evidence:
            indicators.append(
                {
                    "category": category,
                    "rule": rule["description"],
                    "evidence": evidence,
                }
            )
    return indicators


def _url_indicators(urls: List[str], links: List[Tuple[str, str]]) -> List[Dict[str, object]]:
    indicators = []
    for url in urls:
        try:
            parsed = urlparse(url if "://" in url else f"http://{url}")
        except ValueError:
            indicators.append(
                {
                    "category": "malformed_url",
                    "rule": "URL could not be parsed safely",
                    "evidence": [url],
                }
            )
            continue
        hostname = parsed.hostname or ""
        if parsed.scheme.lower() == "http":
            indicators.append(
                {
                    "category": "insecure_url",
                    "rule": "URL uses unencrypted HTTP",
                    "evidence": [url],
                }
            )
        if hostname and re.fullmatch(r"\d{1,3}(?:\.\d{1,3}){3}", hostname):
            indicators.append(
                {
                    "category": "ip_address_url",
                    "rule": "URL uses an IPv4 address instead of a domain",
                    "evidence": [url],
                }
            )
        if parsed.username or "xn--" in hostname.lower():
            reason = "URL contains user-information syntax" if parsed.username else "URL uses a Punycode domain"
            indicators.append(
                {"category": "obfuscated_url", "rule": reason, "evidence": [url]}
            )

    for href, visible_text in links:
        target_domain = _hostname(href)
        visible_match = URL_PATTERN.search(visible_text)
        visible_domain = _hostname(visible_match.group()) if visible_match else None
        if target_domain and visible_domain and not _domains_match(target_domain, visible_domain):
            indicators.append(
                {
                    "category": "url_domain_mismatch",
                    "rule": "Visible link domain differs from the destination domain",
                    "evidence": [
                        {"visible_domain": visible_domain, "destination_domain": target_domain}
                    ],
                }
            )
    return indicators


def analyze_email(subject: str, body: str) -> Dict[str, object]:
    subject = subject if isinstance(subject, str) else ""
    body = body if isinstance(body, str) else ""
    combined = f"{subject}\n{body}"
    parser = _LinkParser()
    try:
        parser.feed(combined)
        parser.close()
    except (AssertionError, NotImplementedError, TypeError):
        parser.links = []

    urls = _extract_urls(combined, parser.links)
    email_addresses = _unique(EMAIL_PATTERN.findall(combined))
    indicators = _phrase_indicators(combined) + _url_indicators(urls, parser.links)
    categories = _unique([indicator["category"] for indicator in indicators])
    return {
        "indicators": indicators,
        "urls": urls,
        "url_count": len(urls),
        "email_addresses": email_addresses,
        "email_address_count": len(email_addresses),
        "categories": categories,
        "indicator_count": len(indicators),
    }
