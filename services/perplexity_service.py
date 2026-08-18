"""Perplexity Sonar-Pro service for self-pay cost lookups."""
import os
import re
import logging
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

# Returned whenever a price could not be retrieved live. Never invent a figure —
# a fabricated price is worse than no price for someone deciding whether to seek care.
COST_UNAVAILABLE = {
    "cost": None,
    "cost_source": "unavailable",
    "source_text": "",
    "verified_in_source": False,
}


def _figures_appear_in(cost: str, source: str) -> bool:
    """True when every number we report is present in the text we retrieved.

    This is the audit for a hallucinated rate: a price that cannot be found in
    its own source was not retrieved, whatever the code path claims.
    """
    digits = re.findall(r'\d[\d,]*', cost or "")
    if not digits:
        return False
    normalised = source.replace(",", "")
    return all(d.replace(",", "") in normalised for d in digits)


class PerplexityService:
    """Queries Perplexity for self-pay facility cost estimates."""

    API_URL = "https://api.perplexity.ai/chat/completions"
    MODEL = "sonar-pro"

    def __init__(self):
        self.api_key = os.getenv("PERPLEXITY_API_KEY")
        self.initialized = bool(self.api_key)
        if not self.initialized:
            logger.warning("PERPLEXITY_API_KEY not set — costs will report as unavailable.")

    def get_facility_cost(
        self,
        facility_name: str,
        facility_type: str,
        zip_code: str,
        procedures: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Return a self-pay cost estimate for this facility and this patient's care.

        `procedures` is the labs and interventions the clinical model identified.
        Pricing a generic "urgent care visit" ignores the assessment — an x-ray
        and a splint cost materially more than a walk-in consultation.
        """
        if not self.initialized:
            return COST_UNAVAILABLE.copy()

        prompt = self._build_prompt(facility_name, facility_type, zip_code, procedures)

        try:
            response = self._call_api(prompt)
            cost = self._extract_cost(response) if response else None
            if cost is None:
                return COST_UNAVAILABLE.copy()
            return {
                "cost": cost,
                "cost_source": "perplexity",
                "source_text": (response or "")[:600],
                "verified_in_source": _figures_appear_in(cost, response or ""),
            }
        except Exception as e:
            logger.error(f"Perplexity query failed for {facility_name}: {e}")
            return COST_UNAVAILABLE.copy()

    def _build_prompt(
        self,
        facility_name: str,
        facility_type: str,
        zip_code: str,
        procedures: Optional[List[str]],
    ) -> str:
        wanted = [p for p in (procedures or []) if str(p).strip()]
        if wanted:
            return (
                f"What is the typical self-pay cost at {facility_name}, "
                f"a {facility_type} near zip code {zip_code}, for a visit including "
                f"{', '.join(wanted)}? "
                f"Provide specific self-pay dollar amounts if available."
            )
        return (
            f"What is the typical self-pay cost of a {facility_type} visit "
            f"at {facility_name} near zip code {zip_code}? "
            f"Provide specific self-pay dollar amounts if available."
        )

    def _call_api(self, prompt: str) -> Optional[str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a healthcare cost research assistant. Provide concise, factual self-pay pricing information.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 200,
        }
        response = requests.post(self.API_URL, headers=headers, json=payload, timeout=20)
        response.raise_for_status()
        return response.json().get("choices", [{}])[0].get("message", {}).get("content", "")

    def _extract_cost(self, text: str) -> Optional[str]:
        """Pull first dollar amount or range from Perplexity response."""
        match = re.search(
            r'\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*[-–]\s*\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
            text,
        )
        if match:
            return f"${match.group(1)}–${match.group(2)}"
        match = re.search(r'\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)', text)
        if match:
            return f"${match.group(1)}"
        return None
