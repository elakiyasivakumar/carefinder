"""Static telehealth provider list. MedGemma decides if telehealth is appropriate."""
from typing import List
from dataclasses import dataclass

from agents.schemas import TelehealthOption


@dataclass
class TelehealthProvider:
    name: str
    estimated_cost: str
    url: str


TELEHEALTH_PROVIDERS = [
    TelehealthProvider(name="Amazon Health (Amazon Clinic)", estimated_cost="$35–75", url="amazon.com/clinic"),
    TelehealthProvider(name="MDLive",                        estimated_cost="$82",    url="mdlive.com"),
    TelehealthProvider(name="Teladoc",                       estimated_cost="$75",    url="teladoc.com"),
    TelehealthProvider(name="Doctor on Demand",              estimated_cost="$75–99", url="doctorondemand.com"),
]


class TelehealthService:
    def __init__(self):
        self.providers = TELEHEALTH_PROVIDERS

    def get_options_for_care_level(self, care_level: int, force: bool = False) -> List[TelehealthOption]:
        """Return all providers for L2/L3. MedGemma already decided telehealth is appropriate."""
        if care_level >= 4:
            return []
        if care_level >= 3 and not force:
            return []
        return [TelehealthOption(name=p.name, conditions=[], estimated_cost=p.estimated_cost, url=p.url)
                for p in self.providers]
