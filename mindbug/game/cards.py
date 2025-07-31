from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from .constants import Keyword, TriggerType


@dataclass
class Card:
    # Represents a card with its base stats and abilities
    name: str
    power: int
    keywords: Set[Keyword] = field(default_factory=set)
    ability_text: str = ""
    ability_trigger: Optional[TriggerType] = None

    def __hash__(self):
        # Cards are uniquely identified by name and power
        return hash((self.name, self.power))

    def __eq__(self, other):
        if not isinstance(other, Card):
            return False
        return self.name == other.name and self.power == other.power


class CardDefinitions:
    # Static definitions for all 32 First Contact cards
    
    @staticmethod
    def get_first_contact_cards() -> Dict[str, Card]:
        # Returns dictionary of all unique cards in First Contact
        cards = {
            "Axolotl Healer": Card(
                name="Axolotl Healer",
                power=4,
                keywords={Keyword.POISONOUS},
                ability_text="Play: Gain 2 life.",
                ability_trigger=TriggerType.PLAY,
            ),
            "Bee Bear": Card(
                name="Bee Bear",
                power=8,
                keywords=set(),
                ability_text="Cannot be blocked by creatures with power 6 or less.",
                ability_trigger=TriggerType.PASSIVE,
            ),
            "Brain Fly": Card(
                name="Brain Fly",
                power=4,
                keywords=set(),
                ability_text="Play: Take control of a creature with power 6 or more.",
                ability_trigger=TriggerType.PLAY,
            ),
            "Chameleon Sniper": Card(
                name="Chameleon Sniper",
                power=1,
                keywords={Keyword.SNEAKY},
                ability_text="Attack: The opponent loses 1 life.",
                ability_trigger=TriggerType.ATTACK,
            ),
            "Compost Dragon": Card(
                name="Compost Dragon",
                power=3,
                keywords={Keyword.HUNTER},
                ability_text="Play: Play a card from your discard pile.",
                ability_trigger=TriggerType.PLAY,
            ),
            "Deathweaver": Card(
                name="Deathweaver",
                power=2,
                keywords={Keyword.POISONOUS},
                ability_text="The opponent cannot activate Play effects.",
                ability_trigger=TriggerType.PASSIVE,
            ),
            "Elephantopus": Card(
                name="Elephantopus",
                power=7,
                keywords={Keyword.TOUGH},
                ability_text="The opponent cannot block with creatures with power 4 or less.",
                ability_trigger=TriggerType.PASSIVE,
            ),
            "Explosive Toad": Card(
                name="Explosive Toad",
                power=5,
                keywords={Keyword.FRENZY},
                ability_text="Defeated: Defeat a creature.",
                ability_trigger=TriggerType.DEFEATED,
            ),
            "Ferret Bomber": Card(
                name="Ferret Bomber",
                power=2,
                keywords={Keyword.SNEAKY},
                ability_text="Play: The opponent discards 2 cards.",
                ability_trigger=TriggerType.PLAY,
            ),
            "Giraffodile": Card(
                name="Giraffodile",
                power=7,
                keywords=set(),
                ability_text="Play: Draw your entire discard pile.",
                ability_trigger=TriggerType.PLAY,
            ),
            "Goblin Werewolf": Card(
                name="Goblin Werewolf",
                power=2,
                keywords={Keyword.HUNTER},
                ability_text="Has +6 power while it is your turn.",
                ability_trigger=TriggerType.PASSIVE,
            ),
            "Gorillion": Card(
                name="Gorillion",
                power=10,
                keywords=set(),
                ability_text="",  # Vanilla creature
                ability_trigger=None,
            ),
            "Grave Robber": Card(
                name="Grave Robber",
                power=7,
                keywords={Keyword.TOUGH},
                ability_text="Play: Play a card from the opponent's discard pile.",
                ability_trigger=TriggerType.PLAY,
            ),
            "Harpy Mother": Card(
                name="Harpy Mother",
                power=5,
                keywords=set(),
                ability_text="Defeated: Take control of up to 2 creatures with power 5 or less.",
                ability_trigger=TriggerType.DEFEATED,
            ),
            "Kangasaurus Rex": Card(
                name="Kangasaurus Rex",
                power=7,
                keywords=set(),
                ability_text="Play: Defeat all enemy creatures with power 4 or less.",
                ability_trigger=TriggerType.PLAY,
            ),
            "Killer Bee": Card(
                name="Killer Bee",
                power=5,
                keywords={Keyword.HUNTER},
                ability_text="Play: The opponent loses 1 life.",
                ability_trigger=TriggerType.PLAY,
            ),
            "Lone Yeti": Card(
                name="Lone Yeti",
                power=5,
                keywords={Keyword.TOUGH},
                ability_text="While this is your only allied creature, it has +5 power and FRENZY.",
                ability_trigger=TriggerType.PASSIVE,
            ),
            "Luchataur": Card(
                name="Luchataur",
                power=9,
                keywords={Keyword.FRENZY},
                ability_text="",  # Vanilla with keyword
                ability_trigger=None,
            ),
            "Mysterious Mermaid": Card(
                name="Mysterious Mermaid",
                power=7,
                keywords=set(),
                ability_text="Play: Set your life equal to the opponent's.",
                ability_trigger=TriggerType.PLAY,
            ),
            "Plated Scorpion": Card(
                name="Plated Scorpion",
                power=2,
                keywords={Keyword.TOUGH, Keyword.POISONOUS},
                ability_text="",  # Vanilla with keywords
                ability_trigger=None,
            ),
            "Rhino Turtle": Card(
                name="Rhino Turtle",
                power=8,
                keywords={Keyword.FRENZY, Keyword.TOUGH},
                ability_text="",  # Vanilla with keywords
                ability_trigger=None,
            ),
            "Shark Dog": Card(
                name="Shark Dog",
                power=4,
                keywords={Keyword.HUNTER},
                ability_text="Attack: Defeat an enemy creature with power 6 or more.",
                ability_trigger=TriggerType.ATTACK,
            ),
            "Sharky Crab-Dog-Mummypus": Card(
                name="Sharky Crab-Dog-Mummypus",
                power=5,
                keywords=set(),
                ability_text="Has HUNTER while an enemy creature does. Repeat for SNEAKY, FRENZY, and POISONOUS.",
                ability_trigger=TriggerType.PASSIVE,
            ),
            "Shield Bugs": Card(
                name="Shield Bugs",
                power=4,
                keywords={Keyword.TOUGH},
                ability_text="Other allied creatures have +1 power.",
                ability_trigger=TriggerType.PASSIVE,
            ),
            "Snail Hydra": Card(
                name="Snail Hydra",
                power=9,
                keywords=set(),
                ability_text="Attack: If you control fewer creatures than the opponent, defeat a creature.",
                ability_trigger=TriggerType.ATTACK,
            ),
            "Snail Thrower": Card(
                name="Snail Thrower",
                power=1,
                keywords={Keyword.POISONOUS},
                ability_text="Other allied creatures with power 4 or less have HUNTER and POISONOUS.",
                ability_trigger=TriggerType.PASSIVE,
            ),
            "Spider Owl": Card(
                name="Spider Owl",
                power=3,
                keywords={Keyword.SNEAKY, Keyword.POISONOUS},
                ability_text="",  # Vanilla with keywords
                ability_trigger=None,
            ),
            "Strange Barrel": Card(
                name="Strange Barrel",
                power=6,
                keywords=set(),
                ability_text="Defeated: Steal 2 random cards from the opponent's hand.",
                ability_trigger=TriggerType.DEFEATED,
            ),
            "Tiger Squirrel": Card(
                name="Tiger Squirrel",
                power=3,
                keywords={Keyword.SNEAKY},
                ability_text="Play: Defeat an enemy creature with power 7 or more.",
                ability_trigger=TriggerType.PLAY,
            ),
            "Turbo Bug": Card(
                name="Turbo Bug",
                power=4,
                keywords=set(),
                ability_text="Attack: The opponent loses all life except 1.",
                ability_trigger=TriggerType.ATTACK,
            ),
            "Tusked Extorter": Card(
                name="Tusked Extorter",
                power=8,
                keywords=set(),
                ability_text="Attack: The opponent discards a card.",
                ability_trigger=TriggerType.ATTACK,
            ),
            "Urchin Hurler": Card(
                name="Urchin Hurler",
                power=5,
                keywords={Keyword.HUNTER},
                ability_text="Other allied creatures have +2 power while it is your turn.",
                ability_trigger=TriggerType.PASSIVE,
            ),
        }
        return cards

    @staticmethod
    def get_first_contact_deck() -> List[Card]:
        # Builds a complete 48-card deck with correct quantities
        cards = CardDefinitions.get_first_contact_cards()
        deck = []
        
        # Double quantity cards (2 copies each)
        for name in [
            "Axolotl Healer",
            "Compost Dragon",
            "Explosive Toad",
            "Ferret Bomber",
            "Goblin Werewolf",
            "Grave Robber",
            "Kangasaurus Rex",
            "Killer Bee",
            "Luchataur",
            "Plated Scorpion",
            "Rhino Turtle",
            "Shield Bugs",
            "Snail Hydra",
            "Spider Owl",
            "Tiger Squirrel",
            "Tusked Extorter",
        ]:
            deck.extend([cards[name]] * 2)
        
        # Single quantity cards (1 copy each)
        for name in [
            "Bee Bear",
            "Brain Fly",
            "Chameleon Sniper",
            "Deathweaver",
            "Elephantopus",
            "Giraffodile",
            "Gorillion",
            "Harpy Mother",
            "Lone Yeti",
            "Mysterious Mermaid",
            "Shark Dog",
            "Sharky Crab-Dog-Mummypus",
            "Snail Thrower",
            "Strange Barrel",
            "Turbo Bug",
            "Urchin Hurler",
        ]:
            deck.append(cards[name])
        
        # Verify deck size
        assert len(deck) == 48, f"Deck should have 48 cards, but has {len(deck)}"
        return deck