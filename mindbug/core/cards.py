"""Card definitions for Mindbug First Contact."""

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Set

from .constants import Keyword, TriggerType


@dataclass(frozen=True)
class Card:
    """Immutable card definition."""

    name: str
    power: int
    keywords: FrozenSet[Keyword] = field(default_factory=frozenset)
    ability_text: str = ""
    ability_trigger: TriggerType | None = None

    def __str__(self) -> str:
        keywords_str = ", ".join(k.value for k in self.keywords) if self.keywords else "None"
        return f"{self.name} (Power: {self.power}, Keywords: {keywords_str})"


class CardDatabase:
    """Manages all card definitions."""

    _cards: Dict[str, Card] = {}
    _initialized: bool = False

    @classmethod
    def initialize(cls) -> None:
        """Initialize the card database with all First Contact cards."""
        if cls._initialized:
            return

        # Define all 32 unique cards
        cls._cards = {
            # Vanilla and keyword-only creatures
            "Gorillion": Card(
                name="Gorillion",
                power=10,
                keywords=frozenset(),
                ability_text="",
            ),
            "Luchataur": Card(
                name="Luchataur",
                power=9,
                keywords=frozenset({Keyword.FRENZY}),
                ability_text="",
            ),
            "Rhino Turtle": Card(
                name="Rhino Turtle",
                power=8,
                keywords=frozenset({Keyword.FRENZY, Keyword.TOUGH}),
                ability_text="",
            ),
            "Plated Scorpion": Card(
                name="Plated Scorpion",
                power=2,
                keywords=frozenset({Keyword.TOUGH, Keyword.POISONOUS}),
                ability_text="",
            ),
            "Spider Owl": Card(
                name="Spider Owl",
                power=3,
                keywords=frozenset({Keyword.SNEAKY, Keyword.POISONOUS}),
                ability_text="",
            ),
            # Play ability creatures
            "Axolotl Healer": Card(
                name="Axolotl Healer",
                power=4,
                keywords=frozenset({Keyword.POISONOUS}),
                ability_text="Play: Gain 2 life.",
                ability_trigger=TriggerType.PLAY,
            ),
            "Brain Fly": Card(
                name="Brain Fly",
                power=4,
                keywords=frozenset(),
                ability_text="Play: Take control of a creature with power 6 or more.",
                ability_trigger=TriggerType.PLAY,
            ),
            "Compost Dragon": Card(
                name="Compost Dragon",
                power=3,
                keywords=frozenset({Keyword.HUNTER}),
                ability_text="Play: Play a card from your discard pile.",
                ability_trigger=TriggerType.PLAY,
            ),
            "Ferret Bomber": Card(
                name="Ferret Bomber",
                power=2,
                keywords=frozenset({Keyword.SNEAKY}),
                ability_text="Play: The opponent discards 2 cards.",
                ability_trigger=TriggerType.PLAY,
            ),
            "Giraffodile": Card(
                name="Giraffodile",
                power=7,
                keywords=frozenset(),
                ability_text="Play: Draw your entire discard pile.",
                ability_trigger=TriggerType.PLAY,
            ),
            "Grave Robber": Card(
                name="Grave Robber",
                power=7,
                keywords=frozenset({Keyword.TOUGH}),
                ability_text="Play: Play a card from the opponent's discard pile.",
                ability_trigger=TriggerType.PLAY,
            ),
            "Kangasaurus Rex": Card(
                name="Kangasaurus Rex",
                power=7,
                keywords=frozenset(),
                ability_text="Play: Defeat all enemy creatures with power 4 or less.",
                ability_trigger=TriggerType.PLAY,
            ),
            "Killer Bee": Card(
                name="Killer Bee",
                power=5,
                keywords=frozenset({Keyword.HUNTER}),
                ability_text="Play: The opponent loses 1 life.",
                ability_trigger=TriggerType.PLAY,
            ),
            "Mysterious Mermaid": Card(
                name="Mysterious Mermaid",
                power=7,
                keywords=frozenset(),
                ability_text="Play: Set your life equal to the opponent's.",
                ability_trigger=TriggerType.PLAY,
            ),
            "Tiger Squirrel": Card(
                name="Tiger Squirrel",
                power=3,
                keywords=frozenset({Keyword.SNEAKY}),
                ability_text="Play: Defeat an enemy creature with power 7 or more.",
                ability_trigger=TriggerType.PLAY,
            ),
            # Attack ability creatures
            "Chameleon Sniper": Card(
                name="Chameleon Sniper",
                power=1,
                keywords=frozenset({Keyword.SNEAKY}),
                ability_text="Attack: The opponent loses 1 life.",
                ability_trigger=TriggerType.ATTACK,
            ),
            "Shark Dog": Card(
                name="Shark Dog",
                power=4,
                keywords=frozenset({Keyword.HUNTER}),
                ability_text="Attack: Defeat an enemy creature with power 6 or more.",
                ability_trigger=TriggerType.ATTACK,
            ),
            "Snail Hydra": Card(
                name="Snail Hydra",
                power=9,
                keywords=frozenset(),
                ability_text="Attack: If you control fewer creatures than the opponent, defeat a creature.",
                ability_trigger=TriggerType.ATTACK,
            ),
            "Turbo Bug": Card(
                name="Turbo Bug",
                power=4,
                keywords=frozenset(),
                ability_text="Attack: The opponent loses all life except 1.",
                ability_trigger=TriggerType.ATTACK,
            ),
            "Tusked Extorter": Card(
                name="Tusked Extorter",
                power=8,
                keywords=frozenset(),
                ability_text="Attack: The opponent discards a card.",
                ability_trigger=TriggerType.ATTACK,
            ),
            # Defeated ability creatures
            "Explosive Toad": Card(
                name="Explosive Toad",
                power=5,
                keywords=frozenset({Keyword.FRENZY}),
                ability_text="Defeated: Defeat a creature.",
                ability_trigger=TriggerType.DEFEATED,
            ),
            "Harpy Mother": Card(
                name="Harpy Mother",
                power=5,
                keywords=frozenset(),
                ability_text="Defeated: Take control of up to 2 creatures with power 5 or less.",
                ability_trigger=TriggerType.DEFEATED,
            ),
            "Strange Barrel": Card(
                name="Strange Barrel",
                power=6,
                keywords=frozenset(),
                ability_text="Defeated: Steal 2 random cards from the opponent's hand.",
                ability_trigger=TriggerType.DEFEATED,
            ),
            # Passive ability creatures
            "Bee Bear": Card(
                name="Bee Bear",
                power=8,
                keywords=frozenset(),
                ability_text="Cannot be blocked by creatures with power 6 or less.",
                ability_trigger=TriggerType.PASSIVE,
            ),
            "Deathweaver": Card(
                name="Deathweaver",
                power=2,
                keywords=frozenset({Keyword.POISONOUS}),
                ability_text="The opponent cannot activate Play effects.",
                ability_trigger=TriggerType.PASSIVE,
            ),
            "Elephantopus": Card(
                name="Elephantopus",
                power=7,
                keywords=frozenset({Keyword.TOUGH}),
                ability_text="The opponent cannot block with creatures with power 4 or less.",
                ability_trigger=TriggerType.PASSIVE,
            ),
            "Goblin Werewolf": Card(
                name="Goblin Werewolf",
                power=2,
                keywords=frozenset({Keyword.HUNTER}),
                ability_text="Has +6 power while it is your turn.",
                ability_trigger=TriggerType.PASSIVE,
            ),
            "Lone Yeti": Card(
                name="Lone Yeti",
                power=5,
                keywords=frozenset({Keyword.TOUGH}),
                ability_text="While this is your only allied creature, it has +5 power and FRENZY.",
                ability_trigger=TriggerType.PASSIVE,
            ),
            "Sharky Crab-Dog-Mummypus": Card(
                name="Sharky Crab-Dog-Mummypus",
                power=5,
                keywords=frozenset(),
                ability_text="Has HUNTER while an enemy creature does. Repeat for SNEAKY, FRENZY, and POISONOUS.",
                ability_trigger=TriggerType.PASSIVE,
            ),
            "Shield Bugs": Card(
                name="Shield Bugs",
                power=4,
                keywords=frozenset({Keyword.TOUGH}),
                ability_text="Other allied creatures have +1 power.",
                ability_trigger=TriggerType.PASSIVE,
            ),
            "Snail Thrower": Card(
                name="Snail Thrower",
                power=1,
                keywords=frozenset({Keyword.POISONOUS}),
                ability_text="Other allied creatures with power 4 or less have HUNTER and POISONOUS.",
                ability_trigger=TriggerType.PASSIVE,
            ),
            "Urchin Hurler": Card(
                name="Urchin Hurler",
                power=5,
                keywords=frozenset({Keyword.HUNTER}),
                ability_text="Other allied creatures have +2 power while it is your turn.",
                ability_trigger=TriggerType.PASSIVE,
            ),
        }

        cls._initialized = True

    @classmethod
    def get_card(cls, name: str) -> Card:
        """Get a card by name."""
        cls.initialize()
        if name not in cls._cards:
            raise ValueError(f"Unknown card: {name}")
        return cls._cards[name]

    @classmethod
    def get_all_cards(cls) -> Dict[str, Card]:
        """Get all card definitions."""
        cls.initialize()
        return cls._cards.copy()

    @classmethod
    def get_first_contact_deck(cls) -> List[Card]:
        """Get the complete 48-card First Contact deck."""
        cls.initialize()

        deck = []

        # Cards with 2 copies each (16 cards = 32 total)
        double_cards = [
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
        ]

        for name in double_cards:
            deck.extend([cls._cards[name]] * 2)

        # Cards with 1 copy each (16 cards)
        single_cards = [
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
        ]

        for name in single_cards:
            deck.append(cls._cards[name])

        assert len(deck) == 48, f"Deck should have 48 cards but has {len(deck)}"
        return deck
