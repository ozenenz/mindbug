"""Creature state representation."""

from dataclasses import dataclass
from typing import List, Set

from .cards import Card
from .constants import Keyword, Player


@dataclass
class CreatureState:
    """Represents a creature in play with its current state."""

    card: Card
    controller: Player  # Who currently controls it
    owner: Player  # Who originally played it (for discard)
    is_exhausted: bool = False  # TOUGH keyword exhaustion
    attack_count: int = 0  # Tracks attacks this turn for FRENZY

    def get_effective_power(
        self, is_controllers_turn: bool, allied_creatures: List["CreatureState"]
    ) -> int:
        """Calculate power including all modifiers."""
        power = self.card.power

        # Goblin Werewolf: +6 on controller's turn
        if self.card.name == "Goblin Werewolf" and is_controllers_turn:
            power += 6

        # Lone Yeti: +5 when alone
        if self.card.name == "Lone Yeti" and len(allied_creatures) == 1:
            power += 5

        # Shield Bugs buff other creatures
        for creature in allied_creatures:
            if creature.card.name == "Shield Bugs" and creature != self:
                power += 1

        # Urchin Hurler buff on controller's turn
        if is_controllers_turn:
            for creature in allied_creatures:
                if creature.card.name == "Urchin Hurler" and creature != self:
                    power += 2

        return max(1, power)  # Minimum 1 power

    def get_effective_keywords(
        self,
        allied_creatures: List["CreatureState"],
        enemy_creatures: List["CreatureState"],
    ) -> Set[Keyword]:
        """Get all keywords including dynamic ones."""
        keywords = set(self.card.keywords)  # Convert frozenset to set</
        # Lone Yeti gains FRENZY when alone
        if self.card.name == "Lone Yeti" and len(allied_creatures) == 1:
            keywords.add(Keyword.FRENZY)

        # Sharky Crab-Dog-Mummypus copies enemy keywords
        if self.card.name == "Sharky Crab-Dog-Mummypus":
            enemy_keywords = set()
            for creature in enemy_creatures:
                enemy_effective = set(creature.card.keywords)

                # Check for Lone Yeti FRENZY
                if creature.card.name == "Lone Yeti" and len(enemy_creatures) == 1:
                    enemy_effective.add(Keyword.FRENZY)

                # Check for Snail Thrower effects
                for ally in enemy_creatures:
                    if (
                        ally.card.name == "Snail Thrower"
                        and ally != creature
                        and creature.card.power <= 4
                    ):
                        enemy_effective.add(Keyword.HUNTER)
                        enemy_effective.add(Keyword.POISONOUS)

                enemy_keywords.update(enemy_effective)

            # Copy relevant keywords
            copyable = {Keyword.HUNTER, Keyword.SNEAKY, Keyword.FRENZY, Keyword.POISONOUS}
            keywords.update(enemy_keywords & copyable)

        # Snail Thrower grants keywords to small allies
        for creature in allied_creatures:
            if creature.card.name == "Snail Thrower" and creature != self and self.card.power <= 4:
                keywords.add(Keyword.HUNTER)
                keywords.add(Keyword.POISONOUS)

        return keywords

    def __str__(self) -> str:
        """String representation for debugging."""
        exhausted = " [Exhausted]" if self.is_exhausted else ""
        return f"{self.card.name} (Power: {self.card.power}, Controller: {self.controller.name}){exhausted}"
