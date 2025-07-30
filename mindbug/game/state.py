import random
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .actions import Action
from .cards import Card
from .constants import (
    MAX_HAND_SIZE,
    MINDBUGS_PER_PLAYER,
    STARTING_HAND_SIZE,
    STARTING_LIFE,
    ActionType,
    Keyword,
    Player,
    Zone,
)


@dataclass
class CreatureState:
    card: Card
    controller: Player
    owner: Player
    is_exhausted: bool = False
    attack_count: int = 0
    play_area_index: Optional[int] = None

    def get_effective_power(
        self, is_controllers_turn: bool, allied_creatures: List["CreatureState"]
    ) -> int:
        power = self.card.power
        if self.card.name == "Goblin Werewolf" and is_controllers_turn:
            power += 6
        if self.card.name == "Lone Yeti" and len(allied_creatures) == 1:
            power += 5
        for creature in allied_creatures:
            if creature.card.name == "Shield Bugs" and creature != self:
                power += 1
        if is_controllers_turn:
            for creature in allied_creatures:
                if creature.card.name == "Urchin Hurler" and creature != self:
                    power += 2
        return max(1, power)

    def get_effective_keywords(
        self,
        allied_creatures: List["CreatureState"],
        enemy_creatures: List["CreatureState"],
    ) -> Set[Keyword]:
        keywords = self.card.keywords.copy()
        if self.card.name == "Lone Yeti" and len(allied_creatures) == 1:
            keywords.add(Keyword.FRENZY)
        if self.card.name == "Sharky Crab-Dog-Mummypus":
            enemy_keywords = set()
            for creature in enemy_creatures:
                enemy_keywords.update(creature.card.keywords)
            keywords.update(
                enemy_keywords
                & {Keyword.HUNTER, Keyword.SNEAKY, Keyword.FRENZY, Keyword.POISONOUS}
            )
        for creature in allied_creatures:
            if (
                creature.card.name == "Snail Thrower"
                and creature != self
                and self.card.power <= 4
            ):
                keywords.add(Keyword.HUNTER)
                keywords.add(Keyword.POISONOUS)
        return keywords


@dataclass
class GameState:
    current_player: Player
    life: Dict[Player, int] = field(
        default_factory=lambda: {
            Player.PLAYER_1: STARTING_LIFE,
            Player.PLAYER_2: STARTING_LIFE,
        }
    )
    mindbugs_available: Dict[Player, int] = field(
        default_factory=lambda: {
            Player.PLAYER_1: MINDBUGS_PER_PLAYER,
            Player.PLAYER_2: MINDBUGS_PER_PLAYER,
        }
    )
    hands: Dict[Player, List[Card]] = field(
        default_factory=lambda: {Player.PLAYER_1: [], Player.PLAYER_2: []}
    )
    decks: Dict[Player, List[Card]] = field(
        default_factory=lambda: {Player.PLAYER_1: [], Player.PLAYER_2: []}
    )
    play_areas: Dict[Player, List[CreatureState]] = field(
        default_factory=lambda: {Player.PLAYER_1: [], Player.PLAYER_2: []}
    )
    discard_piles: Dict[Player, List[Card]] = field(
        default_factory=lambda: {Player.PLAYER_1: [], Player.PLAYER_2: []}
    )
    extra_turn_pending: Optional[Player] = None
    mindbug_decision_pending: bool = False
    pending_creature_play: Optional[Tuple[Player, Card]] = None
    hunter_choice_pending: bool = False
    valid_blockers: List[int] = field(default_factory=list)
    attacking_creature: Optional[CreatureState] = None
    deathweaver_active: Dict[Player, bool] = field(
        default_factory=lambda: {Player.PLAYER_1: False, Player.PLAYER_2: False}
    )
    creatures_attacked_this_turn: Set[int] = field(default_factory=set)

    def copy(self) -> "GameState":
        return deepcopy(self)

    def is_terminal(self) -> bool:
        if any(life <= 0 for life in self.life.values()):
            return True
        legal_actions = self.get_legal_actions()
        if (
            not legal_actions
            and not self.mindbug_decision_pending
            and not self.hunter_choice_pending
        ):
            return True
        return False

    def get_winner(self) -> Optional[Player]:
        if not self.is_terminal():
            return None
        if self.life[Player.PLAYER_1] <= 0 and self.life[Player.PLAYER_2] <= 0:
            return None  # Draw
        elif self.life[Player.PLAYER_1] <= 0:
            return Player.PLAYER_2
        elif self.life[Player.PLAYER_2] <= 0:
            return Player.PLAYER_1
        return self.current_player.other()

    def get_legal_actions(self) -> List[Action]:
        actions = []
        if self.mindbug_decision_pending and self.pending_creature_play:
            opponent = self.current_player
            if self.mindbugs_available[opponent] > 0:
                actions.append(Action(ActionType.USE_MINDBUG, opponent))
            actions.append(Action(ActionType.PASS_MINDBUG, opponent))
            return actions
        if self.hunter_choice_pending and self.valid_blockers:
            for blocker_idx in self.valid_blockers:
                actions.append(
                    Action(
                        ActionType.CHOOSE_BLOCKER,
                        self.current_player,
                        target_index=blocker_idx,
                    )
                )
            return actions
        for card in self.hands[self.current_player]:
            actions.append(
                Action(ActionType.PLAY_CREATURE, self.current_player, card=card)
            )
        for i, creature_state in enumerate(self.play_areas[self.current_player]):
            allied_creatures = self.play_areas[self.current_player]
            enemy_creatures = self.play_areas[self.current_player.other()]
            keywords = creature_state.get_effective_keywords(
                allied_creatures, enemy_creatures
            )
            creature_id = id(creature_state)
            can_attack = False
            if creature_id not in self.creatures_attacked_this_turn:
                can_attack = True
            elif Keyword.FRENZY in keywords and creature_state.attack_count == 1:
                can_attack = True
            if can_attack:
                actions.append(
                    Action(ActionType.ATTACK, self.current_player, creature_index=i)
                )
        return actions

    def draw_cards(self, player: Player, count: int):
        cards_drawn = 0
        for _ in range(count):
            if not self.decks[player]:
                break
            if len(self.hands[player]) < MAX_HAND_SIZE:
                self.hands[player].append(self.decks[player].pop())
                cards_drawn += 1
        return cards_drawn

    def discard_random(self, player: Player, count: int):
        count = min(count, len(self.hands[player]))
        if count > 0:
            discarded = random.sample(self.hands[player], count)
            for card in discarded:
                self.hands[player].remove(card)
                self.discard_piles[player].append(card)

    def update_creature_indices(self):
        for player in [Player.PLAYER_1, Player.PLAYER_2]:
            for i, creature in enumerate(self.play_areas[player]):
                creature.play_area_index = i
