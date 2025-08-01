"""Main game engine implementing all Mindbug rules."""

import random
from typing import List, Optional, Tuple

from .actions import Action
from .cards import Card, CardDatabase
from .constants import (
    DECK_SIZE_PER_PLAYER,
    STARTING_HAND_SIZE,
    ActionType,
    Keyword,
    Player,
    TriggerType,
)
from .creature import CreatureState
from .state import GameState


class MindbugEngine:
    """Core game engine handling all rules and mechanics."""

    @staticmethod
    def create_initial_state(
        deck: Optional[List[Card]] = None,
        starting_player: Optional[Player] = None,
        shuffle: bool = True,
    ) -> GameState:
        """Create a new game with shuffled decks and starting hands."""
        if deck is None:
            deck = CardDatabase.get_first_contact_deck()

        deck = deck.copy()
        if shuffle:
            random.shuffle(deck)

        # Random starting player if not specified
        if starting_player is None:
            starting_player = random.choice([Player.PLAYER_1, Player.PLAYER_2])

        state = GameState(current_player=starting_player)

        # Deal cards to each player
        for player in [Player.PLAYER_1, Player.PLAYER_2]:
            player_cards = deck[:DECK_SIZE_PER_PLAYER]
            deck = deck[DECK_SIZE_PER_PLAYER:]

            # First 5 go to hand, rest to deck
            state.hands[player] = player_cards[:STARTING_HAND_SIZE]
            state.decks[player] = player_cards[STARTING_HAND_SIZE:]

        return state

    @staticmethod
    def apply_action(state: GameState, action: Action) -> GameState:
        """Apply an action and return new state."""
        new_state = state.copy()

        if action.action_type == ActionType.PLAY_CREATURE:
            MindbugEngine._handle_play_creature(new_state, action)
        elif action.action_type == ActionType.ATTACK:
            MindbugEngine._handle_attack(new_state, action)
        elif action.action_type == ActionType.USE_MINDBUG:
            MindbugEngine._handle_use_mindbug(new_state, action)
        elif action.action_type == ActionType.PASS_MINDBUG:
            MindbugEngine._handle_pass_mindbug(new_state, action)
        elif action.action_type == ActionType.CHOOSE_BLOCKER:
            MindbugEngine._handle_choose_blocker(new_state, action)

        return new_state

    @staticmethod
    def _handle_play_creature(state: GameState, action: Action) -> None:
        """Handle playing a creature from hand."""
        player = action.player
        card = action.card

        # Remove from hand
        state.hands[player].remove(card)

        # Draw to refill hand to 5
        cards_to_draw = STARTING_HAND_SIZE - len(state.hands[player])
        state.draw_cards(player, cards_to_draw)

        # Enter Mindbug decision phase
        state.mindbug_decision_pending = True
        state.pending_creature_play = (player, card)
        state.current_player = player.other()  # Opponent decides

    @staticmethod
    def _handle_use_mindbug(state: GameState, action: Action) -> None:
        """Handle opponent stealing creature with Mindbug."""
        opponent = action.player
        player, card = state.pending_creature_play

        # Spend Mindbug
        state.mindbugs_available[opponent] -= 1

        # Create creature under opponent's control
        creature = CreatureState(
            card=card, controller=opponent, owner=player  # Original owner for discard
        )
        state.play_areas[opponent].append(creature)

        # Update Deathweaver status
        state.update_deathweaver_status()

        # Resolve play ability if not blocked
        if not state.deathweaver_active[opponent]:
            MindbugEngine._resolve_play_ability(state, card, opponent)

        # Clear Mindbug phase
        state.mindbug_decision_pending = False
        state.pending_creature_play = None

        # Original player gets extra turn
        state.extra_turn_pending = player
        state.current_player = player

    @staticmethod
    def _handle_pass_mindbug(state: GameState, action: Action) -> None:
        """Handle opponent passing on Mindbug."""
        player, card = state.pending_creature_play

        # Create creature under player's control
        creature = CreatureState(card=card, controller=player, owner=player)
        state.play_areas[player].append(creature)

        # Update Deathweaver status
        state.update_deathweaver_status()

        # Resolve play ability if not blocked
        if not state.deathweaver_active[player]:
            MindbugEngine._resolve_play_ability(state, card, player)

        # Clear Mindbug phase
        state.mindbug_decision_pending = False
        state.pending_creature_play = None

        # Normal turn progression
        MindbugEngine._end_turn(state, player)

    @staticmethod
    def _handle_attack(state: GameState, action: Action) -> None:
        """Handle creature attack."""
        attacker_idx = action.creature_index
        attacker = state.play_areas[action.player][attacker_idx]
        creature_id = id(attacker)

        # Track attack
        state.creatures_attacked_this_turn.add(creature_id)
        attacker.attack_count += 1

        # Resolve attack ability
        MindbugEngine._resolve_attack_ability(state, attacker.card, action.player)

        # Check if attacker still exists
        if attacker not in state.play_areas[action.player]:
            MindbugEngine._end_turn(state, action.player)
            return

        # Get valid blockers
        blockers = MindbugEngine._get_valid_blockers(state, attacker, action.player)

        if not blockers:
            # No blockers - deal 1 damage
            state.life[action.player.other()] -= 1
            MindbugEngine._check_and_continue_attack(state, action.player, attacker)
        else:
            # Check for HUNTER
            allied = state.play_areas[action.player]
            enemy = state.play_areas[action.player.other()]
            keywords = attacker.get_effective_keywords(allied, enemy)

            if Keyword.HUNTER in keywords:
                # Attacker chooses blocker
                state.hunter_choice_pending = True
                state.valid_blockers = blockers
                state.attacking_creature = attacker
            else:
                # Defender chooses blocker (AI for now)
                blocker_idx = MindbugEngine._choose_blocker_ai(
                    state, attacker, blockers, action.player
                )
                MindbugEngine._resolve_combat(state, attacker, action.player, blocker_idx)
                MindbugEngine._check_and_continue_attack(state, action.player, attacker)

    @staticmethod
    def _handle_choose_blocker(state: GameState, action: Action) -> None:
        """Handle HUNTER blocker choice."""
        blocker_idx = action.target_index
        attacker = state.attacking_creature
        attacker_player = state.current_player

        # Clear HUNTER state
        state.hunter_choice_pending = False
        state.valid_blockers = []
        state.attacking_creature = None

        # Resolve combat
        MindbugEngine._resolve_combat(state, attacker, attacker_player, blocker_idx)
        MindbugEngine._check_and_continue_attack(state, attacker_player, attacker)

    @staticmethod
    def _check_and_continue_attack(
        state: GameState, attacker_player: Player, attacker: CreatureState
    ) -> None:
        """Check if attack phase should continue (FRENZY) or end."""
        if attacker not in state.play_areas[attacker_player]:
            MindbugEngine._end_turn(state, attacker_player)
            return

        # Check for FRENZY second attack
        allied = state.play_areas[attacker_player]
        enemy = state.play_areas[attacker_player.other()]
        keywords = attacker.get_effective_keywords(allied, enemy)

        if attacker.attack_count == 1 and Keyword.FRENZY in keywords:
            return  # Can attack again

        MindbugEngine._end_turn(state, attacker_player)

    @staticmethod
    def _end_turn(state: GameState, current_player: Player) -> None:
        """End turn and handle extra turn logic."""
        state.creatures_attacked_this_turn.clear()
        state.reset_attack_counts()

        if state.extra_turn_pending == current_player:
            state.extra_turn_pending = None
            # Keep current player's turn
        else:
            state.current_player = current_player.other()

    @staticmethod
    def _get_valid_blockers(
        state: GameState, attacker: CreatureState, attacker_player: Player
    ) -> List[int]:
        """Get indices of creatures that can block the attacker."""
        defender_player = attacker_player.other()
        blockers = []

        attacker_allied = state.play_areas[attacker_player]
        attacker_enemy = state.play_areas[defender_player]
        attacker_keywords = attacker.get_effective_keywords(attacker_allied, attacker_enemy)

        for i, creature in enumerate(attacker_enemy):
            creature_keywords = creature.get_effective_keywords(attacker_enemy, attacker_allied)

            # SNEAKY can only be blocked by SNEAKY
            if Keyword.SNEAKY in attacker_keywords and Keyword.SNEAKY not in creature_keywords:
                continue

            # Bee Bear: Can't be blocked by power 6 or less
            if attacker.card.name == "Bee Bear":
                power = creature.get_effective_power(
                    state.current_player == defender_player, attacker_enemy
                )
                if power <= 6:
                    continue

            # Elephantopus restriction (unless HUNTER)
            elephantopus_present = any(c.card.name == "Elephantopus" for c in attacker_allied)
            if elephantopus_present and Keyword.HUNTER not in attacker_keywords:
                power = creature.get_effective_power(
                    state.current_player == defender_player, attacker_enemy
                )
                if power <= 4:
                    continue

            blockers.append(i)

        return blockers

    @staticmethod
    def _choose_blocker_ai(
        state: GameState,
        attacker: CreatureState,
        blocker_indices: List[int],
        attacker_player: Player,
    ) -> int:
        """AI logic for choosing best blocker."""
        defender_player = attacker_player.other()
        defender_creatures = state.play_areas[defender_player]

        attacker_allied = state.play_areas[attacker_player]
        attacker_enemy = defender_creatures

        attacker_power = attacker.get_effective_power(
            state.current_player == attacker_player, attacker_allied
        )
        attacker_keywords = attacker.get_effective_keywords(attacker_allied, attacker_enemy)

        # Evaluate each blocker
        best_score = -1000
        best_idx = blocker_indices[0]

        for idx in blocker_indices:
            blocker = defender_creatures[idx]
            blocker_power = blocker.get_effective_power(
                state.current_player == defender_player, defender_creatures
            )
            blocker_keywords = blocker.get_effective_keywords(defender_creatures, attacker_allied)

            # Calculate outcomes
            attacker_dies = Keyword.POISONOUS in blocker_keywords or (
                blocker_power >= attacker_power and Keyword.POISONOUS not in attacker_keywords
            )

            blocker_dies = Keyword.POISONOUS in attacker_keywords or (
                attacker_power >= blocker_power and Keyword.POISONOUS not in blocker_keywords
            )

            # TOUGH adjustments
            if blocker_dies and Keyword.TOUGH in blocker_keywords and not blocker.is_exhausted:
                blocker_dies = False

            # Score the outcome
            score = 0
            if attacker_dies:
                score += 10 + attacker_power
            if not blocker_dies:
                score += 5 + blocker_power
            if blocker_dies:
                score -= blocker_power

            if score > best_score:
                best_score = score
                best_idx = idx

        return best_idx

    @staticmethod
    def _resolve_combat(
        state: GameState, attacker: CreatureState, attacker_player: Player, blocker_idx: int
    ) -> None:
        """Resolve combat between two creatures."""
        defender_player = attacker_player.other()
        attacker_creatures = state.play_areas[attacker_player]
        blocker_creatures = state.play_areas[defender_player]

        if attacker not in attacker_creatures or blocker_idx >= len(blocker_creatures):
            return

        blocker = blocker_creatures[blocker_idx]

        # Get current stats
        attacker_power = attacker.get_effective_power(
            state.current_player == attacker_player, attacker_creatures
        )
        blocker_power = blocker.get_effective_power(
            state.current_player == defender_player, blocker_creatures
        )

        attacker_keywords = attacker.get_effective_keywords(attacker_creatures, blocker_creatures)
        blocker_keywords = blocker.get_effective_keywords(blocker_creatures, attacker_creatures)

        # Determine defeats
        attacker_defeated = False
        blocker_defeated = False

        # POISONOUS always defeats
        if Keyword.POISONOUS in attacker_keywords:
            blocker_defeated = True
        if Keyword.POISONOUS in blocker_keywords:
            attacker_defeated = True

        # Power comparison (if not already defeated by POISONOUS)
        if not blocker_defeated and Keyword.POISONOUS not in attacker_keywords:
            if attacker_power >= blocker_power:
                blocker_defeated = True

        if not attacker_defeated and Keyword.POISONOUS not in blocker_keywords:
            if blocker_power >= attacker_power:
                attacker_defeated = True

        # TOUGH prevents first defeat
        if attacker_defeated and Keyword.TOUGH in attacker_keywords and not attacker.is_exhausted:
            attacker.is_exhausted = True
            attacker_defeated = False

        if blocker_defeated and Keyword.TOUGH in blocker_keywords and not blocker.is_exhausted:
            blocker.is_exhausted = True
            blocker_defeated = False

        # Apply defeats in active player order
        if state.current_player == attacker_player:
            if attacker_defeated:
                MindbugEngine._defeat_creature(state, attacker, attacker_player)
            if blocker_defeated and blocker in state.play_areas[defender_player]:
                MindbugEngine._defeat_creature(state, blocker, defender_player)
        else:
            if blocker_defeated:
                MindbugEngine._defeat_creature(state, blocker, defender_player)
            if attacker_defeated and attacker in state.play_areas[attacker_player]:
                MindbugEngine._defeat_creature(state, attacker, attacker_player)

    @staticmethod
    def _defeat_creature(state: GameState, creature: CreatureState, controller: Player) -> None:
        """Remove creature from play and trigger defeated abilities."""
        if creature not in state.play_areas[controller]:
            return

        # Remove from play
        state.play_areas[controller].remove(creature)
        creature_id = id(creature)
        state.creatures_attacked_this_turn.discard(creature_id)

        # Add to owner's discard (not controller's)
        state.discard_piles[creature.owner].append(creature.card)

        # Trigger defeated ability
        MindbugEngine._resolve_defeated_ability(state, creature.card, controller)

    @staticmethod
    def _resolve_play_ability(state: GameState, card: Card, controller: Player) -> None:
        """Resolve creature's play ability."""
        if card.ability_trigger != TriggerType.PLAY:
            return

        opponent = controller.other()

        if card.name == "Axolotl Healer":
            state.life[controller] += 2

        elif card.name == "Brain Fly":
            # Take control of power 6+ creature
            targets = []
            for i, c in enumerate(state.play_areas[opponent]):
                power = c.get_effective_power(
                    state.current_player == opponent, state.play_areas[opponent]
                )
                if power >= 6:
                    targets.append((i, c))

            if targets:
                _, creature = random.choice(targets)
                state.play_areas[opponent].remove(creature)
                creature.controller = controller
                state.play_areas[controller].append(creature)

        elif card.name == "Compost Dragon":
            # Play random card from discard
            if state.discard_piles[controller]:
                card_to_play = random.choice(state.discard_piles[controller])
                state.discard_piles[controller].remove(card_to_play)
                creature = CreatureState(card=card_to_play, controller=controller, owner=controller)
                state.play_areas[controller].append(creature)
                # Note: This doesn't trigger play abilities

        elif card.name == "Ferret Bomber":
            state.discard_random(opponent, 2)

        elif card.name == "Giraffodile":
            # Draw entire discard pile
            state.hands[controller].extend(state.discard_piles[controller])
            state.discard_piles[controller] = []
            # Discard down to hand limit
            while len(state.hands[controller]) > STARTING_HAND_SIZE:
                card_to_discard = random.choice(state.hands[controller])
                state.hands[controller].remove(card_to_discard)
                state.discard_piles[controller].append(card_to_discard)

        elif card.name == "Grave Robber":
            # Play from opponent's discard
            if state.discard_piles[opponent]:
                card_to_play = random.choice(state.discard_piles[opponent])
                state.discard_piles[opponent].remove(card_to_play)
                creature = CreatureState(
                    card=card_to_play, controller=controller, owner=opponent  # Original owner
                )
                state.play_areas[controller].append(creature)

        elif card.name == "Kangasaurus Rex":
            # Defeat all power 4 or less
            to_defeat = []
            for c in state.play_areas[opponent]:
                power = c.get_effective_power(
                    state.current_player == opponent, state.play_areas[opponent]
                )
                if power <= 4:
                    to_defeat.append(c)

            # Defeat in reverse order to avoid index issues
            for creature in reversed(to_defeat):
                MindbugEngine._defeat_creature(state, creature, opponent)

        elif card.name == "Killer Bee":
            state.life[opponent] -= 1

        elif card.name == "Mysterious Mermaid":
            state.life[controller] = state.life[opponent]

        elif card.name == "Tiger Squirrel":
            # Defeat power 7+ creature
            targets = []
            for c in state.play_areas[opponent]:
                power = c.get_effective_power(
                    state.current_player == opponent, state.play_areas[opponent]
                )
                if power >= 7:
                    targets.append(c)

            if targets:
                creature = random.choice(targets)
                MindbugEngine._defeat_creature(state, creature, opponent)

    @staticmethod
    def _resolve_attack_ability(state: GameState, card: Card, controller: Player) -> None:
        """Resolve creature's attack ability."""
        if card.ability_trigger != TriggerType.ATTACK:
            return

        opponent = controller.other()

        if card.name == "Chameleon Sniper":
            state.life[opponent] -= 1

        elif card.name == "Shark Dog":
            # Defeat power 6+ creature
            targets = []
            for c in state.play_areas[opponent]:
                power = c.get_effective_power(
                    state.current_player == opponent, state.play_areas[opponent]
                )
                if power >= 6:
                    targets.append(c)

            if targets:
                creature = random.choice(targets)
                MindbugEngine._defeat_creature(state, creature, opponent)

        elif card.name == "Snail Hydra":
            # If fewer creatures, defeat one
            if len(state.play_areas[controller]) < len(state.play_areas[opponent]):
                if state.play_areas[opponent]:
                    creature = random.choice(state.play_areas[opponent])
                    MindbugEngine._defeat_creature(state, creature, opponent)

        elif card.name == "Turbo Bug":
            state.life[opponent] = 1

        elif card.name == "Tusked Extorter":
            state.discard_random(opponent, 1)

    @staticmethod
    def _resolve_defeated_ability(state: GameState, card: Card, controller: Player) -> None:
        """Resolve creature's defeated ability."""
        if card.ability_trigger != TriggerType.DEFEATED:
            return

        opponent = controller.other()

        if card.name == "Explosive Toad":
            # Defeat random creature
            all_creatures = [
                (p, c) for p in [Player.PLAYER_1, Player.PLAYER_2] for c in state.play_areas[p]
            ]
            if all_creatures:
                player, creature = random.choice(all_creatures)
                MindbugEngine._defeat_creature(state, creature, player)

        elif card.name == "Harpy Mother":
            # Take control of up to 2 power 5 or less creatures
            targets = []
            for c in state.play_areas[opponent]:
                power = c.get_effective_power(
                    state.current_player == opponent, state.play_areas[opponent]
                )
                if power <= 5:
                    targets.append(c)

            # Take up to 2 random targets
            random.shuffle(targets)
            for creature in targets[:2]:
                state.play_areas[opponent].remove(creature)
                creature.controller = controller
                state.play_areas[controller].append(creature)

        elif card.name == "Strange Barrel":
            # Steal 2 random cards from hand
            cards_stolen = []
            for _ in range(2):
                if state.hands[opponent]:
                    card = random.choice(state.hands[opponent])
                    state.hands[opponent].remove(card)
                    state.hands[controller].append(card)
                    cards_stolen.append(card)
