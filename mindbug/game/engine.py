import random
from copy import deepcopy
from typing import List, Optional, Set, Tuple

from .actions import Action
from .cards import Card, CardDefinitions
from .constants import (
    MINDBUGS_PER_PLAYER,
    STARTING_HAND_SIZE,
    STARTING_LIFE,
    ActionType,
    Keyword,
    Player,
    TriggerType,
)
from .state import CreatureState, GameState


class MindbugEngine:
    # Main game engine implementing all Mindbug First Contact rules
    
    @staticmethod
    def create_initial_state(
        deck: Optional[List[Card]] = None, starting_player: Optional[Player] = None
    ) -> GameState:
        # Initialize a new game with shuffled decks and starting hands
        if deck is None:
            deck = CardDefinitions.get_first_contact_deck()
        deck = deck.copy()
        random.shuffle(deck)
        
        state = GameState(
            current_player=starting_player
            or random.choice([Player.PLAYER_1, Player.PLAYER_2])
        )
        
        # Deal 10 cards to each player
        for player in [Player.PLAYER_1, Player.PLAYER_2]:
            state.decks[player] = deck[:10]
            deck = deck[10:]
            # Draw starting hand of 5 cards
            state.hands[player] = state.decks[player][:STARTING_HAND_SIZE]
            state.decks[player] = state.decks[player][STARTING_HAND_SIZE:]
        
        return state

    @staticmethod
    def apply_action(state: GameState, action: Action) -> GameState:
        # Apply an action and return new state
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
    def _handle_play_creature(state: GameState, action: Action):
        # Play creature from hand, triggering Mindbug decision
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
        state.current_player = player.other()  # Opponent decides on Mindbug

    @staticmethod
    def _handle_use_mindbug(state: GameState, action: Action):
        # Opponent steals creature with Mindbug
        opponent = action.player  # Player using Mindbug
        player, card = state.pending_creature_play
        
        # Spend Mindbug
        state.mindbugs_available[opponent] -= 1
        
        # Create creature under opponent's control but owned by original player
        creature_state = CreatureState(card=card, controller=opponent, owner=player)
        state.play_areas[opponent].append(creature_state)
        state.update_creature_indices()
        
        # Update Deathweaver status before resolving abilities
        MindbugEngine._update_deathweaver_status(state)
        
        # Resolve play ability if controller isn't blocked by Deathweaver
        if not state.deathweaver_active[opponent]:
            MindbugEngine._resolve_play_ability(state, card, opponent)
        
        # Clear Mindbug phase
        state.mindbug_decision_pending = False
        state.pending_creature_play = None
        
        # Original player gets extra turn after being Mindbugged
        state.extra_turn_pending = player
        state.current_player = player

    @staticmethod
    def _handle_pass_mindbug(state: GameState, action: Action):
        # Opponent passes on using Mindbug
        player, card = state.pending_creature_play
        
        # Create creature under player's control
        creature_state = CreatureState(card=card, controller=player, owner=player)
        state.play_areas[player].append(creature_state)
        state.update_creature_indices()
        
        # Update Deathweaver status
        MindbugEngine._update_deathweaver_status(state)
        
        # Resolve play ability if not blocked
        if not state.deathweaver_active[player]:
            MindbugEngine._resolve_play_ability(state, card, player)
        
        # Clear Mindbug phase
        state.mindbug_decision_pending = False
        state.pending_creature_play = None
        
        # Normal turn progression
        if state.extra_turn_pending == player:
            state.extra_turn_pending = None
        else:
            state.current_player = player.other()

    @staticmethod
    def _handle_attack(state: GameState, action: Action):
        # Handle creature attack
        attacker_idx = action.creature_index
        attacker = state.play_areas[action.player][attacker_idx]
        creature_id = id(attacker)
        
        # Track attack for FRENZY
        state.creatures_attacked_this_turn.add(creature_id)
        attacker.attack_count += 1
        
        # Get current board state
        attacker_creatures = state.play_areas[action.player]
        defender_creatures = state.play_areas[action.player.other()]
        
        # Trigger attack abilities first
        MindbugEngine._resolve_attack_ability(state, attacker.card, action.player)
        
        # Check if attacker still exists
        if attacker not in state.play_areas[action.player]:
            MindbugEngine._end_turn_if_needed(state, action.player)
            return
        
        # Get valid blockers
        defender_player = action.player.other()
        blockers = MindbugEngine._get_valid_blockers(
            state, attacker, attacker_creatures, defender_player, defender_creatures
        )
        
        if not blockers:
            # No blockers - deal 1 damage
            state.life[defender_player] -= 1
            MindbugEngine._check_and_end_attack(state, action.player, attacker)
        else:
            # Handle blocking
            attacker_keywords = attacker.get_effective_keywords(
                attacker_creatures, defender_creatures
            )
            
            if Keyword.HUNTER in attacker_keywords:
                # HUNTER: Attacker chooses blocker
                state.hunter_choice_pending = True
                state.valid_blockers = blockers
                state.attacking_creature = attacker
            else:
                # Defender chooses blocker (AI choice for now)
                blocker_idx = MindbugEngine._choose_blocker_ai(
                    state, attacker, attacker_creatures, blockers, defender_creatures
                )
                MindbugEngine._resolve_combat(
                    state, attacker, action.player, blocker_idx, defender_player
                )
                MindbugEngine._check_and_end_attack(state, action.player, attacker)

    @staticmethod
    def _handle_choose_blocker(state: GameState, action: Action):
        # HUNTER player chooses which creature blocks
        blocker_idx = action.target_index
        attacker = state.attacking_creature
        attacker_player = state.current_player
        defender_player = attacker_player.other()
        
        # Clear HUNTER state
        state.hunter_choice_pending = False
        state.valid_blockers = []
        state.attacking_creature = None
        
        # Resolve combat
        MindbugEngine._resolve_combat(
            state, attacker, attacker_player, blocker_idx, defender_player
        )
        MindbugEngine._check_and_end_attack(state, attacker_player, attacker)

    @staticmethod
    def _check_and_end_attack(
        state: GameState, attacker_player: Player, attacker: CreatureState
    ):
        # Check if attack phase should end
        if attacker not in state.play_areas[attacker_player]:
            MindbugEngine._end_turn_if_needed(state, attacker_player)
            return
        
        # Check for FRENZY second attack
        attacker_creatures = state.play_areas[attacker_player]
        defender_creatures = state.play_areas[attacker_player.other()]
        keywords = attacker.get_effective_keywords(
            attacker_creatures, defender_creatures
        )
        
        # FRENZY allows exactly 2 attacks
        if attacker.attack_count == 1 and Keyword.FRENZY in keywords:
            return  # Can attack again
        
        # Attack complete
        MindbugEngine._end_turn_if_needed(state, attacker_player)

    @staticmethod
    def _end_turn_if_needed(state: GameState, current_player: Player):
        # End turn and handle extra turn logic
        state.creatures_attacked_this_turn.clear()
        
        # Reset attack counts
        for creature in state.play_areas[current_player]:
            creature.attack_count = 0
        
        # Handle extra turn
        if state.extra_turn_pending == current_player:
            state.extra_turn_pending = None
            # Keep current player's turn
        else:
            state.current_player = current_player.other()

    @staticmethod
    def _update_deathweaver_status(state: GameState):
        # Check if each player is affected by opponent's Deathweaver
        state.deathweaver_active[Player.PLAYER_1] = any(
            c.card.name == "Deathweaver" for c in state.play_areas[Player.PLAYER_2]
        )
        state.deathweaver_active[Player.PLAYER_2] = any(
            c.card.name == "Deathweaver" for c in state.play_areas[Player.PLAYER_1]
        )

    @staticmethod
    def _get_valid_blockers(
        state: GameState,
        attacker: CreatureState,
        attacker_creatures: List[CreatureState],
        defender: Player,
        defender_creatures: List[CreatureState],
    ) -> List[int]:
        # Get creatures that can block the attacker
        blockers = []
        attacker_keywords = attacker.get_effective_keywords(
            attacker_creatures, defender_creatures
        )
        
        for i, creature in enumerate(defender_creatures):
            creature_keywords = creature.get_effective_keywords(
                defender_creatures, attacker_creatures
            )
            
            # SNEAKY can only be blocked by SNEAKY
            if (
                Keyword.SNEAKY in attacker_keywords
                and Keyword.SNEAKY not in creature_keywords
            ):
                continue
            
            # Bee Bear ability: Can't be blocked by power 6 or less
            if attacker.card.name == "Bee Bear":
                power = creature.get_effective_power(
                    state.current_player == defender, defender_creatures
                )
                if power <= 6:
                    continue
            
            # Elephantopus restriction (unless attacker has HUNTER)
            elephantopus_present = any(
                c.card.name == "Elephantopus" for c in attacker_creatures
            )
            if elephantopus_present and Keyword.HUNTER not in attacker_keywords:
                power = creature.get_effective_power(
                    state.current_player == defender, defender_creatures
                )
                if power <= 4:
                    continue
            
            blockers.append(i)
        
        return blockers

    @staticmethod
    def _choose_blocker_ai(
        state: GameState,
        attacker: CreatureState,
        attacker_creatures: List[CreatureState],
        blocker_indices: List[int],
        defender_creatures: List[CreatureState],
    ) -> int:
        # AI logic for choosing best blocker
        defender = attacker.controller.other()
        attacker_power = attacker.get_effective_power(
            state.current_player == attacker.controller, attacker_creatures
        )
        attacker_keywords = attacker.get_effective_keywords(
            attacker_creatures, defender_creatures
        )
        
        # Priority 1: Block and survive
        for idx in blocker_indices:
            blocker = defender_creatures[idx]
            blocker_power = blocker.get_effective_power(
                state.current_player == defender, defender_creatures
            )
            blocker_keywords = blocker.get_effective_keywords(
                defender_creatures, attacker_creatures
            )
            
            # Check outcomes
            kills_attacker = (
                Keyword.POISONOUS in blocker_keywords or
                (blocker_power >= attacker_power and Keyword.POISONOUS not in attacker_keywords)
            )
            
            survives = False
            if Keyword.POISONOUS not in attacker_keywords:
                if blocker_power > attacker_power:
                    survives = True
                elif Keyword.TOUGH in blocker_keywords and not blocker.is_exhausted:
                    survives = True
            elif Keyword.TOUGH in blocker_keywords and not blocker.is_exhausted:
                survives = True
            
            if kills_attacker and survives:
                return idx
        
        # Priority 2: Kill attacker even if dying
        for idx in blocker_indices:
            blocker = defender_creatures[idx]
            blocker_keywords = blocker.get_effective_keywords(
                defender_creatures, attacker_creatures
            )
            blocker_power = blocker.get_effective_power(
                state.current_player == defender, defender_creatures
            )
            
            if (Keyword.POISONOUS in blocker_keywords or
                (blocker_power >= attacker_power and Keyword.POISONOUS not in attacker_keywords)):
                return idx
        
        # Priority 3: Block with weakest creature
        weakest_idx = blocker_indices[0]
        weakest_power = float('inf')
        
        for idx in blocker_indices:
            blocker = defender_creatures[idx]
            power = blocker.get_effective_power(
                state.current_player == defender, defender_creatures
            )
            if power < weakest_power:
                weakest_power = power
                weakest_idx = idx
        
        return weakest_idx

    @staticmethod
    def _resolve_combat(
        state: GameState,
        attacker: CreatureState,
        attacker_player: Player,
        blocker_idx: int,
        blocker_player: Player,
    ):
        # Resolve combat between two creatures
        attacker_creatures = state.play_areas[attacker_player]
        blocker_creatures = state.play_areas[blocker_player]
        
        # Verify creatures still exist
        if attacker not in attacker_creatures or blocker_idx >= len(blocker_creatures):
            return
        
        blocker = blocker_creatures[blocker_idx]
        
        # Get current stats
        attacker_power = attacker.get_effective_power(
            state.current_player == attacker_player, attacker_creatures
        )
        blocker_power = blocker.get_effective_power(
            state.current_player == blocker_player, blocker_creatures
        )
        
        attacker_keywords = attacker.get_effective_keywords(
            attacker_creatures, blocker_creatures
        )
        blocker_keywords = blocker.get_effective_keywords(
            blocker_creatures, attacker_creatures
        )
        
        # Determine who is defeated
        attacker_defeated = False
        blocker_defeated = False
        
        # POISONOUS always defeats
        if Keyword.POISONOUS in attacker_keywords:
            blocker_defeated = True
        if Keyword.POISONOUS in blocker_keywords:
            attacker_defeated = True
        
        # Power comparison
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
                attacker_idx = attacker_creatures.index(attacker)
                MindbugEngine._defeat_creature(state, attacker_player, attacker_idx)
            if blocker_defeated and blocker in state.play_areas[blocker_player]:
                blocker_idx = state.play_areas[blocker_player].index(blocker)
                MindbugEngine._defeat_creature(state, blocker_player, blocker_idx)
        else:
            if blocker_defeated:
                MindbugEngine._defeat_creature(state, blocker_player, blocker_idx)
            if attacker_defeated and attacker in state.play_areas[attacker_player]:
                attacker_idx = state.play_areas[attacker_player].index(attacker)
                MindbugEngine._defeat_creature(state, attacker_player, attacker_idx)

    @staticmethod
    def _defeat_creature(state: GameState, player: Player, creature_idx: int):
        # Remove creature and trigger defeated abilities
        if creature_idx >= len(state.play_areas[player]):
            return
        
        creature = state.play_areas[player].pop(creature_idx)
        creature_id = id(creature)
        
        # Clear attack tracking
        state.creatures_attacked_this_turn.discard(creature_id)
        
        # Add to owner's discard (not controller's)
        state.discard_piles[creature.owner].append(creature.card)
        
        # Update indices
        state.update_creature_indices()
        
        # Trigger defeated ability
        MindbugEngine._resolve_defeated_ability(
            state, creature.card, creature.controller
        )

    @staticmethod
    def _resolve_play_ability(state: GameState, card: Card, controller: Player):
        # Resolve creature's play ability
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
                    targets.append(i)
            
            if targets:
                idx = random.choice(targets)
                creature = state.play_areas[opponent].pop(idx)
                creature.controller = controller
                state.play_areas[controller].append(creature)
                state.update_creature_indices()
                
        elif card.name == "Compost Dragon":
            # Play random card from discard
            if state.discard_piles[controller]:
                card_to_play = random.choice(state.discard_piles[controller])
                state.discard_piles[controller].remove(card_to_play)
                creature = CreatureState(
                    card=card_to_play, controller=controller, owner=controller
                )
                state.play_areas[controller].append(creature)
                state.update_creature_indices()
                
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
                    card=card_to_play, controller=controller, owner=opponent
                )
                state.play_areas[controller].append(creature)
                state.update_creature_indices()
                
        elif card.name == "Kangasaurus Rex":
            # Defeat all power 4 or less
            to_defeat = []
            for i, c in enumerate(state.play_areas[opponent]):
                power = c.get_effective_power(
                    state.current_player == opponent, state.play_areas[opponent]
                )
                if power <= 4:
                    to_defeat.append(i)
            
            # Defeat in reverse order
            for idx in reversed(to_defeat):
                MindbugEngine._defeat_creature(state, opponent, idx)
                
        elif card.name == "Killer Bee":
            state.life[opponent] -= 1
            
        elif card.name == "Mysterious Mermaid":
            state.life[controller] = state.life[opponent]
            
        elif card.name == "Tiger Squirrel":
            # Defeat power 7+ creature
            targets = []
            for i, c in enumerate(state.play_areas[opponent]):
                power = c.get_effective_power(
                    state.current_player == opponent, state.play_areas[opponent]
                )
                if power >= 7:
                    targets.append(i)
            
            if targets:
                idx = random.choice(targets)
                MindbugEngine._defeat_creature(state, opponent, idx)

    @staticmethod
    def _resolve_attack_ability(state: GameState, card: Card, controller: Player):
        # Resolve creature's attack ability
        if card.ability_trigger != TriggerType.ATTACK:
            return
        
        opponent = controller.other()
        
        if card.name == "Chameleon Sniper":
            state.life[opponent] -= 1
            
        elif card.name == "Shark Dog":
            # Defeat power 6+ creature
            targets = []
            for i, c in enumerate(state.play_areas[opponent]):
                power = c.get_effective_power(
                    state.current_player == opponent, state.play_areas[opponent]
                )
                if power >= 6:
                    targets.append(i)
            
            if targets:
                idx = random.choice(targets)
                MindbugEngine._defeat_creature(state, opponent, idx)
                
        elif card.name == "Snail Hydra":
            # If fewer creatures, defeat one
            if len(state.play_areas[controller]) < len(state.play_areas[opponent]):
                if state.play_areas[opponent]:
                    idx = random.randint(0, len(state.play_areas[opponent]) - 1)
                    MindbugEngine._defeat_creature(state, opponent, idx)
                    
        elif card.name == "Turbo Bug":
            state.life[opponent] = 1
            
        elif card.name == "Tusked Extorter":
            state.discard_random(opponent, 1)

    @staticmethod
    def _resolve_defeated_ability(state: GameState, card: Card, controller: Player):
        # Resolve creature's defeated ability
        if card.ability_trigger != TriggerType.DEFEATED:
            return
        
        opponent = controller.other()
        
        if card.name == "Explosive Toad":
            # Defeat random creature
            all_creatures = [
                (p, i)
                for p in [Player.PLAYER_1, Player.PLAYER_2]
                for i in range(len(state.play_areas[p]))
            ]
            if all_creatures:
                player, idx = random.choice(all_creatures)
                MindbugEngine._defeat_creature(state, player, idx)
                
        elif card.name == "Harpy Mother":
            # Take control of up to 2 power 5 or less creatures
            targets = []
            for i, c in enumerate(state.play_areas[opponent]):
                power = c.get_effective_power(
                    state.current_player == opponent, state.play_areas[opponent]
                )
                if power <= 5:
                    targets.append((i, c))
            
            # Take up to 2 random targets
            random.shuffle(targets)
            creatures_taken = []
            
            for idx, creature in targets[:2]:
                creatures_taken.append(creature)
            
            # Move creatures to controller
            for creature in creatures_taken:
                state.play_areas[opponent].remove(creature)
                creature.controller = controller
                state.play_areas[controller].append(creature)
            
            state.update_creature_indices()
            
        elif card.name == "Strange Barrel":
            # Steal 2 random cards from hand
            cards_to_steal = min(2, len(state.hands[opponent]))
            for _ in range(cards_to_steal):
                if state.hands[opponent]:
                    card = random.choice(state.hands[opponent])
                    state.hands[opponent].remove(card)
                    state.hands[controller].append(card)