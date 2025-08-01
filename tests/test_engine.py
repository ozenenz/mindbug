"""Comprehensive tests for game engine and rules."""

import pytest

from mindbug.core import (
    Action,
    ActionType,
    CardDatabase,
    CreatureState,
    GameState,
    Keyword,
    MindbugEngine,
    Player,
)


class TestGameSetup:
    """Test game initialization."""

    def test_initial_state(self):
        """Test initial game state setup."""
        state = MindbugEngine.create_initial_state()

        # Life totals
        assert state.life[Player.PLAYER_1] == 3
        assert state.life[Player.PLAYER_2] == 3

        # Mindbugs
        assert state.mindbugs_available[Player.PLAYER_1] == 2
        assert state.mindbugs_available[Player.PLAYER_2] == 2

        # Hands and decks
        assert len(state.hands[Player.PLAYER_1]) == 5
        assert len(state.hands[Player.PLAYER_2]) == 5
        assert len(state.decks[Player.PLAYER_1]) == 5
        assert len(state.decks[Player.PLAYER_2]) == 5

        # Empty zones
        assert len(state.play_areas[Player.PLAYER_1]) == 0
        assert len(state.play_areas[Player.PLAYER_2]) == 0
        assert len(state.discard_piles[Player.PLAYER_1]) == 0
        assert len(state.discard_piles[Player.PLAYER_2]) == 0

        # Turn state
        assert state.current_player in [Player.PLAYER_1, Player.PLAYER_2]
        assert not state.mindbug_decision_pending
        assert not state.hunter_choice_pending

    def test_deterministic_setup(self):
        """Test setup with specific parameters."""
        deck = CardDatabase.get_first_contact_deck()
        state = MindbugEngine.create_initial_state(
            deck=deck, starting_player=Player.PLAYER_2, shuffle=False
        )

        assert state.current_player == Player.PLAYER_2

        # Cards dealt in order (no shuffle)
        all_p1_cards = state.hands[Player.PLAYER_1] + state.decks[Player.PLAYER_1]
        all_p2_cards = state.hands[Player.PLAYER_2] + state.decks[Player.PLAYER_2]

        assert all_p1_cards == deck[:10]
        assert all_p2_cards == deck[10:20]


class TestBasicActions:
    """Test core game actions."""

    def test_play_creature_without_mindbug(self):
        """Test playing a creature when opponent passes."""
        state = MindbugEngine.create_initial_state()
        state.current_player = Player.PLAYER_1

        # Get a creature to play
        creature_card = state.hands[Player.PLAYER_1][0]
        initial_hand_size = len(state.hands[Player.PLAYER_1])

        # Play creature
        action = Action(ActionType.PLAY_CREATURE, Player.PLAYER_1, card=creature_card)
        state = MindbugEngine.apply_action(state, action)

        # Should enter Mindbug decision
        assert state.mindbug_decision_pending
        assert state.current_player == Player.PLAYER_2
        assert state.pending_creature_play == (Player.PLAYER_1, creature_card)

        # Hand should be refilled
        assert len(state.hands[Player.PLAYER_1]) == 5

        # Opponent passes
        action = Action(ActionType.PASS_MINDBUG, Player.PLAYER_2)
        state = MindbugEngine.apply_action(state, action)

        # Creature in play
        assert len(state.play_areas[Player.PLAYER_1]) == 1
        assert state.play_areas[Player.PLAYER_1][0].card == creature_card

        # Turn passes
        assert state.current_player == Player.PLAYER_2
        assert not state.mindbug_decision_pending

    def test_play_creature_with_mindbug(self):
        """Test creature being stolen by Mindbug."""
        state = MindbugEngine.create_initial_state()
        state.current_player = Player.PLAYER_1

        creature_card = state.hands[Player.PLAYER_1][0]

        # Play creature
        action = Action(ActionType.PLAY_CREATURE, Player.PLAYER_1, card=creature_card)
        state = MindbugEngine.apply_action(state, action)

        # Opponent uses Mindbug
        action = Action(ActionType.USE_MINDBUG, Player.PLAYER_2)
        state = MindbugEngine.apply_action(state, action)

        # Creature stolen
        assert len(state.play_areas[Player.PLAYER_1]) == 0
        assert len(state.play_areas[Player.PLAYER_2]) == 1
        assert state.play_areas[Player.PLAYER_2][0].card == creature_card
        assert state.play_areas[Player.PLAYER_2][0].controller == Player.PLAYER_2
        assert state.play_areas[Player.PLAYER_2][0].owner == Player.PLAYER_1

        # Mindbug spent
        assert state.mindbugs_available[Player.PLAYER_2] == 1

        # Extra turn for Player 1
        assert state.current_player == Player.PLAYER_1
        assert state.extra_turn_pending == Player.PLAYER_1

    def test_attack_direct_damage(self):
        """Test attacking with no blockers."""
        state = MindbugEngine.create_initial_state()
        state.current_player = Player.PLAYER_1

        # Add attacker
        attacker = CardDatabase.get_card("Gorillion")
        creature = CreatureState(card=attacker, controller=Player.PLAYER_1, owner=Player.PLAYER_1)
        state.play_areas[Player.PLAYER_1].append(creature)

        initial_life = state.life[Player.PLAYER_2]

        # Attack
        action = Action(ActionType.ATTACK, Player.PLAYER_1, creature_index=0)
        state = MindbugEngine.apply_action(state, action)

        # Direct damage
        assert state.life[Player.PLAYER_2] == initial_life - 1
        assert state.current_player == Player.PLAYER_2

    def test_attack_with_blocker(self):
        """Test combat between creatures."""
        state = MindbugEngine.create_initial_state()
        state.current_player = Player.PLAYER_1

        # Add creatures
        attacker_card = CardDatabase.get_card("Gorillion")  # 10 power
        blocker_card = CardDatabase.get_card("Spider Owl")  # 3 power

        state.play_areas[Player.PLAYER_1].append(
            CreatureState(card=attacker_card, controller=Player.PLAYER_1, owner=Player.PLAYER_1)
        )
        state.play_areas[Player.PLAYER_2].append(
            CreatureState(card=blocker_card, controller=Player.PLAYER_2, owner=Player.PLAYER_2)
        )

        # Attack
        action = Action(ActionType.ATTACK, Player.PLAYER_1, creature_index=0)
        state = MindbugEngine.apply_action(state, action)

        # Blocker defeated
        assert len(state.play_areas[Player.PLAYER_1]) == 1
        assert len(state.play_areas[Player.PLAYER_2]) == 0
        assert len(state.discard_piles[Player.PLAYER_2]) == 1

        # No damage to player
        assert state.life[Player.PLAYER_2] == 3


class TestCombatMechanics:
    """Test combat resolution and keywords."""

    def test_equal_power_mutual_destruction(self):
        """Test creatures with equal power destroy each other."""
        state = GameState(current_player=Player.PLAYER_1)

        # Two 5-power creatures
        for player in [Player.PLAYER_1, Player.PLAYER_2]:
            card = CardDatabase.get_card("Killer Bee")
            state.play_areas[player].append(
                CreatureState(card=card, controller=player, owner=player)
            )

        attacker = state.play_areas[Player.PLAYER_1][0]
        MindbugEngine._resolve_combat(state, attacker, Player.PLAYER_1, 0)

        # Both defeated
        assert len(state.play_areas[Player.PLAYER_1]) == 0
        assert len(state.play_areas[Player.PLAYER_2]) == 0
        assert len(state.discard_piles[Player.PLAYER_1]) == 1
        assert len(state.discard_piles[Player.PLAYER_2]) == 1

    def test_poisonous_always_defeats(self):
        """Test POISONOUS keyword."""
        state = GameState(current_player=Player.PLAYER_1)

        # Small poisonous vs large creature
        poisonous = CardDatabase.get_card("Chameleon Sniper")  # 1 power, POISONOUS
        big = CardDatabase.get_card("Gorillion")  # 10 power

        state.play_areas[Player.PLAYER_1].append(
            CreatureState(card=poisonous, controller=Player.PLAYER_1, owner=Player.PLAYER_1)
        )
        state.play_areas[Player.PLAYER_2].append(
            CreatureState(card=big, controller=Player.PLAYER_2, owner=Player.PLAYER_2)
        )

        attacker = state.play_areas[Player.PLAYER_1][0]
        MindbugEngine._resolve_combat(state, attacker, Player.PLAYER_1, 0)

        # Both defeated (mutual destruction)
        assert len(state.play_areas[Player.PLAYER_1]) == 0
        assert len(state.play_areas[Player.PLAYER_2]) == 0

    def test_tough_survives_first_defeat(self):
        """Test TOUGH keyword."""
        state = GameState(current_player=Player.PLAYER_1)

        # Tough creature vs stronger
        tough = CardDatabase.get_card("Elephantopus")  # 7 power, TOUGH
        strong = CardDatabase.get_card("Gorillion")  # 10 power

        tough_creature = CreatureState(
            card=tough, controller=Player.PLAYER_1, owner=Player.PLAYER_1
        )
        state.play_areas[Player.PLAYER_1].append(tough_creature)
        state.play_areas[Player.PLAYER_2].append(
            CreatureState(card=strong, controller=Player.PLAYER_2, owner=Player.PLAYER_2)
        )

        attacker = state.play_areas[Player.PLAYER_1][0]
        MindbugEngine._resolve_combat(state, attacker, Player.PLAYER_1, 0)

        # Tough survives but exhausted
        assert len(state.play_areas[Player.PLAYER_1]) == 1
        assert state.play_areas[Player.PLAYER_1][0].is_exhausted
        assert len(state.play_areas[Player.PLAYER_2]) == 1

    def test_tough_vs_poisonous(self):
        """Test TOUGH prevents first defeat from POISONOUS."""
        state = GameState(current_player=Player.PLAYER_1)

        # Tough vs Poisonous
        tough = CardDatabase.get_card("Rhino Turtle")  # 8 power, TOUGH, FRENZY
        poisonous = CardDatabase.get_card("Spider Owl")  # 3 power, POISONOUS, SNEAKY

        tough_creature = CreatureState(
            card=tough, controller=Player.PLAYER_1, owner=Player.PLAYER_1
        )
        state.play_areas[Player.PLAYER_1].append(tough_creature)
        state.play_areas[Player.PLAYER_2].append(
            CreatureState(card=poisonous, controller=Player.PLAYER_2, owner=Player.PLAYER_2)
        )

        attacker = state.play_areas[Player.PLAYER_1][0]
        MindbugEngine._resolve_combat(state, attacker, Player.PLAYER_1, 0)

        # Tough exhausted but alive, poisonous defeated
        assert len(state.play_areas[Player.PLAYER_1]) == 1
        assert state.play_areas[Player.PLAYER_1][0].is_exhausted
        assert len(state.play_areas[Player.PLAYER_2]) == 0

    def test_sneaky_blocking_restriction(self):
        """Test SNEAKY can only be blocked by SNEAKY."""
        state = GameState(current_player=Player.PLAYER_1)

        # Sneaky attacker
        sneaky = CardDatabase.get_card("Chameleon Sniper")
        attacker = CreatureState(card=sneaky, controller=Player.PLAYER_1, owner=Player.PLAYER_1)
        state.play_areas[Player.PLAYER_1].append(attacker)

        # Non-sneaky blocker
        normal = CardDatabase.get_card("Gorillion")
        state.play_areas[Player.PLAYER_2].append(
            CreatureState(card=normal, controller=Player.PLAYER_2, owner=Player.PLAYER_2)
        )

        # Check valid blockers
        valid = MindbugEngine._get_valid_blockers(state, attacker, Player.PLAYER_1)
        assert len(valid) == 0

        # Add sneaky blocker
        sneaky_blocker = CardDatabase.get_card("Spider Owl")
        state.play_areas[Player.PLAYER_2].append(
            CreatureState(card=sneaky_blocker, controller=Player.PLAYER_2, owner=Player.PLAYER_2)
        )

        valid = MindbugEngine._get_valid_blockers(state, attacker, Player.PLAYER_1)
        assert len(valid) == 1
        assert valid[0] == 1  # Index of Spider Owl

    def test_hunter_forces_block(self):
        """Test HUNTER keyword."""
        state = MindbugEngine.create_initial_state()
        state.current_player = Player.PLAYER_1

        # Hunter attacker
        hunter = CardDatabase.get_card("Killer Bee")
        state.play_areas[Player.PLAYER_1].append(
            CreatureState(card=hunter, controller=Player.PLAYER_1, owner=Player.PLAYER_1)
        )

        # Potential blockers
        for i in range(2):
            card = CardDatabase.get_card("Spider Owl")
            state.play_areas[Player.PLAYER_2].append(
                CreatureState(card=card, controller=Player.PLAYER_2, owner=Player.PLAYER_2)
            )

        # Attack
        action = Action(ActionType.ATTACK, Player.PLAYER_1, creature_index=0)
        state = MindbugEngine.apply_action(state, action)

        # Should enter hunter choice
        assert state.hunter_choice_pending
        assert len(state.valid_blockers) == 2
        assert state.current_player == Player.PLAYER_1

        # Choose blocker
        action = Action(ActionType.CHOOSE_BLOCKER, Player.PLAYER_1, target_index=0)
        state = MindbugEngine.apply_action(state, action)

        # Combat resolved
        assert not state.hunter_choice_pending
        assert len(state.play_areas[Player.PLAYER_2]) == 1  # One died

    def test_hunter_overrides_restrictions(self):
        """Test HUNTER overrides blocking restrictions."""
        state = GameState(current_player=Player.PLAYER_1)

        # Elephantopus preventing blocks
        elephantopus = CardDatabase.get_card("Elephantopus")
        state.play_areas[Player.PLAYER_1].append(
            CreatureState(card=elephantopus, controller=Player.PLAYER_1, owner=Player.PLAYER_1)
        )

        # Hunter attacker
        hunter = CardDatabase.get_card("Killer Bee")
        attacker = CreatureState(card=hunter, controller=Player.PLAYER_1, owner=Player.PLAYER_1)
        state.play_areas[Player.PLAYER_1].append(attacker)

        # Low power blocker (normally can't block due to Elephantopus)
        low = CardDatabase.get_card("Chameleon Sniper")  # 1 power
        state.play_areas[Player.PLAYER_2].append(
            CreatureState(card=low, controller=Player.PLAYER_2, owner=Player.PLAYER_2)
        )

        # HUNTER can force it to block
        valid = MindbugEngine._get_valid_blockers(state, attacker, Player.PLAYER_1)
        assert len(valid) == 1

    def test_frenzy_two_attacks(self):
        """Test FRENZY allows exactly 2 attacks."""
        state = MindbugEngine.create_initial_state()
        state.current_player = Player.PLAYER_1

        # Frenzy creature
        frenzy = CardDatabase.get_card("Explosive Toad")
        creature = CreatureState(card=frenzy, controller=Player.PLAYER_1, owner=Player.PLAYER_1)
        state.play_areas[Player.PLAYER_1].append(creature)

        # First attack
        actions = state.get_legal_actions()
        attack_actions = [a for a in actions if a.action_type == ActionType.ATTACK]
        assert len(attack_actions) == 1

        state = MindbugEngine.apply_action(state, attack_actions[0])

        # Still Player 1's turn, can attack again
        assert state.current_player == Player.PLAYER_1
        actions = state.get_legal_actions()
        attack_actions = [a for a in actions if a.action_type == ActionType.ATTACK]
        assert len(attack_actions) == 1

        # Second attack
        state = MindbugEngine.apply_action(state, attack_actions[0])

        # Turn ends after second attack
        assert state.current_player == Player.PLAYER_2


class TestMindbugMechanics:
    """Test Mindbug-specific rules."""

    def test_mindbug_extra_turn(self):
        """Test extra turn after being Mindbugged."""
        state = MindbugEngine.create_initial_state()
        state.current_player = Player.PLAYER_1

        # P1 plays creature
        creature = state.hands[Player.PLAYER_1][0]
        action = Action(ActionType.PLAY_CREATURE, Player.PLAYER_1, card=creature)
        state = MindbugEngine.apply_action(state, action)

        # P2 uses Mindbug
        action = Action(ActionType.USE_MINDBUG, Player.PLAYER_2)
        state = MindbugEngine.apply_action(state, action)

        # P1 gets extra turn
        assert state.current_player == Player.PLAYER_1
        assert state.extra_turn_pending == Player.PLAYER_1

        # P1 plays another creature
        if state.hands[Player.PLAYER_1]:
            creature2 = state.hands[Player.PLAYER_1][0]
            action = Action(ActionType.PLAY_CREATURE, Player.PLAYER_1, card=creature2)
            state = MindbugEngine.apply_action(state, action)

            # P2 passes
            action = Action(ActionType.PASS_MINDBUG, Player.PLAYER_2)
            state = MindbugEngine.apply_action(state, action)

            # Extra turn consumed, now P2's turn
            assert state.current_player == Player.PLAYER_2
            assert state.extra_turn_pending is None

    def test_mindbug_ownership_tracking(self):
        """Test creature ownership vs control."""
        state = MindbugEngine.create_initial_state()
        state.current_player = Player.PLAYER_1

        # P1 plays creature
        creature_card = state.hands[Player.PLAYER_1][0]
        action = Action(ActionType.PLAY_CREATURE, Player.PLAYER_1, card=creature_card)
        state = MindbugEngine.apply_action(state, action)

        # P2 uses Mindbug
        action = Action(ActionType.USE_MINDBUG, Player.PLAYER_2)
        state = MindbugEngine.apply_action(state, action)

        # Check ownership
        stolen = state.play_areas[Player.PLAYER_2][0]
        assert stolen.controller == Player.PLAYER_2
        assert stolen.owner == Player.PLAYER_1

        # Defeat the creature
        MindbugEngine._defeat_creature(state, stolen, Player.PLAYER_2)

        # Goes to owner's discard
        assert len(state.discard_piles[Player.PLAYER_1]) == 1
        assert len(state.discard_piles[Player.PLAYER_2]) == 0

    def test_cannot_mindbug_twice(self):
        """Test creature can only be Mindbugged once."""
        # This is implicitly tested by the game flow - once a creature
        # is in play, it can't be Mindbugged again
        state = MindbugEngine.create_initial_state()
        state.current_player = Player.PLAYER_1

        # Add creature already in play
        creature = CardDatabase.get_card("Gorillion")
        state.play_areas[Player.PLAYER_1].append(
            CreatureState(card=creature, controller=Player.PLAYER_1, owner=Player.PLAYER_1)
        )

        # Only attack actions available
        actions = state.get_legal_actions()
        play_actions = [a for a in actions if a.action_type == ActionType.PLAY_CREATURE]
        attack_actions = [a for a in actions if a.action_type == ActionType.ATTACK]

        assert len(attack_actions) == 1
        assert all(a.action_type != ActionType.USE_MINDBUG for a in actions)

    def test_mindbug_with_deathweaver(self):
        """Test Mindbug interaction with Deathweaver."""
        state = MindbugEngine.create_initial_state()
        state.current_player = Player.PLAYER_1

        # P1 has Deathweaver
        deathweaver = CardDatabase.get_card("Deathweaver")
        state.play_areas[Player.PLAYER_1].append(
            CreatureState(card=deathweaver, controller=Player.PLAYER_1, owner=Player.PLAYER_1)
        )

        # P2 plays Axolotl Healer
        healer = CardDatabase.get_card("Axolotl Healer")
        state.hands[Player.PLAYER_2] = [healer]
        state.current_player = Player.PLAYER_2
        state.life[Player.PLAYER_1] = 2

        action = Action(ActionType.PLAY_CREATURE, Player.PLAYER_2, card=healer)
        state = MindbugEngine.apply_action(state, action)

        # P1 uses Mindbug
        action = Action(ActionType.USE_MINDBUG, Player.PLAYER_1)
        state = MindbugEngine.apply_action(state, action)

        # P1 controls healer but Deathweaver blocks own Play effects
        assert len(state.play_areas[Player.PLAYER_1]) == 2
        assert state.life[Player.PLAYER_1] == 2  # No heal

    def test_no_mindbugs_available(self):
        """Test when player has no Mindbugs left."""
        state = MindbugEngine.create_initial_state()
        state.mindbugs_available[Player.PLAYER_2] = 0
        state.current_player = Player.PLAYER_1

        # P1 plays creature
        creature = state.hands[Player.PLAYER_1][0]
        action = Action(ActionType.PLAY_CREATURE, Player.PLAYER_1, card=creature)
        state = MindbugEngine.apply_action(state, action)

        # P2 can only pass
        actions = state.get_legal_actions()
        assert len(actions) == 1
        assert actions[0].action_type == ActionType.PASS_MINDBUG


class TestWinConditions:
    """Test various ways the game can end."""

    def test_life_loss_victory(self):
        """Test winning by reducing opponent to 0 life."""
        state = GameState(current_player=Player.PLAYER_1)
        state.life[Player.PLAYER_2] = 1

        assert not state.is_terminal()

        state.life[Player.PLAYER_2] = 0
        assert state.is_terminal()
        assert state.get_winner() == Player.PLAYER_1

    def test_simultaneous_death_draw(self):
        """Test both players at 0 life is a draw."""
        state = GameState(current_player=Player.PLAYER_1)
        state.life[Player.PLAYER_1] = 0
        state.life[Player.PLAYER_2] = 0

        assert state.is_terminal()
        assert state.get_winner() is None

    def test_no_actions_loss(self):
        """Test losing when no legal actions."""
        state = GameState(current_player=Player.PLAYER_1)

        # Empty hand and no creatures
        state.hands[Player.PLAYER_1] = []
        state.play_areas[Player.PLAYER_1] = []

        assert state.is_terminal()
        assert state.get_winner() == Player.PLAYER_2

    def test_game_not_terminal_with_actions(self):
        """Test game continues with legal actions."""
        state = MindbugEngine.create_initial_state()

        assert not state.is_terminal()
        assert state.get_winner() is None

        # Should have legal actions
        actions = state.get_legal_actions()
        assert len(actions) > 0


class TestSpecialInteractions:
    """Test complex card interactions."""

    def test_shield_bugs_stacking(self):
        """Test multiple Shield Bugs stack."""
        state = GameState(current_player=Player.PLAYER_1)

        # Two Shield Bugs
        shield_bugs = CardDatabase.get_card("Shield Bugs")
        for _ in range(2):
            state.play_areas[Player.PLAYER_1].append(
                CreatureState(card=shield_bugs, controller=Player.PLAYER_1, owner=Player.PLAYER_1)
            )

        # Other creature
        other = CardDatabase.get_card("Spider Owl")  # 3 power
        other_creature = CreatureState(
            card=other, controller=Player.PLAYER_1, owner=Player.PLAYER_1
        )
        state.play_areas[Player.PLAYER_1].append(other_creature)

        # Gets +2 power from both Shield Bugs
        power = other_creature.get_effective_power(True, state.play_areas[Player.PLAYER_1])
        assert power == 5  # 3 + 1 + 1

    def test_sharky_with_lone_yeti(self):
        """Test Sharky copying Lone Yeti's conditional FRENZY."""
        state = GameState(current_player=Player.PLAYER_1)

        # Sharky
        sharky = CardDatabase.get_card("Sharky Crab-Dog-Mummypus")
        state.play_areas[Player.PLAYER_1].append(
            CreatureState(card=sharky, controller=Player.PLAYER_1, owner=Player.PLAYER_1)
        )

        # Lone enemy Yeti
        yeti = CardDatabase.get_card("Lone Yeti")
        state.play_areas[Player.PLAYER_2].append(
            CreatureState(card=yeti, controller=Player.PLAYER_2, owner=Player.PLAYER_2)
        )

        # Sharky should have FRENZY
        keywords = state.play_areas[Player.PLAYER_1][0].get_effective_keywords(
            state.play_areas[Player.PLAYER_1], state.play_areas[Player.PLAYER_2]
        )
        assert Keyword.FRENZY in keywords

        # Add another enemy
        other = CardDatabase.get_card("Spider Owl")
        state.play_areas[Player.PLAYER_2].append(
            CreatureState(card=other, controller=Player.PLAYER_2, owner=Player.PLAYER_2)
        )

        # Yeti loses FRENZY, so does Sharky
        keywords = state.play_areas[Player.PLAYER_1][0].get_effective_keywords(
            state.play_areas[Player.PLAYER_1], state.play_areas[Player.PLAYER_2]
        )
        assert Keyword.FRENZY not in keywords

    def test_grave_robber_ownership(self):
        """Test Grave Robber maintains correct ownership."""
        state = MindbugEngine.create_initial_state()
        state.current_player = Player.PLAYER_1

        # Card in P2's discard
        card = CardDatabase.get_card("Gorillion")
        state.discard_piles[Player.PLAYER_2].append(card)

        # P1 plays Grave Robber
        robber = CardDatabase.get_card("Grave Robber")
        state.play_areas[Player.PLAYER_1].append(
            CreatureState(card=robber, controller=Player.PLAYER_1, owner=Player.PLAYER_1)
        )

        MindbugEngine._resolve_play_ability(state, robber, Player.PLAYER_1)

        # Check stolen creature
        assert len(state.play_areas[Player.PLAYER_1]) == 2
        stolen = next(c for c in state.play_areas[Player.PLAYER_1] if c.card.name == "Gorillion")
        assert stolen.controller == Player.PLAYER_1
        assert stolen.owner == Player.PLAYER_2

        # When defeated, goes to P2's discard
        MindbugEngine._defeat_creature(state, stolen, Player.PLAYER_1)
        assert "Gorillion" in [c.name for c in state.discard_piles[Player.PLAYER_2]]
        assert "Gorillion" not in [c.name for c in state.discard_piles[Player.PLAYER_1]]

    def test_elephantopus_vs_bee_bear(self):
        """Test overlapping blocking restrictions."""
        state = GameState(current_player=Player.PLAYER_1)

        # P1 has Elephantopus
        elephantopus = CardDatabase.get_card("Elephantopus")
        state.play_areas[Player.PLAYER_1].append(
            CreatureState(card=elephantopus, controller=Player.PLAYER_1, owner=Player.PLAYER_1)
        )

        # P1 attacks with Bee Bear
        bee_bear = CardDatabase.get_card("Bee Bear")
        attacker = CreatureState(card=bee_bear, controller=Player.PLAYER_1, owner=Player.PLAYER_1)
        state.play_areas[Player.PLAYER_1].append(attacker)

        # P2 has various creatures
        creatures = [
            ("Spider Owl", 3, False),  # Can't block (both restrictions)
            ("Killer Bee", 5, False),  # Can't block (Bee Bear)
            ("Strange Barrel", 6, False),  # Can't block (Bee Bear)
            ("Kangasaurus Rex", 7, True),  # Can block
        ]

        for name, _, can_block in creatures:
            state.play_areas[Player.PLAYER_2] = [
                CreatureState(
                    card=CardDatabase.get_card(name),
                    controller=Player.PLAYER_2,
                    owner=Player.PLAYER_2,
                )
            ]

            valid = MindbugEngine._get_valid_blockers(state, attacker, Player.PLAYER_1)

            if can_block:
                assert len(valid) == 1
            else:
                assert len(valid) == 0
