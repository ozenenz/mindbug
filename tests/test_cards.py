"""Comprehensive tests for all card implementations."""

import pytest

from mindbug.core import (
    Action,
    ActionType,
    Card,
    CardDatabase,
    CreatureState,
    GameState,
    Keyword,
    MindbugEngine,
    Player,
    TriggerType,
)


class TestCardDatabase:
    """Test card definitions."""

    def test_all_cards_present(self):
        """Verify all 32 unique cards are defined."""
        cards = CardDatabase.get_all_cards()
        assert len(cards) == 32

        # Check specific cards exist
        expected_cards = [
            "Axolotl Healer",
            "Bee Bear",
            "Brain Fly",
            "Chameleon Sniper",
            "Compost Dragon",
            "Deathweaver",
            "Elephantopus",
            "Explosive Toad",
            "Ferret Bomber",
            "Giraffodile",
            "Goblin Werewolf",
            "Gorillion",
            "Grave Robber",
            "Harpy Mother",
            "Kangasaurus Rex",
            "Killer Bee",
            "Lone Yeti",
            "Luchataur",
            "Mysterious Mermaid",
            "Plated Scorpion",
            "Rhino Turtle",
            "Shark Dog",
            "Sharky Crab-Dog-Mummypus",
            "Shield Bugs",
            "Snail Hydra",
            "Snail Thrower",
            "Spider Owl",
            "Strange Barrel",
            "Tiger Squirrel",
            "Turbo Bug",
            "Tusked Extorter",
            "Urchin Hurler",
        ]

        for name in expected_cards:
            assert name in cards, f"Missing card: {name}"

    def test_deck_composition(self):
        """Verify deck has correct card quantities."""
        deck = CardDatabase.get_first_contact_deck()
        assert len(deck) == 48

        # Count occurrences
        card_counts = {}
        for card in deck:
            card_counts[card.name] = card_counts.get(card.name, 0) + 1

        # Verify single cards
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
            assert card_counts.get(name, 0) == 1, f"{name} should appear once"

        # Verify double cards
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
            assert card_counts.get(name, 0) == 2, f"{name} should appear twice"


class TestPlayAbilities:
    """Test all Play: abilities."""

    def test_axolotl_healer(self):
        """Axolotl Healer: Gain 2 life."""
        state = GameState(current_player=Player.PLAYER_1)
        state.life[Player.PLAYER_1] = 3

        card = CardDatabase.get_card("Axolotl Healer")
        creature = CreatureState(card=card, controller=Player.PLAYER_1, owner=Player.PLAYER_1)
        state.play_areas[Player.PLAYER_1].append(creature)

        MindbugEngine._resolve_play_ability(state, card, Player.PLAYER_1)

        assert state.life[Player.PLAYER_1] == 5

    def test_brain_fly(self):
        """Brain Fly: Take control of power 6+ creature."""
        state = GameState(current_player=Player.PLAYER_1)

        # Add target creature
        target = CardDatabase.get_card("Gorillion")  # 10 power
        creature = CreatureState(card=target, controller=Player.PLAYER_2, owner=Player.PLAYER_2)
        state.play_areas[Player.PLAYER_2].append(creature)

        card = CardDatabase.get_card("Brain Fly")
        MindbugEngine._resolve_play_ability(state, card, Player.PLAYER_1)

        # Should steal the creature
        assert len(state.play_areas[Player.PLAYER_1]) == 1
        assert len(state.play_areas[Player.PLAYER_2]) == 0
        assert state.play_areas[Player.PLAYER_1][0].card.name == "Gorillion"
        assert state.play_areas[Player.PLAYER_1][0].controller == Player.PLAYER_1

    def test_brain_fly_no_valid_target(self):
        """Brain Fly with no valid targets."""
        state = GameState(current_player=Player.PLAYER_1)

        # Add low power creature
        target = CardDatabase.get_card("Chameleon Sniper")  # 1 power
        creature = CreatureState(card=target, controller=Player.PLAYER_2, owner=Player.PLAYER_2)
        state.play_areas[Player.PLAYER_2].append(creature)

        card = CardDatabase.get_card("Brain Fly")
        MindbugEngine._resolve_play_ability(state, card, Player.PLAYER_1)

        # Nothing should happen
        assert len(state.play_areas[Player.PLAYER_1]) == 0
        assert len(state.play_areas[Player.PLAYER_2]) == 1

    def test_compost_dragon(self):
        """Compost Dragon: Play card from discard."""
        state = GameState(current_player=Player.PLAYER_1)

        # Add card to discard
        discarded = CardDatabase.get_card("Gorillion")
        state.discard_piles[Player.PLAYER_1].append(discarded)

        card = CardDatabase.get_card("Compost Dragon")
        MindbugEngine._resolve_play_ability(state, card, Player.PLAYER_1)

        # Should play from discard
        assert len(state.discard_piles[Player.PLAYER_1]) == 0
        assert len(state.play_areas[Player.PLAYER_1]) == 1
        assert state.play_areas[Player.PLAYER_1][0].card.name == "Gorillion"

    def test_ferret_bomber(self):
        """Ferret Bomber: Opponent discards 2."""
        state = GameState(current_player=Player.PLAYER_1)

        # Give opponent cards
        cards = [
            CardDatabase.get_card("Gorillion"),
            CardDatabase.get_card("Luchataur"),
            CardDatabase.get_card("Rhino Turtle"),
        ]
        state.hands[Player.PLAYER_2] = cards

        card = CardDatabase.get_card("Ferret Bomber")
        MindbugEngine._resolve_play_ability(state, card, Player.PLAYER_1)

        # Should discard 2
        assert len(state.hands[Player.PLAYER_2]) == 1
        assert len(state.discard_piles[Player.PLAYER_2]) == 2

    def test_giraffodile(self):
        """Giraffodile: Draw entire discard pile."""
        state = GameState(current_player=Player.PLAYER_1)

        # Setup discard pile
        discarded = [
            CardDatabase.get_card("Gorillion"),
            CardDatabase.get_card("Luchataur"),
        ]
        state.discard_piles[Player.PLAYER_1] = discarded
        state.hands[Player.PLAYER_1] = []

        card = CardDatabase.get_card("Giraffodile")
        MindbugEngine._resolve_play_ability(state, card, Player.PLAYER_1)

        # Should draw all
        assert len(state.hands[Player.PLAYER_1]) == 2
        assert len(state.discard_piles[Player.PLAYER_1]) == 0

    def test_giraffodile_hand_limit(self):
        """Giraffodile with hand size limit."""
        state = GameState(current_player=Player.PLAYER_1)

        # Setup large discard pile
        discarded = [CardDatabase.get_card("Gorillion") for _ in range(8)]
        state.discard_piles[Player.PLAYER_1] = discarded
        state.hands[Player.PLAYER_1] = []

        card = CardDatabase.get_card("Giraffodile")
        MindbugEngine._resolve_play_ability(state, card, Player.PLAYER_1)

        # Should maintain hand limit
        assert len(state.hands[Player.PLAYER_1]) == 5
        assert len(state.discard_piles[Player.PLAYER_1]) == 3

    def test_grave_robber(self):
        """Grave Robber: Play from opponent's discard."""
        state = GameState(current_player=Player.PLAYER_1)

        # Add to opponent's discard
        card_to_rob = CardDatabase.get_card("Gorillion")
        state.discard_piles[Player.PLAYER_2].append(card_to_rob)

        card = CardDatabase.get_card("Grave Robber")
        MindbugEngine._resolve_play_ability(state, card, Player.PLAYER_1)

        # Should steal and play
        assert len(state.discard_piles[Player.PLAYER_2]) == 0
        assert len(state.play_areas[Player.PLAYER_1]) == 1
        assert state.play_areas[Player.PLAYER_1][0].card.name == "Gorillion"
        assert state.play_areas[Player.PLAYER_1][0].owner == Player.PLAYER_2  # Original owner

    def test_kangasaurus_rex(self):
        """Kangasaurus Rex: Defeat all power 4 or less."""
        state = GameState(current_player=Player.PLAYER_1)

        # Add various power creatures
        creatures = [
            ("Chameleon Sniper", 1),  # Should die
            ("Plated Scorpion", 2),  # Should die
            ("Spider Owl", 3),  # Should die
            ("Axolotl Healer", 4),  # Should die
            ("Killer Bee", 5),  # Should survive
            ("Gorillion", 10),  # Should survive
        ]

        for name, _ in creatures:
            card = CardDatabase.get_card(name)
            creature = CreatureState(card=card, controller=Player.PLAYER_2, owner=Player.PLAYER_2)
            state.play_areas[Player.PLAYER_2].append(creature)

        card = CardDatabase.get_card("Kangasaurus Rex")
        MindbugEngine._resolve_play_ability(state, card, Player.PLAYER_1)

        # Check survivors
        assert len(state.play_areas[Player.PLAYER_2]) == 2
        survivors = [c.card.name for c in state.play_areas[Player.PLAYER_2]]
        assert "Killer Bee" in survivors
        assert "Gorillion" in survivors

        # Check defeated went to discard
        assert len(state.discard_piles[Player.PLAYER_2]) == 4

    def test_killer_bee(self):
        """Killer Bee: Opponent loses 1 life."""
        state = GameState(current_player=Player.PLAYER_1)
        state.life[Player.PLAYER_2] = 3

        card = CardDatabase.get_card("Killer Bee")
        MindbugEngine._resolve_play_ability(state, card, Player.PLAYER_1)

        assert state.life[Player.PLAYER_2] == 2

    def test_mysterious_mermaid(self):
        """Mysterious Mermaid: Set life equal to opponent's."""
        state = GameState(current_player=Player.PLAYER_1)
        state.life[Player.PLAYER_1] = 1
        state.life[Player.PLAYER_2] = 3

        card = CardDatabase.get_card("Mysterious Mermaid")
        MindbugEngine._resolve_play_ability(state, card, Player.PLAYER_1)

        assert state.life[Player.PLAYER_1] == 3
        assert state.life[Player.PLAYER_2] == 3

    def test_tiger_squirrel(self):
        """Tiger Squirrel: Defeat power 7+ creature."""
        state = GameState(current_player=Player.PLAYER_1)

        # Add creatures
        low = CardDatabase.get_card("Spider Owl")  # 3 power
        high = CardDatabase.get_card("Gorillion")  # 10 power

        state.play_areas[Player.PLAYER_2].append(
            CreatureState(card=low, controller=Player.PLAYER_2, owner=Player.PLAYER_2)
        )
        state.play_areas[Player.PLAYER_2].append(
            CreatureState(card=high, controller=Player.PLAYER_2, owner=Player.PLAYER_2)
        )

        card = CardDatabase.get_card("Tiger Squirrel")
        MindbugEngine._resolve_play_ability(state, card, Player.PLAYER_1)

        # High power defeated
        assert len(state.play_areas[Player.PLAYER_2]) == 1
        assert state.play_areas[Player.PLAYER_2][0].card.name == "Spider Owl"
        assert len(state.discard_piles[Player.PLAYER_2]) == 1


class TestAttackAbilities:
    """Test all Attack: abilities."""

    def test_chameleon_sniper(self):
        """Chameleon Sniper: Opponent loses 1 life."""
        state = GameState(current_player=Player.PLAYER_1)
        state.life[Player.PLAYER_2] = 3

        card = CardDatabase.get_card("Chameleon Sniper")
        MindbugEngine._resolve_attack_ability(state, card, Player.PLAYER_1)

        assert state.life[Player.PLAYER_2] == 2

    def test_shark_dog(self):
        """Shark Dog: Defeat power 6+ creature."""
        state = GameState(current_player=Player.PLAYER_1)

        # Add creatures
        low = CardDatabase.get_card("Spider Owl")  # 3 power
        high = CardDatabase.get_card("Gorillion")  # 10 power

        state.play_areas[Player.PLAYER_2].extend(
            [
                CreatureState(card=low, controller=Player.PLAYER_2, owner=Player.PLAYER_2),
                CreatureState(card=high, controller=Player.PLAYER_2, owner=Player.PLAYER_2),
            ]
        )

        card = CardDatabase.get_card("Shark Dog")
        MindbugEngine._resolve_attack_ability(state, card, Player.PLAYER_1)

        # High power defeated
        assert len(state.play_areas[Player.PLAYER_2]) == 1
        assert state.play_areas[Player.PLAYER_2][0].card.name == "Spider Owl"

    def test_snail_hydra(self):
        """Snail Hydra: If fewer creatures, defeat one."""
        state = GameState(current_player=Player.PLAYER_1)

        # P1 has fewer creatures
        p1_creature = CardDatabase.get_card("Snail Hydra")
        state.play_areas[Player.PLAYER_1].append(
            CreatureState(card=p1_creature, controller=Player.PLAYER_1, owner=Player.PLAYER_1)
        )

        # P2 has more
        for _ in range(3):
            card = CardDatabase.get_card("Spider Owl")
            state.play_areas[Player.PLAYER_2].append(
                CreatureState(card=card, controller=Player.PLAYER_2, owner=Player.PLAYER_2)
            )

        card = CardDatabase.get_card("Snail Hydra")
        MindbugEngine._resolve_attack_ability(state, card, Player.PLAYER_1)

        # Should defeat one
        assert len(state.play_areas[Player.PLAYER_2]) == 2
        assert len(state.discard_piles[Player.PLAYER_2]) == 1

    def test_snail_hydra_equal_creatures(self):
        """Snail Hydra with equal creatures."""
        state = GameState(current_player=Player.PLAYER_1)

        # Equal creatures
        for player in [Player.PLAYER_1, Player.PLAYER_2]:
            for _ in range(2):
                card = CardDatabase.get_card("Spider Owl")
                state.play_areas[player].append(
                    CreatureState(card=card, controller=player, owner=player)
                )

        card = CardDatabase.get_card("Snail Hydra")
        MindbugEngine._resolve_attack_ability(state, card, Player.PLAYER_1)

        # Nothing happens
        assert len(state.play_areas[Player.PLAYER_2]) == 2
        assert len(state.discard_piles[Player.PLAYER_2]) == 0

    def test_turbo_bug(self):
        """Turbo Bug: Opponent loses all life except 1."""
        state = GameState(current_player=Player.PLAYER_1)
        state.life[Player.PLAYER_2] = 3

        card = CardDatabase.get_card("Turbo Bug")
        MindbugEngine._resolve_attack_ability(state, card, Player.PLAYER_1)

        assert state.life[Player.PLAYER_2] == 1

    def test_turbo_bug_already_at_one(self):
        """Turbo Bug when opponent already at 1 life."""
        state = GameState(current_player=Player.PLAYER_1)
        state.life[Player.PLAYER_2] = 1

        card = CardDatabase.get_card("Turbo Bug")
        MindbugEngine._resolve_attack_ability(state, card, Player.PLAYER_1)

        assert state.life[Player.PLAYER_2] == 1

    def test_tusked_extorter(self):
        """Tusked Extorter: Opponent discards 1."""
        state = GameState(current_player=Player.PLAYER_1)

        # Give opponent cards
        cards = [
            CardDatabase.get_card("Gorillion"),
            CardDatabase.get_card("Luchataur"),
        ]
        state.hands[Player.PLAYER_2] = cards

        card = CardDatabase.get_card("Tusked Extorter")
        MindbugEngine._resolve_attack_ability(state, card, Player.PLAYER_1)

        assert len(state.hands[Player.PLAYER_2]) == 1
        assert len(state.discard_piles[Player.PLAYER_2]) == 1


class TestDefeatedAbilities:
    """Test all Defeated: abilities."""

    def test_explosive_toad(self):
        """Explosive Toad: Defeat a creature."""
        state = GameState(current_player=Player.PLAYER_1)

        # Add creatures
        for player in [Player.PLAYER_1, Player.PLAYER_2]:
            card = CardDatabase.get_card("Spider Owl")
            state.play_areas[player].append(
                CreatureState(card=card, controller=player, owner=player)
            )

        card = CardDatabase.get_card("Explosive Toad")
        MindbugEngine._resolve_defeated_ability(state, card, Player.PLAYER_1)

        # Should defeat one creature
        total = len(state.play_areas[Player.PLAYER_1]) + len(state.play_areas[Player.PLAYER_2])
        assert total == 1

    def test_harpy_mother(self):
        """Harpy Mother: Take control of up to 2 power 5 or less."""
        state = GameState(current_player=Player.PLAYER_1)

        # Add various creatures to opponent
        creatures = [
            ("Chameleon Sniper", 1),  # Valid
            ("Spider Owl", 3),  # Valid
            ("Killer Bee", 5),  # Valid
            ("Gorillion", 10),  # Too strong
        ]

        for name, _ in creatures:
            card = CardDatabase.get_card(name)
            state.play_areas[Player.PLAYER_2].append(
                CreatureState(card=card, controller=Player.PLAYER_2, owner=Player.PLAYER_2)
            )

        card = CardDatabase.get_card("Harpy Mother")
        MindbugEngine._resolve_defeated_ability(state, card, Player.PLAYER_1)

        # Should take 2 creatures
        assert len(state.play_areas[Player.PLAYER_1]) == 2
        assert len(state.play_areas[Player.PLAYER_2]) == 2

        # Gorillion should remain with opponent
        opponent_names = [c.card.name for c in state.play_areas[Player.PLAYER_2]]
        assert "Gorillion" in opponent_names

    def test_strange_barrel(self):
        """Strange Barrel: Steal 2 cards from opponent's hand."""
        state = GameState(current_player=Player.PLAYER_1)

        # Give opponent cards
        cards = [
            CardDatabase.get_card("Gorillion"),
            CardDatabase.get_card("Luchataur"),
            CardDatabase.get_card("Rhino Turtle"),
        ]
        state.hands[Player.PLAYER_2] = cards
        state.hands[Player.PLAYER_1] = []

        card = CardDatabase.get_card("Strange Barrel")
        MindbugEngine._resolve_defeated_ability(state, card, Player.PLAYER_1)

        # Should steal 2
        assert len(state.hands[Player.PLAYER_1]) == 2
        assert len(state.hands[Player.PLAYER_2]) == 1


class TestPassiveAbilities:
    """Test all passive abilities."""

    def test_bee_bear_blocking(self):
        """Bee Bear: Can't be blocked by power 6 or less."""
        state = GameState(current_player=Player.PLAYER_1)

        # Bee Bear attacking
        bee_bear = CardDatabase.get_card("Bee Bear")
        attacker = CreatureState(card=bee_bear, controller=Player.PLAYER_1, owner=Player.PLAYER_1)
        state.play_areas[Player.PLAYER_1].append(attacker)

        # Various blockers
        blockers = [
            ("Chameleon Sniper", 1, False),  # Can't block
            ("Killer Bee", 5, False),  # Can't block
            ("Strange Barrel", 6, False),  # Can't block
            ("Kangasaurus Rex", 7, True),  # Can block
            ("Gorillion", 10, True),  # Can block
        ]

        for name, _, can_block in blockers:
            state.play_areas[Player.PLAYER_2] = [
                CreatureState(
                    card=CardDatabase.get_card(name),
                    controller=Player.PLAYER_2,
                    owner=Player.PLAYER_2,
                )
            ]

            valid = MindbugEngine._get_valid_blockers(state, attacker, Player.PLAYER_1)

            if can_block:
                assert len(valid) == 1, f"{name} should be able to block"
            else:
                assert len(valid) == 0, f"{name} should NOT be able to block"

    def test_deathweaver_blocks_play_effects(self):
        """Deathweaver: Opponent can't activate Play effects."""
        state = GameState(current_player=Player.PLAYER_1)

        # P2 has Deathweaver
        deathweaver = CardDatabase.get_card("Deathweaver")
        state.play_areas[Player.PLAYER_2].append(
            CreatureState(card=deathweaver, controller=Player.PLAYER_2, owner=Player.PLAYER_2)
        )

        # Update status
        state.update_deathweaver_status()
        assert state.deathweaver_active[Player.PLAYER_1]

        # P1 plays Axolotl Healer
        healer = CardDatabase.get_card("Axolotl Healer")
        state.life[Player.PLAYER_1] = 2

        # Manually trigger what would happen
        creature = CreatureState(card=healer, controller=Player.PLAYER_1, owner=Player.PLAYER_1)
        state.play_areas[Player.PLAYER_1].append(creature)

        # Play ability should be blocked
        if not state.deathweaver_active[Player.PLAYER_1]:
            MindbugEngine._resolve_play_ability(state, healer, Player.PLAYER_1)

        assert state.life[Player.PLAYER_1] == 2  # No life gain

    def test_elephantopus_blocking_restriction(self):
        """Elephantopus: Opponent can't block with power 4 or less."""
        state = GameState(current_player=Player.PLAYER_1)

        # P1 has Elephantopus and attacker
        elephantopus = CardDatabase.get_card("Elephantopus")
        attacker_card = CardDatabase.get_card("Spider Owl")

        state.play_areas[Player.PLAYER_1].extend(
            [
                CreatureState(card=elephantopus, controller=Player.PLAYER_1, owner=Player.PLAYER_1),
                CreatureState(
                    card=attacker_card, controller=Player.PLAYER_1, owner=Player.PLAYER_1
                ),
            ]
        )

        attacker = state.play_areas[Player.PLAYER_1][1]

        # Various blockers
        blockers = [
            ("Chameleon Sniper", 1, False),  # Can't block
            ("Axolotl Healer", 4, False),  # Can't block
            ("Killer Bee", 5, True),  # Can block
            ("Gorillion", 10, True),  # Can block
        ]

        for name, _, can_block in blockers:
            state.play_areas[Player.PLAYER_2] = [
                CreatureState(
                    card=CardDatabase.get_card(name),
                    controller=Player.PLAYER_2,
                    owner=Player.PLAYER_2,
                )
            ]

            valid = MindbugEngine._get_valid_blockers(state, attacker, Player.PLAYER_1)

            if can_block:
                assert len(valid) == 1, f"{name} should be able to block"
            else:
                assert len(valid) == 0, f"{name} should NOT be able to block"

    def test_goblin_werewolf_power_boost(self):
        """Goblin Werewolf: +6 power on controller's turn."""
        state = GameState(current_player=Player.PLAYER_1)

        card = CardDatabase.get_card("Goblin Werewolf")
        creature = CreatureState(card=card, controller=Player.PLAYER_1, owner=Player.PLAYER_1)
        state.play_areas[Player.PLAYER_1].append(creature)

        # On controller's turn
        power_active = creature.get_effective_power(True, state.play_areas[Player.PLAYER_1])
        assert power_active == 8  # 2 + 6

        # Not controller's turn
        power_inactive = creature.get_effective_power(False, state.play_areas[Player.PLAYER_1])
        assert power_inactive == 2

    def test_lone_yeti_boost(self):
        """Lone Yeti: +5 power and FRENZY when alone."""
        state = GameState(current_player=Player.PLAYER_1)

        card = CardDatabase.get_card("Lone Yeti")
        creature = CreatureState(card=card, controller=Player.PLAYER_1, owner=Player.PLAYER_1)
        state.play_areas[Player.PLAYER_1].append(creature)

        # When alone
        power = creature.get_effective_power(True, state.play_areas[Player.PLAYER_1])
        keywords = creature.get_effective_keywords(
            state.play_areas[Player.PLAYER_1], state.play_areas[Player.PLAYER_2]
        )

        assert power == 10  # 5 + 5
        assert Keyword.FRENZY in keywords
        assert Keyword.TOUGH in keywords

        # With ally
        ally = CardDatabase.get_card("Spider Owl")
        state.play_areas[Player.PLAYER_1].append(
            CreatureState(card=ally, controller=Player.PLAYER_1, owner=Player.PLAYER_1)
        )

        power = creature.get_effective_power(True, state.play_areas[Player.PLAYER_1])
        keywords = creature.get_effective_keywords(
            state.play_areas[Player.PLAYER_1], state.play_areas[Player.PLAYER_2]
        )

        assert power == 5
        assert Keyword.FRENZY not in keywords
        assert Keyword.TOUGH in keywords

    def test_sharky_crab_dog_mummypus(self):
        """Sharky Crab-Dog-Mummypus: Copy enemy keywords."""
        state = GameState(current_player=Player.PLAYER_1)

        # Sharky
        sharky = CardDatabase.get_card("Sharky Crab-Dog-Mummypus")
        creature = CreatureState(card=sharky, controller=Player.PLAYER_1, owner=Player.PLAYER_1)
        state.play_areas[Player.PLAYER_1].append(creature)

        # Enemy with keywords
        enemy = CardDatabase.get_card("Spider Owl")  # SNEAKY, POISONOUS
        state.play_areas[Player.PLAYER_2].append(
            CreatureState(card=enemy, controller=Player.PLAYER_2, owner=Player.PLAYER_2)
        )

        keywords = creature.get_effective_keywords(
            state.play_areas[Player.PLAYER_1], state.play_areas[Player.PLAYER_2]
        )

        assert Keyword.SNEAKY in keywords
        assert Keyword.POISONOUS in keywords

        # Test with HUNTER
        state.play_areas[Player.PLAYER_2] = [
            CreatureState(
                card=CardDatabase.get_card("Killer Bee"),  # HUNTER
                controller=Player.PLAYER_2,
                owner=Player.PLAYER_2,
            )
        ]

        keywords = creature.get_effective_keywords(
            state.play_areas[Player.PLAYER_1], state.play_areas[Player.PLAYER_2]
        )

        assert Keyword.HUNTER in keywords
        assert Keyword.SNEAKY not in keywords
        assert Keyword.POISONOUS not in keywords

    def test_shield_bugs_buff(self):
        """Shield Bugs: Other allies +1 power."""
        state = GameState(current_player=Player.PLAYER_1)

        # Shield Bugs
        shield_bugs = CardDatabase.get_card("Shield Bugs")
        state.play_areas[Player.PLAYER_1].append(
            CreatureState(card=shield_bugs, controller=Player.PLAYER_1, owner=Player.PLAYER_1)
        )

        # Other creature
        other = CardDatabase.get_card("Spider Owl")  # 3 power
        other_creature = CreatureState(
            card=other, controller=Player.PLAYER_1, owner=Player.PLAYER_1
        )
        state.play_areas[Player.PLAYER_1].append(other_creature)

        # Check buffed power
        power = other_creature.get_effective_power(True, state.play_areas[Player.PLAYER_1])
        assert power == 4  # 3 + 1

        # Shield Bugs itself not buffed
        shield_creature = state.play_areas[Player.PLAYER_1][0]
        power = shield_creature.get_effective_power(True, state.play_areas[Player.PLAYER_1])
        assert power == 4  # Base power, no self-buff

    def test_snail_thrower_grants_keywords(self):
        """Snail Thrower: Power 4 or less get HUNTER and POISONOUS."""
        state = GameState(current_player=Player.PLAYER_1)

        # Snail Thrower
        snail_thrower = CardDatabase.get_card("Snail Thrower")
        state.play_areas[Player.PLAYER_1].append(
            CreatureState(card=snail_thrower, controller=Player.PLAYER_1, owner=Player.PLAYER_1)
        )

        # Low power creature
        low = CardDatabase.get_card("Spider Owl")  # 3 power
        low_creature = CreatureState(card=low, controller=Player.PLAYER_1, owner=Player.PLAYER_1)
        state.play_areas[Player.PLAYER_1].append(low_creature)

        # High power creature
        high = CardDatabase.get_card("Gorillion")  # 10 power
        high_creature = CreatureState(card=high, controller=Player.PLAYER_1, owner=Player.PLAYER_1)
        state.play_areas[Player.PLAYER_1].append(high_creature)

        # Check keywords
        low_keywords = low_creature.get_effective_keywords(
            state.play_areas[Player.PLAYER_1], state.play_areas[Player.PLAYER_2]
        )

        assert Keyword.HUNTER in low_keywords
        assert Keyword.POISONOUS in low_keywords
        assert Keyword.SNEAKY in low_keywords  # Original

        high_keywords = high_creature.get_effective_keywords(
            state.play_areas[Player.PLAYER_1], state.play_areas[Player.PLAYER_2]
        )

        assert Keyword.HUNTER not in high_keywords
        assert Keyword.POISONOUS not in high_keywords

    def test_urchin_hurler_buff(self):
        """Urchin Hurler: Other allies +2 power on your turn."""
        state = GameState(current_player=Player.PLAYER_1)

        # Urchin Hurler
        urchin = CardDatabase.get_card("Urchin Hurler")
        state.play_areas[Player.PLAYER_1].append(
            CreatureState(card=urchin, controller=Player.PLAYER_1, owner=Player.PLAYER_1)
        )

        # Other creature
        other = CardDatabase.get_card("Spider Owl")  # 3 power
        other_creature = CreatureState(
            card=other, controller=Player.PLAYER_1, owner=Player.PLAYER_1
        )
        state.play_areas[Player.PLAYER_1].append(other_creature)

        # On controller's turn
        power_active = other_creature.get_effective_power(True, state.play_areas[Player.PLAYER_1])
        assert power_active == 5  # 3 + 2

        # Not controller's turn
        power_inactive = other_creature.get_effective_power(
            False, state.play_areas[Player.PLAYER_1]
        )
        assert power_inactive == 3
