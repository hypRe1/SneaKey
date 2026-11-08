"""Random sentences for typing tests.

The original recorder drew from 13 fixed sentences, so the same sentences (and the same key sequences) recurred
across sessions. A classifier can then partly learn the passage instead of the sound. This generator composes
fresh sentences from a small probabilistic grammar:

  * Sentence frames are varied: statements, questions, commands, compound and subordinate clauses, relative clauses,
    fronted time/place phrases and existentials. Optional modifiers (adjectives, adverbs, prepositional phrases)
    are switched on independently, so the same frame yields many different shapes.
  * Tense, modality and person vary per clause, with subject-verb agreement.
  * Each sentence is the best of several candidates: candidates that reuse word trigrams from earlier recordings,
    reuse content words from this test, or repeat this test's recent frames are penalised.
  * Optionally ("coverage" mode), words with under-recorded letters are mildly preferred, to balance the training
    data. The default "natural" mode leaves letter frequencies alone.

Everything is seeded, and the seed, mode and per-sentence frame are stored in the session metadata, so a test's
text can be regenerated exactly.
"""
import random
import re
from collections import Counter

GENERATOR_VERSION = 1

# ---- lexicon ----
# Nouns: (singular, plural). Grouped so verbs and prepositions get plausible partners.
PEOPLE = [("teacher", "teachers"), ("farmer", "farmers"), ("doctor", "doctors"), ("pilot", "pilots"),
          ("neighbour", "neighbours"), ("student", "students"), ("journalist", "journalists"), ("baker", "bakers"),
          ("judge", "judges"), ("captain", "captains"), ("plumber", "plumbers"), ("violinist", "violinists"),
          ("child", "children"), ("stranger", "strangers"), ("engineer", "engineers"), ("waiter", "waiters"),
          ("gardener", "gardeners"), ("cousin", "cousins"), ("lawyer", "lawyers"), ("chef", "chefs"),
          ("visitor", "visitors"), ("mechanic", "mechanics"), ("sculptor", "sculptors"), ("referee", "referees")]
ANIMALS = [("fox", "foxes"), ("zebra", "zebras"), ("puppy", "puppies"), ("owl", "owls"), ("goat", "goats"),
           ("kitten", "kittens"), ("jackal", "jackals"), ("walrus", "walruses"), ("pigeon", "pigeons"),
           ("beaver", "beavers"), ("lizard", "lizards"), ("yak", "yaks"), ("squirrel", "squirrels"), ("moth", "moths")]
THINGS = [("box", "boxes"), ("jacket", "jackets"), ("ladder", "ladders"), ("puzzle", "puzzles"),
          ("kettle", "kettles"), ("wallet", "wallets"), ("bicycle", "bicycles"), ("blanket", "blankets"),
          ("quilt", "quilts"), ("lantern", "lanterns"), ("parcel", "parcels"), ("violin", "violins"),
          ("map", "maps"), ("envelope", "envelopes"), ("hammer", "hammers"), ("basket", "baskets"),
          ("mirror", "mirrors"), ("notebook", "notebooks"), ("jar", "jars"), ("compass", "compasses"),
          ("ticket", "tickets"), ("umbrella", "umbrellas"), ("keyboard", "keyboards"), ("wax candle", "wax candles"),
          ("radio", "radios"), ("bucket", "buckets"), ("glove", "gloves"), ("zipper", "zippers"),
          ("photograph", "photographs"), ("sandwich", "sandwiches"), ("rope", "ropes"), ("mug", "mugs")]
PLACES = ["the kitchen", "the harbour", "the library", "the market", "the station", "the garden", "the attic",
          "the village square", "the riverbank", "the old quarry", "the museum", "the bakery", "the workshop",
          "the hallway", "the jetty", "the meadow", "the hospital", "the cinema", "the bus depot", "the valley",
          "the back yard", "the factory", "the zoo", "the forest path", "the hotel lobby", "the cellar"]
NAMES = ["Maya", "Felix", "Jonah", "Priya", "Oscar", "Zoe", "Victor", "Hannah", "Kwame", "Lucia", "Quentin",
         "Yusuf", "Ingrid", "Wesley", "Beatrix", "Diego", "Freya", "Xavier", "Nadia", "Theo", "Greta", "Jasper"]
ABSTRACT = [("idea", "ideas"), ("question", "questions"), ("plan", "plans"), ("rumour", "rumours"),
            ("mistake", "mistakes"), ("excuse", "excuses"), ("joke", "jokes"), ("secret", "secrets"),
            ("warning", "warnings"), ("story", "stories"), ("problem", "problems"), ("theory", "theories")]

# Verbs: (base, third person singular, past, past participle, -ing).
TRANSITIVE_THING = [("carry", "carries", "carried", "carried", "carrying"), ("fix", "fixes", "fixed", "fixed", "fixing"),
                    ("drop", "drops", "dropped", "dropped", "dropping"), ("borrow", "borrows", "borrowed", "borrowed", "borrowing"),
                    ("hide", "hides", "hid", "hidden", "hiding"), ("find", "finds", "found", "found", "finding"),
                    ("polish", "polishes", "polished", "polished", "polishing"), ("wrap", "wraps", "wrapped", "wrapped", "wrapping"),
                    ("sell", "sells", "sold", "sold", "selling"), ("paint", "paints", "painted", "painted", "painting"),
                    ("pack", "packs", "packed", "packed", "packing"), ("lose", "loses", "lost", "lost", "losing"),
                    ("weigh", "weighs", "weighed", "weighed", "weighing"), ("squeeze", "squeezes", "squeezed", "squeezed", "squeezing"),
                    ("inspect", "inspects", "inspected", "inspected", "inspecting"), ("examine", "examines", "examined", "examined", "examining"),
                    ("juggle", "juggles", "juggled", "juggled", "juggling"), ("unpack", "unpacks", "unpacked", "unpacked", "unpacking")]
TRANSITIVE_PERSON = [("thank", "thanks", "thanked", "thanked", "thanking"), ("visit", "visits", "visited", "visited", "visiting"),
                     ("warn", "warns", "warned", "warned", "warning"), ("follow", "follows", "followed", "followed", "following"),
                     ("ignore", "ignores", "ignored", "ignored", "ignoring"), ("invite", "invites", "invited", "invited", "inviting"),
                     ("question", "questions", "questioned", "questioned", "questioning"), ("amaze", "amazes", "amazed", "amazed", "amazing"),
                     ("meet", "meets", "met", "met", "meeting"), ("help", "helps", "helped", "helped", "helping"),
                     ("annoy", "annoys", "annoyed", "annoyed", "annoying"), ("recognise", "recognises", "recognised", "recognised", "recognising")]
TRANSITIVE_ABSTRACT = [("explain", "explains", "explained", "explained", "explaining"), ("forget", "forgets", "forgot", "forgotten", "forgetting"),
                       ("reject", "rejects", "rejected", "rejected", "rejecting"), ("share", "shares", "shared", "shared", "sharing"),
                       ("whisper", "whispers", "whispered", "whispered", "whispering"), ("doubt", "doubts", "doubted", "doubted", "doubting"),
                       ("invent", "invents", "invented", "invented", "inventing"), ("mention", "mentions", "mentioned", "mentioned", "mentioning")]
INTRANSITIVE = [("wait", "waits", "waited", "waited", "waiting"), ("laugh", "laughs", "laughed", "laughed", "laughing"),
                ("sleep", "sleeps", "slept", "slept", "sleeping"), ("argue", "argues", "argued", "argued", "arguing"),
                ("wander", "wanders", "wandered", "wandered", "wandering"), ("hesitate", "hesitates", "hesitated", "hesitated", "hesitating"),
                ("vanish", "vanishes", "vanished", "vanished", "vanishing"), ("jog", "jogs", "jogged", "jogged", "jogging"),
                ("shiver", "shivers", "shivered", "shivered", "shivering"), ("complain", "complains", "complained", "complained", "complaining"),
                ("sneeze", "sneezes", "sneezed", "sneezed", "sneezing"), ("relax", "relaxes", "relaxed", "relaxed", "relaxing"),
                ("whistle", "whistles", "whistled", "whistled", "whistling"), ("queue", "queues", "queued", "queued", "queueing")]
ADJECTIVES = ["quiet", "heavy", "bright", "dusty", "fragile", "lazy", "curious", "enormous", "tiny", "crooked",
              "polished", "ancient", "noisy", "brave", "clumsy", "gentle", "jealous", "fuzzy", "wicked", "vivid",
              "awkward", "cheerful", "stubborn", "spotless", "frozen", "golden", "hollow", "patient", "quirky",
              "rusty", "sleepy", "velvet", "yellow", "zealous", "humble", "eager", "muddy", "sharp", "wobbly"]
STATE_ADJECTIVES = ["tired", "nervous", "delighted", "puzzled", "furious", "calm", "anxious", "excited", "bored",
                    "jealous", "grateful", "hungry", "sceptical", "exhausted", "relieved", "amazed"]
ADVERBS = ["quickly", "slowly", "quietly", "carefully", "suddenly", "eagerly", "gently", "boldly", "lazily",
           "nervously", "patiently", "awkwardly", "happily", "rarely", "loudly", "clumsily", "wisely", "barely"]
TIME_PHRASES = ["yesterday", "last week", "before dawn", "every morning", "after lunch", "on Tuesday",
                "at midnight", "in the summer", "twice a year", "later that evening", "during the storm",
                "the next day", "over the weekend", "an hour ago", "in January", "every few minutes"]
CONJUNCTIONS = ["but", "and", "so", "yet"]
SUBORDINATORS = ["because", "although", "while", "when", "until", "unless", "since", "before", "after", "even though"]
PREPOSITIONS = ["near", "behind", "inside", "beside", "under", "above", "across", "towards", "outside", "beyond"]
DETERMINERS_SINGULAR = ["the", "a", "this", "that", "every", "my", "our", "their", "her", "his", "one"]
DETERMINERS_PLURAL = ["the", "some", "these", "those", "several", "many", "few", "two", "three", "our", "my"]
MODALS = ["can", "could", "should", "might", "must", "will", "would"]

PRONOUNS = {  # subject, object, person/number for agreement
    "I": ("I", "me", "1s"), "you": ("you", "you", "2"), "he": ("he", "him", "3s"), "she": ("she", "her", "3s"),
    "we": ("we", "us", "1p"), "they": ("they", "them", "3p"),
}


class _Builder:
    """Composes one sentence. Tracks the content words it used so a test can avoid repeating them."""

    def __init__(self, rng, weight):
        self.rng = rng
        self.weight = weight      # word -> sampling weight multiplier (letter coverage); 1.0 in natural mode
        self.used = []

    # -- choosing words --
    def pick(self, options, key=lambda o: o):
        weights = [self.weight(key(o)) for o in options]
        choice = self.rng.choices(options, weights)[0]
        self.used.append(key(choice))
        return choice

    def chance(self, p):
        return self.rng.random() < p

    # -- noun phrases --
    def noun_phrase(self, kind=None, allow_pronoun=True, allow_name=True, role="subject"):
        """Returns (text, person) where person is used for verb agreement."""
        r = self.rng.random()
        if allow_pronoun and r < 0.18:
            subject, obj, person = PRONOUNS[self.rng.choice(list(PRONOUNS))]
            return (subject if role == "subject" else obj), person
        if allow_name and r < 0.32 and kind in (None, "person"):
            return self.pick(NAMES), "3s"
        kind = kind or self.rng.choice(["person", "person", "animal", "thing"])
        pool = {"person": PEOPLE, "animal": ANIMALS, "thing": THINGS, "abstract": ABSTRACT}[kind]
        singular, plural = self.pick(pool, key=lambda n: n[0])
        is_plural = self.chance(0.35)
        noun = plural if is_plural else singular
        determiner = self.rng.choice(DETERMINERS_PLURAL if is_plural else DETERMINERS_SINGULAR)
        words = [determiner]
        if self.chance(0.4):
            words.append(self.pick(ADJECTIVES))
        words.append(noun)
        if determiner == "a" and words[1][0] in "aeiou":
            words[0] = "an"
        if self.chance(0.12) and kind in ("person", "animal"):
            words += ["from", self.pick(PLACES)]
        return " ".join(words), ("3p" if is_plural else "3s")

    def object_for(self, verb_group):
        kind = {id(TRANSITIVE_THING): "thing", id(TRANSITIVE_PERSON): "person",
                id(TRANSITIVE_ABSTRACT): "abstract"}[id(verb_group)]
        text, _ = self.noun_phrase(kind, allow_pronoun=kind == "person", allow_name=kind == "person", role="object")
        return text

    # -- verb phrases --
    def verb(self, forms, person, tense):
        base, third, past, participle, ing = forms
        be = {"1s": ("am", "was"), "2": ("are", "were"), "3s": ("is", "was")}.get(person, ("are", "were"))
        have = "has" if person == "3s" else "have"
        if tense == "present":
            return third if person == "3s" else base
        if tense == "past":
            return past
        if tense == "progressive":
            return f"{be[0]} {ing}"
        if tense == "past_progressive":
            return f"{be[1]} {ing}"
        if tense == "perfect":
            return f"{have} {participle}"
        if tense == "modal":
            return f"{self.rng.choice(MODALS)} {base}"
        return f"will {base}"

    def predicate(self, person, tense=None):
        tense = tense or self.rng.choice(["present", "past", "past", "progressive", "past_progressive",
                                          "perfect", "modal", "future"])
        r = self.rng.random()
        if r < 0.55:
            group = self.rng.choice([TRANSITIVE_THING, TRANSITIVE_THING, TRANSITIVE_PERSON, TRANSITIVE_ABSTRACT])
            words = [self.verb(self.pick(group, key=lambda v: v[0]), person, tense), self.object_for(group)]
        elif r < 0.85:
            words = [self.verb(self.pick(INTRANSITIVE, key=lambda v: v[0]), person, tense)]
        else:
            copula = {"present": {"1s": "am", "3s": "is"}.get(person, "are"),
                      "past": "was" if person in ("1s", "3s") else "were"}.get(tense, "seemed")
            adjectives = STATE_ADJECTIVES if self.chance(0.6) else ADJECTIVES
            words = [copula, self.pick(adjectives)]
            if self.chance(0.3):
                words.append(f"about the {self.pick(ABSTRACT, key=lambda n: n[0])[0]}")
            return " ".join(words)
        if self.chance(0.35):
            words.append(self.pick(ADVERBS))
        if self.chance(0.4):
            words += [self.rng.choice(PREPOSITIONS), self.pick(PLACES)]
        return " ".join(words)

    def clause(self, tense=None):
        subject, person = self.noun_phrase()
        return f"{subject} {self.predicate(person, tense)}"

    # -- sentence frames --
    def frame(self, name):
        return getattr(self, f"_frame_{name}")()

    def _frame_simple(self):
        return self.clause() + "."

    def _frame_fronted_time(self):
        return f"{self.pick(TIME_PHRASES)}, {self.clause()}."

    def _frame_compound(self):
        return f"{self.clause()}, {self.rng.choice(CONJUNCTIONS)} {self.clause()}."

    def _frame_subordinate_first(self):
        return f"{self.rng.choice(SUBORDINATORS)} {self.clause()}, {self.clause()}."

    def _frame_subordinate_last(self):
        return f"{self.clause()} {self.rng.choice(SUBORDINATORS)} {self.clause()}."

    def _frame_relative(self):
        subject, person = self.noun_phrase(allow_pronoun=False, allow_name=False)
        who = "who" if any(subject.endswith(p[0]) or subject.endswith(p[1]) for p in PEOPLE) else "that"
        inner = self.predicate(person, self.rng.choice(["past", "perfect", "past_progressive"]))
        return f"{subject} {who} {inner} {self.predicate(person)}."

    def _frame_existential(self):
        thing, person = self.noun_phrase(self.rng.choice(["thing", "animal"]), allow_pronoun=False, allow_name=False)
        verb = {"3s": "is", "3p": "are"}[person] if self.chance(0.5) else {"3s": "was", "3p": "were"}[person]
        return f"there {verb} {thing} {self.rng.choice(PREPOSITIONS)} {self.pick(PLACES)}."

    def _frame_question_yes_no(self):
        subject, person = self.noun_phrase()
        group = self.rng.choice([TRANSITIVE_THING, TRANSITIVE_PERSON])
        base = self.pick(group, key=lambda v: v[0])[0]
        auxiliary = self.rng.choice(["did", "will", "could", "should"] + (["does"] if person == "3s" else ["do"]))
        return f"{auxiliary} {subject} {base} {self.object_for(group)}?"

    def _frame_question_wh(self):
        subject, person = self.noun_phrase()
        wh = self.rng.choice(["why", "when", "where", "how"])
        auxiliary = self.rng.choice(["did", "would", "might"] + (["does"] if person == "3s" else ["do"]))
        base = self.pick(INTRANSITIVE, key=lambda v: v[0])[0]
        tail = f" {self.rng.choice(PREPOSITIONS)} {self.pick(PLACES)}" if self.chance(0.4) else ""
        return f"{wh} {auxiliary} {subject} {base}{tail}?"

    def _frame_question_perfect(self):
        subject, person = self.noun_phrase()
        group = self.rng.choice([TRANSITIVE_THING, TRANSITIVE_ABSTRACT])
        forms = self.pick(group, key=lambda v: v[0])
        have = "has" if person == "3s" else "have"
        return f"{have} {subject} ever {forms[3]} {self.object_for(group)}?"

    def _frame_command(self):
        group = self.rng.choice([TRANSITIVE_THING, TRANSITIVE_PERSON])
        words = [self.pick(group, key=lambda v: v[0])[0], self.object_for(group)]
        if self.chance(0.5):
            words += [self.rng.choice(["before", "after", "until"]), self.pick(TIME_PHRASES)
                      if self.chance(0.5) else self.clause("present")]
        prefix = self.rng.choice(["", "", "please ", "never ", "always ", "now "])
        return prefix + " ".join(words) + "."

    def _frame_reported(self):
        subject, person = self.noun_phrase(allow_pronoun=True)
        verb = self.rng.choice(["said", "insisted", "suspected", "noticed", "admitted", "explained", "claimed"])
        return f"{subject} {verb} that {self.clause(self.rng.choice(['past', 'past_progressive', 'perfect', 'modal']))}."


FRAMES = ["simple", "fronted_time", "compound", "subordinate_first", "subordinate_last", "relative",
          "existential", "question_yes_no", "question_wh", "question_perfect", "command", "reported"]


def _finish(sentence):
    sentence = re.sub(r"\s+", " ", sentence).strip()
    return sentence[0].upper() + sentence[1:]


def _trigrams(text):
    """Word trigrams within each clause: sequences across sentence or comma boundaries aren't repeated phrases."""
    grams = set()
    for clause in re.split(r"[.,?!;]", text.lower()):
        words = re.findall(r"[a-z]+", clause)
        grams |= {tuple(words[i:i + 3]) for i in range(len(words) - 2)}
    return grams


def _letter_weight(key_counts, target_per_key=400):
    """Mild preference for words containing under-recorded letters (like the original practice-word mode)."""
    def need(letter):
        presses = key_counts.get(f"Key{letter.upper()}", 0)
        return 1 + 2 * max(0.0, 1 - presses / target_per_key)

    def weight(word):
        letters = [c for c in word.lower() if c.isalpha()]
        return max(need(c) for c in letters) if letters else 1.0
    return weight


def generate_test(target_chars=350, seed=None, mode="natural", previous_texts=(), key_counts=None, candidates=8):
    """Sentences totalling about target_chars characters.

    previous_texts: target texts of earlier recordings, whose word trigrams are avoided.
    Returns {"text", "sentences": [{"text", "frame"}], "seed", "mode", "version", "repeated_trigrams"}.
    """
    seed = random.SystemRandom().randrange(2 ** 32) if seed is None else int(seed)
    rng = random.Random(seed)
    weight = _letter_weight(key_counts or {}) if mode == "coverage" else (lambda word: 1.0)
    seen_trigrams = set()
    for text in previous_texts:
        seen_trigrams |= _trigrams(text)

    sentences, used_words, frame_counts = [], Counter(), Counter()
    total = 0
    while total < target_chars:
        best = None
        for _ in range(candidates):
            builder = _Builder(rng, weight)
            frame = rng.choice(FRAMES)
            text = _finish(builder.frame(frame))
            score = (
                -3.0 * len(_trigrams(text) & seen_trigrams)                 # word sequences seen before
                - 2.0 * sum(used_words[w] for w in builder.used)            # content words reused in this test
                - 2.5 * frame_counts[frame]                                  # this test's frames already used
                - (4.0 if sentences and sentences[-1]["frame"] == frame else 0)
                - 0.02 * abs(len(text) - 70)                                 # prefer moderate sentence lengths
                + rng.random()                                               # tie-breaking noise
            )
            if best is None or score > best[0]:
                best = (score, text, frame, builder.used)
        _, text, frame, used = best
        sentences.append({"text": text, "frame": frame})
        used_words.update(used)
        frame_counts[frame] += 1
        seen_trigrams |= _trigrams(text)
        total += len(text) + 1

    full = " ".join(s["text"] for s in sentences)
    previous = set()
    for text in previous_texts:
        previous |= _trigrams(text)
    return {"text": full, "sentences": sentences, "seed": seed, "mode": mode, "version": GENERATOR_VERSION,
            "repeated_trigrams": len(_trigrams(full) & previous)}
