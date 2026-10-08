"""Prompt variants to compare. Adding a variant = adding an entry to PROMPTS.

The model only ever sees the review title and text, never the star rating.
"""

OUTPUT_SPEC = """Respond with ONLY a JSON object and no other text, in exactly this form:
{"sentiment": "positive" | "neutral" | "negative",
 "topics": [zero or more of "taste", "price", "packaging", "nutrition_health", "shipping", "freshness_quality"],
 "complaint_summary": "one short sentence describing the main complaint, or an empty string if there is none"}"""

# --- Variant 1: bare-bones instructions ---------------------------------------
V1_BASIC = (
    "You analyze customer reviews of food products. Classify the reviewer's overall "
    "sentiment, list the topics the reviewer discusses, and summarize any complaint.\n\n"
    + OUTPUT_SPEC
)

# --- Variant 2: explicit definitions and rules --------------------------------
V2_DEFINITIONS = (
    "You analyze customer reviews of food products.\n\n"
    "Sentiment (judge from the text itself):\n"
    "- positive: the reviewer is mostly satisfied or recommends the product.\n"
    "- negative: the reviewer is mostly dissatisfied or advises against it.\n"
    "- neutral: mixed, lukewarm, or purely factual with no clear lean.\n\n"
    "Topics (include a topic ONLY if the reviewer explicitly discusses it):\n"
    "- taste: flavor, texture, sweetness, aftertaste.\n"
    "- price: cost, value for money, comparisons to store prices.\n"
    "- packaging: containers, cartons, seals, damage or leaks in the package.\n"
    "- nutrition_health: protein, sugar, calories, ingredients, dietary needs, health effects.\n"
    "- shipping: delivery speed, delivery problems, ordering or seller issues.\n"
    "- freshness_quality: freshness, spoilage, expiration, consistency of quality.\n\n"
    "If no topic applies, use an empty list.\n\n"
    + OUTPUT_SPEC
)

# --- Variant 3: definitions + a few worked examples ---------------------------
# These examples are invented (not taken from the dataset) so they cannot leak
# into your evaluation.
V3_FEWSHOT = (
    V2_DEFINITIONS
    + """

Examples:

Review title: Great but pricey
Review text: The chocolate shake tastes amazing and has plenty of protein, but at this price I can't buy it every week.
Answer: {"sentiment": "positive", "topics": ["taste", "nutrition_health", "price"], "complaint_summary": "Too expensive to buy regularly."}

Review title: Never again
Review text: Arrived two weeks late and the cartons were dented and leaking. Won't order again.
Answer: {"sentiment": "negative", "topics": ["shipping", "packaging"], "complaint_summary": "Late delivery and damaged, leaking cartons."}

Review title: It's fine
Review text: Tastes like regular milk, nothing special. Does what it says.
Answer: {"sentiment": "neutral", "topics": ["taste"], "complaint_summary": ""}"""
)

PROMPTS = {
    "v1_basic": V1_BASIC,
    "v2_definitions": V2_DEFINITIONS,
    "v3_fewshot": V3_FEWSHOT,
}
